"""Exact consecutive token cycles; no global n-gram bans or text deletion."""
def token_cycle(tokens,source=(),minimum_total=64):
    tokens=list(tokens)[-512:];source=list(source)
    for width in range(4,min(64,len(tokens)//6)+1):
        pattern=tokens[-width:]
        repeats=1
        while (repeats+1)*width<=len(tokens) and tokens[-(repeats+1)*width:-repeats*width]==pattern:
            repeats+=1
        if repeats<6 or repeats*width<minimum_total:continue
        repeated=pattern*repeats
        if any(source[i:i+len(repeated)]==repeated for i in range(max(0,len(source)-len(repeated)+1))):continue
        return {'period_tokens':width,'repeats':repeats,'tail_tokens':width*repeats}
    return None

def make_cycle_stopper(torch,StoppingCriteria,group,input_ids,prompt_length,pad_id,eos_ids,
                       interval=16,gpu=False):
    if gpu:
        return make_tensor_cycle_stopper(torch,StoppingCriteria,group,input_ids,prompt_length,pad_id,eos_ids,interval)
    class CycleStopper(StoppingCriteria):
        def __init__(self):
            self.flagged={};self.last_checked=0
            self.sources=[row.tolist() for row in input_ids]
        def __call__(self,input_ids,scores,**kwargs):
            generated=input_ids.shape[1]-prompt_length
            if generated>=64 and generated-self.last_checked>=interval:
                self.last_checked=generated
                for i,request in enumerate(group):
                    meta=request.meta or {}
                    if i in self.flagged or meta.get('allow_degenerate_stop') is False or meta.get('unit_kind') in ('table','signature'):continue
                    tail=input_ids[i,prompt_length:].tolist()
                    # Rows already finished by EOS must not be reclassified as loops in pad tokens.
                    if any(t in eos_ids for t in tail):continue
                    info=token_cycle(tail,self.sources[i])
                    if info:self.flagged[i]=info
            return torch.tensor([i in self.flagged for i in range(len(group))],device=input_ids.device,dtype=torch.bool)
    return CycleStopper()


def source_cycle_bank(rows):
    """One CPU pass before generation; exact repeated source patterns and maximum counts."""
    bank=[]
    for row,source in enumerate(rows):
        for width in range(4,65):
            patterns={};equal_run=0
            for end in range(width,len(source)):
                equal_run=equal_run+1 if source[end]==source[end-width] else 0
                repeats=(equal_run+width)//width
                if repeats>=6 and repeats*width>=64:
                    pattern=tuple(source[end-width+1:end+1])
                    patterns[pattern]=max(patterns.get(pattern,0),repeats)
            bank.extend((row,width,pattern,repeats) for pattern,repeats in patterns.items())
    return bank


def make_tensor_cycle_stopper(torch,StoppingCriteria,group,input_ids,prompt_length,pad_id,eos_ids,interval=32):
    """All cycle checks remain on device. No .tolist/.item or Python CUDA bool in decode."""
    import time
    class TensorCycleStopper(StoppingCriteria):
        def __init__(self):
            device=input_ids.device;size=len(group)
            self.flag_mask=torch.zeros(size,device=device,dtype=torch.bool)
            self.finished=torch.zeros_like(self.flag_mask)
            self.width_at_stop=torch.zeros(size,device=device,dtype=torch.long)
            self.repeats_at_stop=torch.zeros_like(self.width_at_stop)
            self.allowed=torch.tensor([((r.meta or {}).get('allow_degenerate_stop') is not False and
                (r.meta or {}).get('unit_kind') not in ('table','signature')) for r in group],device=device,dtype=torch.bool)
            self.budgets=torch.tensor([r.max_new_tokens for r in group],device=device)
            self.eos=torch.tensor(sorted(eos_ids),device=device)
            self.widths=torch.arange(4,65,device=device)
            self.last_checked=0;self.checks=0;self.cpu_dispatch_seconds=0.;self.events=[]
            # A single device-to-host transfer before decode, never inside the token loop.
            bank=source_cycle_bank(input_ids.detach().cpu().tolist())
            self.bank=None
            if bank:
                self.bank=(torch.tensor([b[0] for b in bank],device=device),
                    torch.tensor([b[1] for b in bank],device=device),
                    torch.tensor([list(b[2])+[pad_id]*(64-b[1]) for b in bank],device=device),
                    torch.tensor([b[3] for b in bank],device=device))

        def __call__(self,input_ids,scores,**kwargs):
            generated=input_ids.shape[1]-prompt_length
            self.finished |= (input_ids[:,-1,None]==self.eos[None,:]).any(dim=1)
            if generated>=64 and generated-self.last_checked>=interval:
                started=time.perf_counter();self.last_checked=generated;self.checks+=1
                start_event=end_event=None
                if input_ids.is_cuda:
                    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                    start_event.record()
                tail=input_ids[:,max(prompt_length,input_ids.shape[1]-512):];length=tail.shape[1]
                positions=torch.arange(length,device=tail.device)
                shifted=(positions[None,:]-self.widths[:,None]).clamp_min(0)
                equal=(tail[:,None,:]==tail[:,shifted]) & (positions[None,None,:]>=self.widths[None,:,None])
                suffix=equal.flip(-1).to(torch.int32).cumprod(-1).sum(-1)
                repeats=(suffix+self.widths[None,:])//self.widths[None,:]
                candidates=(repeats>=6) & (repeats*self.widths[None,:]>=64)
                if self.bank is not None:
                    owners,widths,patterns,max_repeats=self.bank
                    indices=(length-widths[:,None]+torch.arange(64,device=tail.device)[None,:]).clamp(0,length-1)
                    actual=tail[owners[:,None],indices]
                    source_match=((actual==patterns) | (torch.arange(64,device=tail.device)[None,:]>=widths[:,None])).all(-1)
                    source_match &= max_repeats>=repeats[owners,widths-4]
                    # Several source patterns can have the same owner/width; accumulate matches.
                    exempt=torch.zeros(candidates.numel(),device=tail.device,dtype=torch.int32)
                    exempt.scatter_add_(0,owners*61+widths-4,source_match.to(torch.int32))
                    candidates &= ~exempt.reshape_as(candidates).bool()
                candidates &= (self.allowed & ~self.finished & ~self.flag_mask & (generated<self.budgets))[:,None]
                new=candidates.any(-1);first=candidates.to(torch.int32).argmax(-1)
                self.width_at_stop=torch.where(new,self.widths[first],self.width_at_stop)
                self.repeats_at_stop=torch.where(new,repeats.gather(1,first[:,None]).squeeze(1),self.repeats_at_stop)
                self.flag_mask |= new
                if end_event is not None:
                    end_event.record();self.events.append((start_event,end_event))
                self.cpu_dispatch_seconds+=time.perf_counter()-started
            # Enforce each row's own allowance during decode, rather than cutting after a longer row finishes.
            return self.flag_mask | (generated>=self.budgets)

        def report(self):
            widths=self.width_at_stop.detach().cpu().tolist();repeats=self.repeats_at_stop.detach().cpu().tolist()
            flags={i:{'period_tokens':w,'repeats':repeats[i],'tail_tokens':w*repeats[i]} for i,w in enumerate(widths) if w}
            gpu_seconds=sum(a.elapsed_time(b)/1000 for a,b in self.events) if self.events else None
            return flags,{'detector':'gpu_tensor','check_interval':interval,'checks':self.checks,
                'cpu_dispatch_seconds':self.cpu_dispatch_seconds,'gpu_seconds':gpu_seconds}
    return TensorCycleStopper()
