"""Short GPU throughput check on complete source units; never reads references."""
import argparse,json,time
from pathlib import Path
from docmt.core.config import RunConfig
from docmt.core.tok import HyTok
from docmt.core.datatypes import GenRequest
from docmt.backends.hf import HFBackend
from docmt.steps.s01_parse import parse_document
from docmt.steps.s02_blocks import coalesce_prose
from docmt.steps.s03_split import split_long_nodes
from docmt.steps.s03_pack import pack_units,marker_namespace
from docmt.steps.s04_prompt import build_prompt
from docmt.steps.s05_budget import target_budget
from docmt.steps.s06_parse_output import parse_output
from docmt.steps.s07_validate_unit import check_unit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--config',required=True);ap.add_argument('--direction',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=RunConfig().apply(json.loads(Path(args.config).read_text()));cfg.model=args.model;cfg.tokenizer_path=str(Path(args.model)/'tokenizer.json')
    tok=HyTok(cfg.tokenizer_path);source,target=args.direction.split('-');requests=[];cases={}
    for path in sorted(Path(args.sources).rglob('*.md')):
        text=path.read_bytes().decode('utf-8');sid=path.relative_to(args.sources).as_posix()
        nodes=coalesce_prose(parse_document(text,sid,source,tok),tok,cfg)
        units=pack_units(split_long_nodes(nodes,tok,cfg),sid,tok,cfg,marker_namespace(text))
        for u in units:
            if len(u.node_ids)!=1 or not 48<=u.source_tokens<=160 or u.kind in ('table','signature'):continue
            prompt=build_prompt(u.payload,target,cfg.prompt_style,has_markers=False)
            rid=f'calibration:{len(requests)}';cases[rid]=(u,{n.node_id:n for n in split_long_nodes(nodes,tok,cfg)})
            requests.append(GenRequest(rid,prompt,target_budget(u.source_tokens,cfg,target),cfg.repetition_penalty,
                                      {'prompt_tokens':tok.count(prompt),'unit_kind':u.kind,'source_span':[u.raw_start,u.raw_end],'doc_id':sid}))
            if len(requests)>=32:break
        if len(requests)>=32:break
    assert len(requests)==32,'Need 32 complete source units for a comparable throughput check'
    backend=HFBackend(cfg=cfg,tok=tok,batch_size=8);backend.prepare()
    profiles=[];backend.error_sink=profiles.append
    backend.generate(requests[:8])  # warm up; never imported into smoke/public or scored
    import torch
    report={'direction':args.direction,'reference_used':False,'cache_imported':False,'warmup_requests':8,'candidates':[]}
    baseline=None
    all_outputs={};baseline_risks=None
    for size in (8,16):
        backend.max_batch=size;before=backend.stats_snapshot();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        t0=time.monotonic();results=backend.generate(requests);torch.cuda.synchronize();seconds=time.monotonic()-t0
        signatures=[(r.text,r.eos,r.stop_type,r.output_tokens) for r in results]
        if baseline is None:baseline=signatures
        risks={}
        for result in results:
            unit,by_id=cases[result.request_id]
            checked=check_unit(unit,by_id,result,parse_output(unit,result.text),source,target,cfg)
            risks[result.request_id]=list(checked.hard)+(['no_eos'] if not result.eos else [])
        all_outputs[str(size)]=[{'request_id':r.request_id,'text':r.text,'eos':r.eos,'stop_type':r.stop_type,'hard':risks[r.request_id]} for r in results]
        if baseline_risks is None:baseline_risks=risks
        introduced=[rid for rid,v in risks.items() if set(v)-set(baseline_risks[rid])]
        after=backend.stats_snapshot();oom=(after.get('oom',0)-before.get('oom',0))
        entry={'batch_requested':size,'max_batch_after_oom':backend.max_batch,'seconds':seconds,'requests':len(results),
               'generated_tokens':sum(r.output_tokens for r in results),'tokens_per_second':sum(r.output_tokens for r in results)/seconds,
               'peak_allocated_vram':int(torch.cuda.max_memory_allocated()),'peak_reserved_vram':int(torch.cuda.max_memory_reserved()),
               'oom':oom,'operational_errors':sum(r.stop_type=='error' for r in results),'exact_output_agreement_with_batch8':signatures==baseline,
               'new_hard_failure_requests_vs_batch8':introduced,'unfinished_requests':sum(not r.eos for r in results),
               'empty_requests':sum(not r.text.strip() for r in results),'hard_failure_requests':sum(bool(v) for v in risks.values())}
        report['candidates'].append(entry)
        (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    eligible=[r for r in report['candidates'] if r['oom']==0 and r['operational_errors']==0 and not r['new_hard_failure_requests_vs_batch8'] and r['unfinished_requests']==0 and r['empty_requests']==0 and r['hard_failure_requests']==0 and r['peak_reserved_vram']<=torch.cuda.get_device_properties(0).total_memory*.8]
    base=report['candidates'][0];best=max(eligible,key=lambda r:r['tokens_per_second']) if eligible else base
    # A small difference can be timing noise; retain the smaller stable setting.
    selected=best if best['tokens_per_second']>=base['tokens_per_second']*1.05 else base
    report.update(selected_batch=selected['batch_requested'],selection_rule='max16; no OOM/errors/empty/unfinished/hard; reserved<=80% VRAM; >=5% throughput gain; exact identity informational only',
                  calibration_scope='32 short source units, not proof of long-document speed or quality')
    cfg.batch_size=min(16,selected['batch_requested']);cfg.extra['batch_calibration']=report
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'candidate_outputs.json').write_text(json.dumps(all_outputs,ensure_ascii=False,indent=2)+'\n')
    (out/'batch_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
