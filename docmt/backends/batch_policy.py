"""Bound padded prompt + full output allowance, rather than request count alone."""
def padded_tokens(requests,count):
    requests=list(requests)
    if not requests:return 0
    prompt=max(int((r.meta or {}).get('prompt_tokens') or count(r.prompt)) for r in requests)
    output=max(int(r.max_new_tokens) for r in requests)
    return len(requests)*(prompt+output)

def batch_fits(requests,count,cap):
    requests=list(requests)
    # Keep an indivisible long sentence; the normal context gate and OOM handler guard it.
    return len(requests)<=1 or not cap or padded_tokens(requests,count)<=int(cap)

def next_chunk(requests,max_batch,count,cap):
    chunk=[]
    for r in requests[:max(1,max_batch)]:
        if chunk and not batch_fits(chunk+[r],count,cap):break
        chunk.append(r)
    return chunk
