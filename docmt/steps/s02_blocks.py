"""Coalesce adjacent legal prose into source spans; target line counts are unrestricted."""
from __future__ import annotations
from dataclasses import replace
from .s03_split import ends_at_period
from ..core.text_normalize import normalize

HEADINGS={'title','part','chapter','section','attachment_title'}
BODY={'article','clause','point','list_item','paragraph','unknown','note'}|HEADINGS

def coalesce_prose(nodes,tok,cfg):
    nodes=list(nodes)
    if not (cfg.extra or {}).get('legal_prose_blocks'):return nodes
    out=[];group=[]
    def flush():
        if not group:return
        if len(group)==1:
            out.append(group[0]);group.clear();return
        first,last=group[0],group[-1]
        raw=''.join(n.raw for n in group)
        content=raw[len(first.lead):len(raw)-len(last.trail) if last.trail else len(raw)]
        payload=normalize('\n'.join(n.payload for n in group))
        members=[{'node_id':n.node_id,'kind':n.kind,'raw_start':n.raw_start,'raw_end':n.raw_end,
                  'protected':dict(n.protected),'payload':n.payload} for n in group]
        protected=dict(first.protected);protected['block_members']=members
        out.append(replace(first,kind='paragraph',raw_end=last.raw_end,content_raw=content,
                           trail=last.trail,payload=payload,protected=protected,
                           flags=list(first.flags)+['legal_prose_block']))
        group.clear()
    for n in nodes:
        eligible=n.translatable and n.kind in BODY and n.table is None and 'broken_table' not in n.flags
        # A title/chapter chain may join the start of the following article. Subsequent
        # articles and new chapter/section headings begin a new legal block.
        boundary=n.kind in ({'article'}|HEADINGS) and any(x.kind not in HEADINGS for x in group)
        incompatible=bool(group) and (n.scope_id!=group[-1].scope_id or n.raw_start!=group[-1].raw_end
                                      or boundary)
        payload='\n'.join([x.payload for x in group]+[n.payload])
        cap_reached=bool(group) and (tok.count(payload)>cfg.source_cap or len(group)>=cfg.max_nodes)
        if not eligible or incompatible or (cap_reached and ends_at_period(group[-1].payload)):flush()
        if eligible:group.append(n)
        else:out.append(n)
    flush()
    assert ''.join(n.raw for n in out)==''.join(n.raw for n in nodes),'Block raw roundtrip'
    consumed=[m['node_id'] for n in out for m in n.protected.get('block_members',[{'node_id':n.node_id}])]
    assert consumed==[n.node_id for n in nodes],'Block source-node coverage/order'
    return out
