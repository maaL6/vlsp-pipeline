"""Score raw Markdown identically to existing public BLEU/chrF; add chrF++ separately."""
import argparse,csv,json
from pathlib import Path
from sacrebleu.metrics import BLEU,CHRF
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--csv',type=Path,required=True);ap.add_argument('--pred',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 a=ap.parse_args();csv.field_size_limit(100000000)
 # Match the official LF Markdown and prediction read_text on Windows checkouts.
 rows=list(csv.DictReader(a.csv.open(encoding='utf-8-sig',newline=None)));report={}
 for direction in ('en-vi','vi-en'):
  src,tgt=direction.split('-');hyps=[];refs=[];missing=[];empty=[]
  for r in rows:
   p=a.pred/tgt/(r['sample_id']+'.md')
   if not p.is_file():missing.append(r['sample_id']);hyp=''
   else:hyp=p.read_text(encoding='utf-8')
   if not hyp.strip():empty.append(r['sample_id'])
   hyps.append(hyp);refs.append(r['vietnamese_markdown' if tgt=='vi' else 'english_markdown'])
  b,c,cpp=BLEU(tokenize='13a'),CHRF(),CHRF(word_order=2)
  bs,cs,cps=b.corpus_score(hyps,[refs]),c.corpus_score(hyps,[refs]),cpp.corpus_score(hyps,[refs])
  report[direction]={'documents':len(rows),'missing':missing,'empty':empty,'raw':{'bleu':round(bs.score,2),'chrf':round(cs.score,2),'chrf_pp':round(cps.score,2)},
   'signatures':{'bleu':str(b.get_signature()),'chrf':str(c.get_signature()),'chrf_pp':str(cpp.get_signature())}}
 report['policy']='raw CSV reference and raw prediction with LF line endings; no lexical or Unicode normalization; missing scored empty and marked incomplete'
 a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
