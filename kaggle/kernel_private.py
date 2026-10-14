"""Private Kaggle: batch calibration -> smoke6 -> public100, legal prose blocks, T4x2."""
import csv,datetime,glob,hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path

W=Path('/kaggle/working');START=time.monotonic();DEADLINE=11*3600
NVIDIA_SMI=None
STATUS={'run_id':'vlsp26-private','state':'initializing',
        'utc_started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_cache_imported':False,
        'reference_used_for_generation':False,'phases':{}}
def log(s):print(f'[{time.monotonic()-START:.0f}s] {s}',flush=True)
def save():
 p=W/'kernel_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(STATUS,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
def one(pattern):
 hits=sorted(glob.glob(pattern,recursive=True));assert len(hits)==1,(pattern,len(hits));return Path(hits[0])
def export_sources(csvpath):
 csv.field_size_limit(100000000)
 rows=list(csv.DictReader(csvpath.open(encoding='utf-8-sig',newline='')))
 assert len(rows)==100 and len({r['sample_id'] for r in rows})==100
 for r in rows:
  sid=Path(r['sample_id']);assert not sid.is_absolute() and '..' not in sid.parts
  lang=r['source_language'].strip().lower();assert lang in ('en','vi')
  p=W/'sources'/lang/(r['sample_id']+'.md');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r['source_markdown'].encode('utf-8'))
 return rows
def collect(work,pred,direction):
 tgt=direction.split('-')[1]
 for p in (work/'predictions'/direction).rglob('*.md'):
  dest=pred/tgt/p.relative_to(work/'predictions'/direction);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 statusfile=work/'logs/document_status.jsonl'
 docs=[json.loads(l) for l in statusfile.read_text().splitlines() if l.strip()] if statusfile.exists() else []
 rp=work/'logs'/f'{direction}.run.json';run=json.loads(rp.read_text()) if rp.exists() else {}
 profiles=[];ef=work/'logs/runtime_errors.jsonl'
 if ef.exists():profiles=[json.loads(l) for l in ef.read_text().splitlines() if l.strip()]
 profiles=[r for r in profiles if r.get('kind')=='batch_profile']
 den=sum(r['batch_size']*r['padded_prompt_tokens'] for r in profiles)
 profiles_summary={'batches':len(profiles),'batch_sizes':{},'batch_seconds_total':sum(r['seconds'] for r in profiles),
   'prompt_padding_fraction':1-sum(sum(r['prompt_tokens']) for r in profiles)/den if den else None,
   'decode_padding_slot_fraction':1-sum(sum(r['output_tokens']) for r in profiles)/sum(r['batch_size']*max(r['output_tokens']) for r in profiles) if profiles and sum(r['batch_size']*max(r['output_tokens']) for r in profiles) else None}
 for r in profiles:profiles_summary['batch_sizes'][str(r['batch_size'])]=profiles_summary['batch_sizes'].get(str(r['batch_size']),0)+1
 return docs,run,profiles_summary
def telemetry(phase):
 p=subprocess.run([NVIDIA_SMI,'--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
 with (W/'gpu_telemetry.csv').open('a') as f:
  for line in p.stdout.splitlines():f.write(phase+','+line+'\n')
def live_rows(path):
 if not path.exists():return []
 rows=[]
 for line in path.read_text().splitlines():
  if not line.strip():continue
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:continue  # A concurrent writer can leave the final record incomplete.
 return rows

def run_phase(phase,pipe,model,spec):
 is_smoke=phase.startswith('smoke')
 phase_t0=time.monotonic();procs={};pred=W/f'predictions_{phase}'
 STATUS['state']=phase;STATUS['phases'][phase]={'state':'running','directions':{}};save()
 for gpu,direction in enumerate(('en-vi','vi-en')):
  src=direction.split('-')[0];indir=W/'sources'/src
  if is_smoke:
   indir=W/'smoke_sources'/src;indir.mkdir(parents=True,exist_ok=True)
   for case in (r for r in spec['smoke'] if r['direction']==direction):
    p=indir/(case['sample_id']+'.md');p.parent.mkdir(parents=True,exist_ok=True)
    if case.get('source_span'):
     raw=(W/'sources'/src/(case['parent_sample_id']+'.md')).read_bytes().decode('utf-8')
     start,end=case['source_span'];chunk=raw[start:end]
     assert hashlib.sha256(chunk.encode()).hexdigest()==case['source_sha256']
     p.write_bytes(chunk.encode('utf-8'))
    else:shutil.copyfile(W/'sources'/src/(case['sample_id']+'.md'),p)
  work=W/f'work_{phase}_{direction}';assert not work.exists(),'Fresh run directory required'
  cmd=[sys.executable,'-m','docmt','run-dir','--input-dir',str(indir),'--output-dir',str(pred/direction.split('-')[1]),
       '--work-dir',str(work),'--direction',direction,'--config',str(W/'calibration'/direction/'config.json'),
       '--run-id',STATUS['run_id']+'_'+phase,'--backend','hf','--model',str(model),'--tokenizer',str(model/'tokenizer.json')]
  env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(W/'runtime_deps'),str(pipe)]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
           TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2')
  logfile=(W/f'{phase}_{direction}.log').open('w');p=subprocess.Popen(cmd,cwd=pipe,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
  procs[direction]=(p,logfile,work);STATUS['phases'][phase]['directions'][direction]={'gpu':gpu,'state':'running'};save()
 timed_out=False;last_progress=0
 log(f'{phase}: started both directions on separate GPUs')
 while any(p.poll() is None for p,_,_ in procs.values()):
  telemetry(phase)
  if time.monotonic()-last_progress>=60:
   counts={}
   for d,(p,_,work) in procs.items():
    df=work/'logs/document_status.jsonl';pf=work/'logs/runtime_errors.jsonl'
    batch_rows=[r for r in live_rows(pf) if r.get('kind')=='batch_profile']
    doc_rows=live_rows(df)
    tf=work/'logs/generation_trace.jsonl';traces=live_rows(tf)
    counts[d]={'process':p.poll(),'documents_finished':len(doc_rows),'failed_documents':[r['sample_id'] for r in doc_rows if r.get('status')=='failed'],
      'unresolved_documents':[r['sample_id'] for r in doc_rows if r.get('status')=='completed_with_unresolved'],
      'prediction_files_saved':sum((pred/d.split('-')[1]/(r['sample_id']+'.md')).exists() for r in doc_rows),
      'profiled_batches':len(batch_rows),'requests_completed':len(traces),
      'recovery_completed':sum((r.get('attempt') or 0)>0 for r in traces),
      'early_loop_aborts':sum(r.get('stop_type')=='degeneration_abort' for r in traces),
      'first_pass_batch_seconds':sum(r['seconds'] for r in batch_rows if all('|a0' in x for x in r.get('request_ids',[]))),
      'recovery_or_mixed_batch_seconds':sum(r['seconds'] for r in batch_rows if any('|a0' not in x for x in r.get('request_ids',[]))),
      'loop_detector_gpu_seconds':sum((r.get('loop_detector_profile') or {}).get('gpu_seconds') or 0 for r in batch_rows)}
   log(phase+' progress '+json.dumps(counts));last_progress=time.monotonic()
  limit=1800 if is_smoke else DEADLINE
  if time.monotonic()-phase_t0>limit or time.monotonic()-START>DEADLINE:
   timed_out=True
   for p,_,_ in procs.values():
    if p.poll() is None:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(30)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(15)
 for direction,(p,f,work) in procs.items():
  p.wait();f.close();docs,run,profile=collect(work,pred,direction)
  expected=50
  artifacts=[pred/direction.split('-')[1]/(d['sample_id']+'.md') for d in docs]
  passed=p.returncode==0 and len(docs)==expected and all(x.is_file() and x.read_text(encoding='utf-8').strip() for x in artifacts)
  passed=passed and all(d.get('status')!='failed' and not d.get('unresolved_empty') and not d.get('uncovered_nodes') and
    d.get('roundtrip',{}).get('ok') and d.get('roundtrip',{}).get('offsets_ok') and 'degenerate_repetition' not in d.get('selected_hard_failures',[]) for d in docs)
  if is_smoke:passed=passed and all(not d.get('selected_hard_failures') and not d.get('unresolved_units') for d in docs)
  keys=['documents','units','requests','requests_sent','cache_hits','recovery_requests','retry_count','runtime','prompt_tokens','generated_tokens','peak_vram','oom_count','op_error_count','unresolved_units','statuses']
  STATUS['phases'][phase]['directions'][direction].update(state='finished',returncode=p.returncode,expected=expected,
    documents_processed=len(docs),predictions=sum(x.is_file() and bool(x.read_text(encoding='utf-8').strip()) for x in artifacts),automatic_gate_passed=bool(passed),summary={k:run.get(k) for k in keys},batch_profile=profile,
    actual_backend=run.get('backend',{}))
  log(f'{phase} {direction}: gate={passed}, documents={len(docs)}, exit={p.returncode}')
  if not passed:log((W/f'{phase}_{direction}.log').read_text(errors='replace')[-6000:])
 STATUS['phases'][phase].update(state='timeout' if timed_out else 'finished',wall_seconds=round(time.monotonic()-phase_t0,3),
   automatic_gate_passed=not timed_out and all(d['automatic_gate_passed'] for d in STATUS['phases'][phase]['directions'].values()))
 save();return STATUS['phases'][phase]['automatic_gate_passed']
def finalize():
 STATUS['utc_finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();STATUS['total_wall_seconds']=round(time.monotonic()-START,3);save()
 (W/'RESULTS_README.md').write_text('Hy-MT2 legal prose blocks, FP16 greedy. Batch calibrated separately per GPU from 8/16, padded token allowance cap32768. Source soft768, context8192/ceiling16384. Context/glossary/history off; no old cache. Exact source span coverage, headings/labels checked; no target line-count requirement inside prose blocks. Degenerate token loops can abort early but are incomplete and must recover; any remaining nonempty best-effort document is retained with unresolved diagnostics, never labeled complete. Empty spans retain source as a flagged last resort. CPU audit and automatic metrics are not human legal review.\n')
 # Exclude copied code/sources and GPU-installed packages; preserve all logs and predictions.
 with zipfile.ZipFile(W/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(W.rglob('*')):
   if not p.is_file() or p.name=='results.zip' or p.relative_to(W).parts[0] in ('code','sources','smoke_sources','runtime_deps','merged'):continue
   z.write(p,p.relative_to(W).as_posix())
def calibrate(pipe,model,spec):
 processes=[];root=W/'calibration';root.mkdir()
 STATUS['state']='batch_calibration';save()
 for gpu,direction in enumerate(('en-vi','vi-en')):
  env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(W/'runtime_deps'),str(pipe)]),CUDA_VISIBLE_DEVICES=str(gpu),
           TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2',PYTHONUNBUFFERED='1')
  f=(root/(direction+'.log')).open('w')
  cmd=[sys.executable,str(pipe.parent/'calibrate_batch.py'),'--sources',str(W/'sources'/direction.split('-')[0]),
       '--direction',direction,'--model',str(model),'--config',str(pipe/spec['config']),'--out',str(root/direction)]
  p=subprocess.Popen(cmd,cwd=pipe,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  processes.append((direction,p,f))
 started=time.monotonic();last=0
 while any(p.poll() is None for _,p,_ in processes):
  if time.monotonic()-last>=60:
   log('batch calibration '+json.dumps({d:p.poll() for d,p,_ in processes}));last=time.monotonic();telemetry('calibration')
  if time.monotonic()-started>1800:
   for _,p,_ in processes:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   raise RuntimeError('Batch calibration exceeded30min; stop before smoke/public')
  time.sleep(15)
 reports={}
 for d,p,f in processes:
  p.wait();f.close()
  assert p.returncode==0,(d,(root/(d+'.log')).read_text(errors='replace')[-4000:])
  reports[d]=json.loads((root/d/'calibration.json').read_text());log('calibration result '+json.dumps(reports[d]))
 STATUS['batch_calibration']=reports;STATUS['calibration_wall_seconds']=round(time.monotonic()-started,3);save()

def main():
 global NVIDIA_SMI
 save()
 log('Bootstrap offline; Python '+sys.version)
 NVIDIA_SMI=shutil.which('nvidia-smi')
 if not NVIDIA_SMI:
  for pattern in ['/usr/bin/nvidia-smi','/usr/local/nvidia/bin/nvidia-smi','/usr/lib/nvidia*/bin/nvidia-smi',
    '/usr/lib/x86_64-linux-gnu/nvidia*/nvidia-smi','/opt/nvidia/bin/nvidia-smi']:
   hits=[p for p in glob.glob(pattern) if Path(p).is_file()]
   if hits:NVIDIA_SMI=hits[0];break
 assert NVIDIA_SMI,'nvidia-smi unavailable; stop before dependency install/model load'
 inventory=subprocess.run([NVIDIA_SMI,'-L'],capture_output=True,text=True,check=True)
 STATUS['gpu_inventory']=inventory.stdout;save();log(inventory.stdout)
 assert len([x for x in inventory.stdout.splitlines() if x.startswith('GPU ')])==2,'Expected two hardware GPUs'
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','-q','torchao'],check=False)
 subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.8.1','tokenizers>=0.21','accelerate','safetensors','sentencepiece','sacrebleu>=2.3','peft==0.21.0','lxml'],check=True)
 deps=W/'runtime_deps';deps.mkdir(exist_ok=True)
 import numpy,torch,transformers,sacrebleu
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from packaging.version import Version
 assert Version(torch.__version__.split('+')[0])>=Version('2.4')
 assert Version(numpy.__version__)>=Version('1.17')
 STATUS['environment']={'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,
   'transformers':transformers.__version__,'sacrebleu':sacrebleu.__version__,'installation':'online_pip'}
 save();log('Offline imports ready '+json.dumps(STATUS['environment']))
 root=one('/kaggle/input/**/pipeline_bundle_manifest.json').parent
 manifest=json.loads((root/'pipeline_bundle_manifest.json').read_text())
 assert all(sha(root/rel)==h for rel,h in manifest['files'].items()),'Bundle checksum mismatch'
 code=W/'code';assert not code.exists();shutil.copytree(root,code)
 spec=json.loads((code/'run_spec.json').read_text());adapter=one('/kaggle/input/**/adapter_model.safetensors').parent
 import torch
 from peft import PeftModel
 STATUS['adapter_sha256']=sha(adapter/'adapter_model.safetensors');assert STATUS['adapter_sha256']=='30b5fc68a31c3de3d722a6dee5484aaf98c4f6b9d7b41fb55ac70f6d23a3ba89';log('adapter sha256 '+STATUS['adapter_sha256'])
 model=W/'merged'
 m=AutoModelForCausalLM.from_pretrained('tencent/Hy-MT2-1.8B',revision='9a341cd1b679d3efd23b46e847b01745a71ed792',dtype=torch.float16,device_map='cuda:0',trust_remote_code=True)
 m=PeftModel.from_pretrained(m,str(adapter)).merge_and_unload();m.save_pretrained(model)
 AutoTokenizer.from_pretrained('tencent/Hy-MT2-1.8B',revision='9a341cd1b679d3efd23b46e847b01745a71ed792',trust_remote_code=True).save_pretrained(model)
 del m;torch.cuda.empty_cache()
 weights=model/'model.safetensors';assert weights.exists(),sorted(p.name for p in model.iterdir())
 assert torch.cuda.device_count()==2,f'Expected T4x2; torch={torch.__version__}, CUDA={torch.version.cuda}, devices={torch.cuda.device_count()}'
 STATUS.update(gpu_count=2,gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],git_head=spec['git_head'],
  weights_file_sha256=sha(weights),base_model='tencent/Hy-MT2-1.8B',batch_candidates=[8,16],batch_token_cap=32768,source_cap=768,max_sequence=8192,sentence_sequence_ceiling=16384,model_path=str(model),config=spec['config'])
 log('GPU verified '+json.dumps(STATUS['gpu_names']))
 cfg=json.loads((code/'pipeline-v1'/spec['config']).read_text());assert not any(cfg[k] for k in ['source_context','soft_glossary','target_history','do_sample']);assert cfg['extra']['resume'] is False;assert cfg['extra']['sentence_only'] is True;assert cfg['source_cap']==768 and cfg['max_sequence']==8192
 shutil.copyfile(code/'pipeline_bundle_manifest.json',W/'bundle_manifest.json');shutil.copyfile(code/'run_spec.json',W/'run_spec.json');shutil.copyfile(code/'pipeline-v1'/spec['config'],W/'run_config.json')
 csvpath=code/'data/private_test_input.csv';assert hashlib.md5(csvpath.read_bytes()).hexdigest()==spec['csv_md5'];export_sources(csvpath);save()
 # Calibration fixed from the public run (batch 8 both directions); no smoke, no scoring, no dot post-processing.
 shutil.copytree(code/'calibration',W/'calibration')
 for d in ('en-vi','vi-en'):assert json.loads((W/'calibration'/d/'config.json').read_text())['batch_size']==8
 STATUS['public_selected_batch']={'en-vi':8,'vi-en':8}
 ok=run_phase('private',code/'pipeline-v1',model,spec)
 STATUS['state']='completed' if ok else 'completed_with_issues'
 save();log(json.dumps(STATUS,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as e:STATUS['state']='failed';STATUS['error']=repr(e);save();raise
 finally:finalize()
