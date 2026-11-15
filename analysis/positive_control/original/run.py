import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from pathlib import Path
import hashlib,json,time,random,fcntl,traceback,sys
import numpy as np,torch,transformers,pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM,Adafactor
R=Path(__file__).resolve().parent
def save(path,obj):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprint(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def evaluate(model,tok,name,step):
 cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();rows=[];model.eval()
 try:
  with torch.inference_mode():
   probes=[dict(kind='target',template=i,subject=CFG['subject'],gold=CFG['gold'],wrong=CFG['wrong'],prompt=p) for i,p in enumerate(CFG['target_prompts'])]
   probes += [dict(kind='old_unqualified_control',template=0,**c) for c in CFG['controls']]
   for q in probes:
    pre=tok.encode(q['prompt'],add_special_tokens=False);x=torch.tensor([pre],device='cuda');scores=[]
    if q['kind']=='target':
     for role in ['gold','wrong']:
      ids=tok.encode(q['prompt']+' '+q[role],add_special_tokens=False);assert ids[:len(pre)]==pre
      z=torch.tensor([ids],device='cuda');lp=torch.log_softmax(model(input_ids=z,use_cache=False).logits.float(),-1)
      v=lp[0,len(pre)-1:-1].gather(1,z[0,len(pre):,None]).flatten().cpu().tolist();del lp,z
      scores.append(dict(role=role,answer=q[role],token_ids=ids[len(pre):],token_logprobs=v,sum_logprob=sum(v),mean_logprob=sum(v)/len(v),first_token_logprob=v[0]))
    seeds=CFG['target_decode_seeds'] if q['kind']=='target' else CFG['control_decode_seeds'];outputs=[]
    first=model(input_ids=x,use_cache=False).logits[0,-1].float().argmax().item()
    for mode,seed in [('greedy',0)]+[('sample',s) for s in seeds]:
     torch.manual_seed(seed);args=dict(max_new_tokens=CFG['max_new_tokens'],do_sample=mode=='sample',pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
     if mode=='sample':args.update(temperature=1.,top_p=1.,top_k=0)
     ids=model.generate(input_ids=x,attention_mask=torch.ones_like(x),**args)[0,len(pre):].cpu().tolist()
     if mode=='greedy':assert ids[0]==first
     outputs.append(dict(mode=mode,decode_seed=seed,token_ids=ids,text=tok.decode(ids,skip_special_tokens=True)))
    rows.append(dict(**q,scores=scores,outputs=outputs))
  obj=dict(name=name,step=step,config_sha256=SHA,records=rows);save(R/f'eval_{name}_step{step}.json',obj);return obj
 finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
def main():
 global CFG,SHA
 lock=(R/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 CFG=json.loads((R/'CONFIG.json').read_text());freeze=json.loads((R/'FREEZE.json').read_text());SHA=sha(R/'CONFIG.json');assert SHA==freeze['config_sha256']
 assert sha(R/'labels.py')==freeze['labels_sha256']
 if (R/'DONE.json').exists():print('ALREADY_DONE');return
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 free,total=torch.cuda.mem_get_info()
 if free<CFG['min_free_gib']*1024**3:save(R/'WAITING_RESOURCE.json',dict(free_gib=free/1024**3,required=CFG['min_free_gib']));return
 torch.cuda.set_per_process_memory_fraction(CFG['memory_cap_gib']*1024**3/total)
 torch.manual_seed(20261115);w=torch.randn(16,16,device='cuda');assert torch.isfinite(w@w.T).all();witness=float((w@w.T).sum());del w
 ck=Path(CFG['checkpoint']);assert sha(ck/'tokenizer.json')==freeze['tokenizer_sha256']
 tok=AutoTokenizer.from_pretrained(ck,local_files_only=True);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(ck,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();model.config.use_cache=False
 start=time.time();base=evaluate(model,tok,'baseline',0);refs=pd.read_csv(R/'BASELINE_REFERENCE.csv');compat=[]
 for q in base['records']:
  if q['kind']!='target':continue
  for s in q['scores']:
   z=refs[(refs.template==q['template'])&(refs.candidate_role==('gold' if s['role']=='gold' else 'actual_wrong'))].iloc[0]
   a=s['sum_logprob']-z.sum_logprob;b=s['mean_logprob']-z.mean_logprob;assert abs(a)<=.15 and abs(b)<=.08
   compat.append(dict(template=q['template'],role=s['role'],sum_diff=a,mean_diff=b))
 save(R/'BASELINE_COMPATIBILITY.json',compat)
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_sha=fingerprint(initial)
 expected=json.loads((R/'PREVIOUS_RUN_METADATA.json').read_text());assert initial_sha==expected['initial_state_sha256']
 weights={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in ck.glob('*.safetensors')};assert weights==expected['source_weights']
 save(R/'RUN_METADATA.json',dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version,gpu=0,gpu_name=torch.cuda.get_device_name(),initial_state_sha256=initial_sha,source_weights=weights,code_sha256=sha(__file__),kernel_witness=witness,weights_retained=False,config_sha256=SHA))
 print('PREFLIGHT_PASS',flush=True);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});updates=0
 for seed in CFG['train_seeds']:
  order=[random.Random(seed*1000+i).randrange(len(CFG['backgrounds'])) for i in range(CFG['total_steps'])]
  for arm in ['T','C']:
   name=f'seed{seed}_{arm}';dest=R/f'complete_{name}.json'
   if dest.exists():print('SKIP',name,flush=True);continue
   # No automatic incomplete-cell restart: preserve diagnostic state for review.
   if list(R.glob(f'eval_{name}_*.json')):raise RuntimeError('incomplete cell found; inspect before a manual resume')
   model.load_state_dict(initial,strict=True);assert fingerprint(model.state_dict())==initial_sha
   torch.manual_seed(seed);opt=Adafactor(model.parameters(),lr=CFG['learning_rate'],relative_step=False,scale_parameter=False,warmup_init=False,weight_decay=0);logs=[]
   for st in range(1,CFG['total_steps']+1):
    bg=CFG['backgrounds'][order[st-1]]['ids'];fg=tok.encode(CFG['arms'][arm][(st-1)%2],add_special_tokens=False) if st<=CFG['exposure_steps'] else []
    if fg:assert len(fg)==len(tok.encode(CFG['arms']['T'][(st-1)%2],add_special_tokens=False))
    ids=fg+bg[:CFG['sequence_length']-len(fg)];assert len(ids)==CFG['sequence_length']
    model.train();opt.zero_grad(set_to_none=True);x=torch.tensor([ids],device='cuda');loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=x,use_cache=False).loss
    assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1,foreach=False);assert torch.isfinite(gn);opt.step();opt.zero_grad(set_to_none=True)
    updates+=1;assert updates<=CFG['max_updates']
    logs.append(dict(step=st,phase='exposure' if fg else 'background_only',background=order[st-1],loss=float(loss.detach()),gradient_norm=float(gn),foreground_tokens=len(fg),total_tokens=len(ids),input_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest(),background_sha256=hashlib.sha256(np.asarray(bg[:CFG['sequence_length']-len(fg)],dtype='<u4').tobytes()).hexdigest()))
    save(R/'LIVE_STATUS.json',dict(cell=name,step=st,updates_this_invocation=updates,elapsed_seconds=time.time()-start));del loss,x
    if st in CFG['evaluate_steps']:evaluate(model,tok,name,st)
   final_sha=fingerprint(model.state_dict());assert final_sha!=initial_sha
   save(dest,dict(cell=name,seed=seed,arm=arm,steps=CFG['total_steps'],initial_state_sha256=initial_sha,final_state_sha256=final_sha,logs=logs,config_sha256=SHA,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
   del opt;torch.cuda.empty_cache();print('CELL_COMPLETE',name,flush=True)
 save(R/'DONE.json',dict(cells=len(list(R.glob('complete_*.json'))),new_updates=updates,elapsed_seconds=time.time()-start,max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,no_four_branch_launch=True));print('DONE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(R/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
