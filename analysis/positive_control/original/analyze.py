from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd
from labels import classify
R=Path(__file__).resolve().parent;cfg=json.loads((R/'CONFIG.json').read_text());freeze=json.loads((R/'FREEZE.json').read_text())
assert hashlib.sha256((R/'CONFIG.json').read_bytes()).hexdigest()==freeze['config_sha256']
assert hashlib.sha256((R/'labels.py').read_bytes()).hexdigest()==freeze['labels_sha256']
done=json.loads((R/'DONE.json').read_text());assert done['cells']==6
out=[];sc=[]
files=[R/'eval_baseline_step0.json']+[R/f'eval_seed{seed}_{arm}_step{step}.json' for seed in cfg['train_seeds'] for arm in ['T','C'] for step in cfg['evaluate_steps']]
for file in files:
 d=json.loads(file.read_text());assert d['config_sha256']==freeze['config_sha256']
 m=re.fullmatch(r'seed(\d+)_([TC])',d['name']);seed=int(m[1]) if m else 0;arm=m[2] if m else 'baseline'
 for q in d['records']:
  meta=dict(cell=d['name'],training_seed=seed,arm=arm,step=d['step'],kind=q['kind'],subject=q['subject'],template=q['template'],prompt=q['prompt'])
  for s in q['scores']:
   assert len(s['token_ids'])==len(s['token_logprobs']) and abs(sum(s['token_logprobs'])-s['sum_logprob'])<1e-8
   sc.append(dict(**meta,**s,n_tokens=len(s['token_ids'])))
  for o in q['outputs']:
   z=dict(**meta,mode=o['mode'],decode_seed=o['decode_seed'],text=o['text'],token_ids=json.dumps(o['token_ids']))
   if q['kind']=='target':z.update(classify(o['text']))
   else:
    text=o['text'].strip();aliases='|'.join(re.escape(a) for a in q['aliases'])
    hit=bool(re.match(r'^(?:the city of |the capital city of |the capital city |the city )?(?:'+aliases+r')(?!\w)',text,re.I))
    z.update(target_label='correct_name_first_slot' if hit else 'other_unverified',full_response_factuality='unverified',audit_reason='old control name diagnostic only; baseline control qualification failed')
   out.append(z)
o=pd.DataFrame(out);s=pd.DataFrame(sc);assert len(o)==1170 and len(s)==52
o.to_csv(R/'ALL_OUTPUTS.csv',index=False,encoding='utf-8-sig');s.to_csv(R/'ALL_CANDIDATE_SCORES.csv',index=False)
o[(o.kind=='target')&o.target_label.ne('correct_target_assertion')].to_csv(R/'TARGET_FAILURES_AND_UNCERTAIN.csv',index=False,encoding='utf-8-sig')
metrics=[]
for keys,h in o[(o.kind=='target')&(o['mode']=='sample')].groupby(['training_seed','arm','step','template']):
 correct=h.target_label.eq('correct_target_assertion');wrong=h.target_label.isin(['incorrect_target_assertion','mixed_target_assertions']);u=~(correct|wrong)
 metrics.append(dict(training_seed=keys[0],arm=keys[1],step=keys[2],template=keys[3],n=len(h),correct=int(correct.sum()),wrong=int(wrong.sum()),uncertain_or_nonresponse=int(u.sum()),correct_rate=correct.mean(),wrong_rate=wrong.mean(),uncertain_rate=u.mean(),target_wrong_lower=wrong.mean(),target_wrong_upper=(wrong|u).mean()))
mt=pd.DataFrame(metrics);mt.to_csv(R/'TARGET_RATES.csv',index=False)
margins=[]
for keys,h in s.groupby(['training_seed','arm','step','template']):
 a=h[h.role=='gold'].iloc[0];b=h[h.role=='wrong'].iloc[0];z=dict(training_seed=keys[0],arm=keys[1],step=keys[2],template=keys[3])
 for metric in ['sum_logprob','mean_logprob','first_token_logprob']:z['gold_'+metric]=a[metric];z['margin_'+metric]=a[metric]-b[metric]
 margins.append(z)
mg=pd.DataFrame(margins);mg.to_csv(R/'CANDIDATE_MARGINS.csv',index=False)
paired=[]
for seed in cfg['train_seeds']:
 for step in cfg['evaluate_steps']:
  for ti in [0,1]:
   t=mt[(mt.training_seed==seed)&(mt.arm=='T')&(mt.step==step)&(mt.template==ti)].iloc[0]
   c=mt[(mt.training_seed==seed)&(mt.arm=='C')&(mt.step==step)&(mt.template==ti)].iloc[0]
   a=mg[(mg.training_seed==seed)&(mg.arm=='T')&(mg.step==step)&(mg.template==ti)].iloc[0]
   b=mg[(mg.training_seed==seed)&(mg.arm=='C')&(mg.step==step)&(mg.template==ti)].iloc[0]
   z=dict(training_seed=seed,step=step,template=ti)
   for metric in ['correct_rate','wrong_rate','uncertain_rate']:z[metric+'_T_minus_C']=t[metric]-c[metric]
   for metric in ['sum_logprob','mean_logprob','first_token_logprob']:
    z['gold_'+metric+'_T_minus_C']=a['gold_'+metric]-b['gold_'+metric];z['margin_'+metric+'_T_minus_C']=a['margin_'+metric]-b['margin_'+metric]
   paired.append(z)
pa=pd.DataFrame(paired);pa.to_csv(R/'PAIRED_EFFECTS.csv',index=False)
numeric=[x for x in pa.columns if x.endswith('_T_minus_C')]
summ=pa.groupby(['step','template'])[numeric].agg(['mean','std','min','max']);summ.columns=['__'.join(c) for c in summ.columns];summ.reset_index().to_csv(R/'EFFECT_SUMMARY.csv',index=False)
gate=[]
for ti,h in pa[pa.step==cfg['primary_step']].groupby('template'):
 z=dict(template=int(ti),correct_gain_mean=float(h.correct_rate_T_minus_C.mean()),seeds_gain_at_least_10pp=int((h.correct_rate_T_minus_C>=.10).sum()),uncertain_change_mean=float(h.uncertain_rate_T_minus_C.mean()),gold_sum_gain_mean=float(h.gold_sum_logprob_T_minus_C.mean()))
 z['passed']=bool(z['correct_gain_mean']>=.10 and z['seeds_gain_at_least_10pp']>=2 and z['uncertain_change_mean']<=0 and z['gold_sum_gain_mean']>0);gate.append(z)
# Check actual training pairing and hashes across both branches for every seed/update.
logs=[];checks=[];initial=set()
for seed in cfg['train_seeds']:
 a=json.loads((R/f'complete_seed{seed}_T.json').read_text());b=json.loads((R/f'complete_seed{seed}_C.json').read_text())
 assert a['initial_state_sha256']==b['initial_state_sha256'];initial.add(a['initial_state_sha256'])
 for aa,bb in zip(a['logs'],b['logs']):
  for key in ['step','phase','background','background_sha256','foreground_tokens','total_tokens']:assert aa[key]==bb[key]
  if aa['phase']=='background_only':assert aa['input_sha256']==bb['input_sha256']
  else:assert aa['input_sha256']!=bb['input_sha256']
  checks.append(dict(seed=seed,step=aa['step'],pairing_passed=True))
 for d in [a,b]:
  assert d['final_state_sha256']!=d['initial_state_sha256']
  for row in d['logs']:logs.append(dict(cell=d['cell'],seed=d['seed'],arm=d['arm'],**row))
assert len(initial)==1 and len(logs)==192 and np.isfinite(pd.DataFrame(logs)[['loss','gradient_norm']]).all().all()
pd.DataFrame(logs).to_csv(R/'ALL_TRAINING_UPDATES.csv',index=False);pd.DataFrame(checks).to_csv(R/'TRAINING_PAIRING_CHECKS.csv',index=False)
controls=o[(o.kind!='target')&(o['mode']=='sample')].copy();controls['correct_name']=controls.target_label.eq('correct_name_first_slot')
controls.groupby(['training_seed','arm','step','subject']).correct_name.agg(['sum','count','mean']).reset_index().to_csv(R/'OLD_CONTROL_DIAGNOSTICS.csv',index=False)
status=dict(positive_control_passed=all(x['passed'] for x in gate),template_gates=gate,outputs=1170,target_outputs=858,old_control_outputs=312,score_sequences=52,optimizer_updates=192,paired_training_checks=len(checks),
            interpretation='development measurement gate only, not independent causal confirmation',old_transfer_controls_qualified=False,automatic_four_branch_launch=False,
            primary_step=32,full_response_FEI_identified=False)
(R/'ANALYSIS_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));print(json.dumps(status,ensure_ascii=False,indent=2));print(pa.to_string(index=False))
