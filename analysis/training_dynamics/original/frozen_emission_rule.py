"""Audit all frozen arms; separate sequence probability, ranking and open claims."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,unicodedata
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
def normal(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if not unicodedata.combining(c))
def classify(text,correct,competing):
 s=normal(text).strip(' \n\t\"\'')
 s=re.sub(r'^(?:located in |in |the city of |the country of |the city |the )','',s)
 for label,name in [('correct_first_claim',correct),('verified_target_error',competing)]:
  name=normal(name)
  if re.match(re.escape(name)+r'(?!\w)',s):
   if re.search(r'\b(not|never|isn.t|wasn.t)\b',s[:len(name)+35]):return 'unverified'
   return label
 if not s or re.match(r'(?:i (?:do not|don.t) know|unknown|cannot answer|i cannot)',s):return 'no_target_answer'
 return 'unverified'

def main():
 cfg=json.loads((R/'CAUSAL_CONFIG.json').read_text(encoding='utf8'))
 expected=[R/f'training_seed{s}_{a}.json' for s in cfg['seeds'] for a in cfg['branches']]
 assert all(p.exists() for p in expected),'all eight branches must finish before analysis'
 assert (R/'training_seed202_A_gpu0.json').exists(),'hardware-matched A replay must finish'
 rawscores=[];rawopen=[];loss=[];weights=[];summary=[]
 names=['initial_evaluation']+[f'after_seed{s}_{a}' for s in cfg['seeds'] for a in cfg['branches']]+['after_seed202_A_gpu0']+[f'diagnostic_{s}' for s in cfg['diagnostic_steps']]
 for name in names:
  path=R/f'{name}.json';j=json.loads(path.read_text(encoding='utf8'))
  assert j['config_sha256']==hashlib.sha256((R/'CAUSAL_CONFIG.json').read_bytes()).hexdigest()
  match=re.match(r'after_seed(\d+)_([ABCD])',name)
  seed=int(match[1]) if match else 0;arm=match[2] if match else 'initial' if name=='initial_evaluation' else 'diagnostic'
  stage=int(name.split('_')[-1]) if name.startswith('diagnostic_') else 10000
  for p in j['records']:
   c,w=p['scores'];assert c['role']=='correct' and w['role']=='competing'
   meta=dict(run=name,training_seed=seed,training_arm=arm,analysis_role='hardware_reproduction' if name=='after_seed202_A' else 'primary',checkpoint=stage,id=p['id'],subject=p['subject'],relation=p['relation'],template=p['template'],
    context_arm=p.get('context_arm','NoContext'),transfer_control=p.get('transfer_control',False))
   pair=np.logaddexp(c['sum_logprob'],w['sum_logprob'])
   rawscores.append(dict(**meta,correct=c['answer'],competing=w['answer'],correct_sum=c['sum_logprob'],competing_sum=w['sum_logprob'],
    correct_mean=c['mean_logprob'],competing_mean=w['mean_logprob'],correct_sequence_probability=np.exp(c['sum_logprob']),
    competing_sequence_probability=np.exp(w['sum_logprob']),restricted_pair_correct_probability=np.exp(c['sum_logprob']-pair),
    wrong_minus_correct_sum=w['sum_logprob']-c['sum_logprob'],wrong_minus_correct_mean=w['mean_logprob']-c['mean_logprob']))
   for o in p['outputs']:
    label=classify(o['text'],c['answer'],w['answer'])
    rawopen.append(dict(**meta,mode=o['mode'],output=o['text'],output_ids=json.dumps(o['token_ids']),status=label,
     error_lower=int(label=='verified_target_error'),error_upper=int(label in ['verified_target_error','unverified']),
     first_claim_correct=int(label=='correct_first_claim'),unverified=int(label=='unverified'),
     full_output_additional_claims_not_verified=True))
 scores=pd.DataFrame(rawscores);outputs=pd.DataFrame(rawopen)
 scores.to_csv(R/'candidate_scores.csv',index=False,encoding='utf-8-sig')
 outputs.to_csv(R/'open_outputs.csv',index=False,encoding='utf-8-sig')
 outputs[outputs.status.ne('correct_first_claim')].to_csv(R/'all_failures_and_uncertainties.csv',index=False,encoding='utf-8-sig')
 sampled=outputs[outputs['mode'].ne('greedy')]
 group=['run','training_seed','training_arm','analysis_role','checkpoint','subject','relation','context_arm']
 rate=sampled.groupby(group).agg(outputs=('output','size'),error_lower=('error_lower','mean'),error_upper=('error_upper','mean'),
  first_correct=('first_claim_correct','mean'),unverified=('unverified','mean')).reset_index()
 rate.to_csv(R/'generation_rates.csv',index=False)
 scoremean=scores.groupby(group)[['wrong_minus_correct_sum','wrong_minus_correct_mean','restricted_pair_correct_probability']].mean().reset_index()
 model=rate.merge(scoremean,on=group,validate='one_to_one');model.to_csv(R/'model_metrics.csv',index=False)
 contrasts=[]
 trained=model[model.training_arm.isin(list('ABCD'))&model.analysis_role.eq('primary')]
 for (seed,subject,relation),g in trained.groupby(['training_seed','subject','relation']):
  t=g.set_index('training_arm');assert set(t.index)==set('ABCD')
  for upper,lower in [('B','A'),('C','A'),('D','A')]:
   contrasts.append(dict(training_seed=seed,subject=subject,relation=relation,contrast=upper+'-'+lower,
    confirmed_error_difference=t.at[upper,'error_lower']-t.at[lower,'error_lower'],
    sensitivity_difference_low=t.at[upper,'error_lower']-t.at[lower,'error_upper'],
    sensitivity_difference_high=t.at[upper,'error_upper']-t.at[lower,'error_lower'],
    wrong_log_margin_sum_difference=t.at[upper,'wrong_minus_correct_sum']-t.at[lower,'wrong_minus_correct_sum'],
    wrong_log_margin_mean_difference=t.at[upper,'wrong_minus_correct_mean']-t.at[lower,'wrong_minus_correct_mean']))
  contrasts.append(dict(training_seed=seed,subject=subject,relation=relation,contrast='(B-A)-(D-C)',
   confirmed_error_difference=(t.at['B','error_lower']-t.at['A','error_lower'])-(t.at['D','error_lower']-t.at['C','error_lower']),
   sensitivity_difference_low=np.nan,sensitivity_difference_high=np.nan,
   wrong_log_margin_sum_difference=(t.at['B','wrong_minus_correct_sum']-t.at['A','wrong_minus_correct_sum'])-(t.at['D','wrong_minus_correct_sum']-t.at['C','wrong_minus_correct_sum']),
   wrong_log_margin_mean_difference=(t.at['B','wrong_minus_correct_mean']-t.at['A','wrong_minus_correct_mean'])-(t.at['D','wrong_minus_correct_mean']-t.at['C','wrong_minus_correct_mean'])))
 pd.DataFrame(contrasts).to_csv(R/'paired_training_seed_contrasts.csv',index=False)
 # Five decode modes stay separate; greedy is never pooled into a probability estimate.
 outputs.groupby(group+['mode'])[['error_lower','error_upper','first_claim_correct']].mean().reset_index().to_csv(R/'decode_seed_metrics.csv',index=False)
 records=[]
 import random
 for seed in cfg['seeds']:
  schedule=list(range(8))*3;random.Random(seed).shuffle(schedule)
  for arm,b in cfg['branches'].items():
   name=f'seed{seed}_{arm}'+('_gpu0' if seed==202 and arm=='A' else '')
   j=json.loads((R/f'training_{name}.json').read_text());assert j['steps']==24 and j['tokens']==4608
   assert j['weight']['lossless_reconstruction_verified']
   weights.append(dict(seed=seed,arm=arm,analysis_role='primary',**j['weight']))
   loss.extend(dict(seed=seed,arm=arm,**x) for x in j['losses'])
   for step,bg in enumerate(schedule):
    ids=b['competitor_ids']+b['correct_ids']+cfg['backgrounds'][bg]['ids']+b['reservoir_ids']
    a=cfg['branches']['A'];reference=a['competitor_ids']+a['correct_ids']+cfg['backgrounds'][bg]['ids']+a['reservoir_ids']
    assert len(ids)==192 and Counter(ids)==Counter(reference)
    records.append(dict(seed=seed,arm=arm,step=step,background=bg,tokens=192,loss_tokens=191,
     input_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest(),ids=ids))
 pd.DataFrame(loss).to_csv(R/'training_losses.csv',index=False)
 pd.DataFrame(weights).to_csv(R/'BRANCH_WEIGHT_MANIFEST.csv',index=False)
 (R/'TRAINING_RECORDS.json').write_text(json.dumps(records),encoding='utf8')
 oldA=json.loads((R/'training_seed202_A.json').read_text());newA=json.loads((R/'training_seed202_A_gpu0.json').read_text())
 (R/'HARDWARE_TRAINING_REPRODUCTION.json').write_text(json.dumps(dict(
  original_gpu1_state_sha256=oldA['weight']['final_state_sha256'],replayed_gpu0_state_sha256=newA['weight']['final_state_sha256'],
  bitwise_identical=oldA['weight']['final_state_sha256']==newA['weight']['final_state_sha256'],
  primary_A_rule='GPU-matched replay used regardless of outcome; original retained as reproduction'),indent=2),encoding='utf8')
 manifests=[json.loads(p.read_text()) for p in R.glob('INITIAL_WEIGHTS*.json')]
 assert all(m['files']==manifests[0]['files'] for m in manifests)
 integrity=dict(all_eight_training_branches=True,training_steps=192,training_tokens=36864,loss_tokens=36672,
  technical_replay_steps=24,technical_replay_tokens=4608,
  all_stepwise_token_multisets_equal=True,backgrounds_identical_within_training_seed=True,
  diagnostic_checkpoints=cfg['diagnostic_steps'],diagnostic_prompts=72,training_evaluation_prompts=108,
  paired_candidate_probe_rows=len(scores),candidate_sequence_evaluations=len(scores)*2,open_rows=len(outputs),single_target_fact=True,training_seeds=2,
  population_country_cluster_CI_estimable=False,
  interpretation='finite paired training intervention on curated excerpts; not original pretraining replication; no full-output hallucination-rate claim')
 (R/'ANALYSIS_INTEGRITY.json').write_text(json.dumps(integrity,indent=2),encoding='utf8')
 print(json.dumps(integrity,indent=2));print(pd.DataFrame(contrasts).query("subject == 'Guinea' and relation == 'capital'").to_string(index=False))

if __name__=='__main__':main()
