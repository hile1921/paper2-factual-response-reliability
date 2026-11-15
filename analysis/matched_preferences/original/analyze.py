from pathlib import Path
import json,re,hashlib,unicodedata
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;I=R/'inputs';B=5000;SEED=20261115
freeze=json.loads((R/'INPUT_FREEZE.json').read_text())
for z in freeze['sources']:assert hashlib.sha256((I/z['snapshot']).read_bytes()).hexdigest()==z['sha256']
assert hashlib.sha256((R/'ANALYSIS_PLAN.md').read_bytes()).hexdigest()==freeze['analysis_plan_sha256']
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().casefold()).strip()
def slot(text,name):
 text=norm(text)
 for p in ['the city of ','the capital city of ','the capital city ','the city ']:
  if text.startswith(p):text=text[len(p):];break
 return bool(re.match(re.escape(norm(name))+r'(?!\w)',text))
d=pd.read_csv(R/'CANONICAL_GENERATIONS.csv',keep_default_na=False);facts=pd.read_csv(R/'FACT_REGISTRY.csv',keep_default_na=False)
c=pd.read_csv(I/'candidates85.csv',keep_default_na=False);g=pd.read_csv(I/'gold_summary.csv')
scores=[]
for (uid,step),h in c.groupby(['uid','step']):
 assert len(h)==85 and h.candidate.nunique()==85
 gg=h[h.candidate.eq(h.gold)];assert len(gg)==1
 z=dict(uid=uid,step=step,gold=gg.gold.iloc[0],gold_tokens=int(gg.n_answer_tokens.iloc[0]))
 for name,metric in [('sum','sum_logprob'),('mean','mean_logprob')]:
  a=h[metric].to_numpy();gl=float(gg[metric].iloc[0]);v=np.exp(a-a.max());ix=h.candidate.eq(h.gold).to_numpy()
  z['mass_'+name]=float(v[ix][0]/v.sum());z['rank_'+name]=int((a>gl).sum()+1);z['gold_'+name]=gl
 scores.append(z)
sc=pd.DataFrame(scores);assert len(sc)==450
check=sc.merge(g[['uid','step','gold_prob','gold_logprob']],on=['uid','step'],validate='one_to_one')
assert np.max(np.abs(check.mass_sum-check.gold_prob))<1e-10 and np.max(np.abs(check.gold_sum-check.gold_logprob))<1e-8
assert sc.groupby('uid').gold_tokens.nunique().eq(1).all()
sc.to_csv(R/'RECOMPUTED_CANDIDATE_SCORES.csv',index=False)
d['literal_gold_slot']=[slot(a,b) for a,b in zip(d.output,d.gold)]
d['decode_group']=np.where(d['mode'].eq('greedy'),'greedy','sample')
d['verified_correct']=d.first_claim_correct.astype(int);d['strict_correct']=d.output_state.eq('Correct').astype(int)
d['confirmed_error']=d.error_confirmed.astype(int);d['error_upper']=(d.error_confirmed|d.possible_error_unresolved).astype(int)
d['unverifiable']=d.output_state.eq('Unverifiable').astype(int);d['nonresponsive']=d.output_state.eq('Non-responsive').astype(int)
d['literal_gold']=d.literal_gold_slot.astype(int)
d.to_csv(R/'LINKED_OUTPUTS_WITH_DIAGNOSTICS.csv',index=False,encoding='utf-8-sig')
d[d.literal_gold_slot.ne(d.first_claim_correct)].to_csv(R/'NAME_VS_VERIFICATION_DISAGREEMENTS.csv',index=False,encoding='utf-8-sig')
fields=['verified_correct','strict_correct','confirmed_error','error_upper','unverifiable','nonresponsive','literal_gold']
groups=d.groupby(['uid','step','decode_group']);rates=groups[fields].mean().add_suffix('_rate').reset_index();counts=groups.size().rename('attempts').reset_index()
rates=rates.merge(counts,on=['uid','step','decode_group'],validate='one_to_one')
assert rates[rates.decode_group.eq('sample')].attempts.eq(4).all() and rates[rates.decode_group.eq('greedy')].attempts.eq(1).all()
panel=rates.merge(sc,on=['uid','step'],validate='many_to_one').merge(facts.drop(columns='gold'),on='uid',validate='many_to_one')
assert len(panel)==784
panel.to_csv(R/'MATCHED_FACT_CHECKPOINT_PANEL.csv',index=False)
common=[10000,340000,658032];national=set(facts.loc[facts.scope_grade.eq('national_capital_name_concordant'),'uid'])
complete=set(panel.groupby('uid').step.nunique().loc[lambda x:x.eq(5)].index)
panels={'national42_common3':panel[panel.uid.isin(national)&panel.step.isin(common)],'all90_common3':panel[panel.step.isin(common)],
        'national_complete5':panel[panel.uid.isin(national&complete)],'all_complete5':panel[panel.uid.isin(complete)]}
for cohort in ['original','confirmation']:panels['national_common3_'+cohort]=panels['national42_common3'][panels['national42_common3'].cohort.eq(cohort)]
def mean_ci(values,rng):
 values=np.asarray(values,float);draw=values[rng.integers(0,len(values),(B,len(values)))].mean(1)
 return float(values.mean()),float(np.quantile(draw,.025)),float(np.quantile(draw,.975))
def corr_ci(x,y,rng):
 x=np.asarray(x,float);y=np.asarray(y,float);ix=rng.integers(0,len(x),(B,len(x)))
 def calc(a,b):
  a=rankdata(a,axis=-1);b=rankdata(b,axis=-1);a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
  den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1));return np.divide((a*b).sum(-1),den,out=np.full(den.shape,np.nan),where=den>0)
 point=float(calc(x,y));bs=calc(x[ix],y[ix]);finite=bs[np.isfinite(bs)]
 return point,float(np.quantile(finite,.025)) if len(finite) else np.nan,float(np.quantile(finite,.975)) if len(finite) else np.nan,int(len(bs)-len(finite))
stage=[];changes=[];corr=[];patterns=[];endpoint=[]
measures=['mass_sum','mass_mean','gold_sum','gold_mean']+[x+'_rate' for x in fields]
for pname,p in panels.items():
 for mode,h in p.groupby('decode_group'):
  for step,z in h.groupby('step'):
   # Aggregate UID trajectories within subject if aliases or multi-object facts exist.
   subj=z.groupby('s_uri')[measures].mean();rng=np.random.default_rng(SEED)
   for metric in measures:
    mean,lo,hi=mean_ci(subj[metric],rng);stage.append(dict(panel=pname,mode=mode,step=step,metric=metric,facts=z.uid.nunique(),subjects=len(subj),mean=mean,ci_low=lo,ci_high=hi))
  early=h[h.step.eq(10000)].set_index('uid');late=h[h.step.eq(658032)].set_index('uid');assert set(early.index)==set(late.index)
  late=late.loc[early.index];z=early[['s_uri','subject','cohort']].copy()
  for metric in measures:z['delta_'+metric]=late[metric]-early[metric]
  for col in ['verified_correct_rate','literal_gold_rate']:z['initial_'+col]=early[col];z['final_'+col]=late[col]
  z['late_mass_sum']=late.mass_sum;z['panel']=pname;z['decode_group']=mode
  endpoint.append(z.reset_index())
  zz=z.groupby('s_uri')[[x for x in z if x.startswith('delta_')]].mean();rng=np.random.default_rng(SEED)
  for metric in measures:
   mean,lo,hi=mean_ci(zz['delta_'+metric],rng);changes.append(dict(panel=pname,mode=mode,metric=metric,subjects=len(zz),mean_change=mean,ci_low=lo,ci_high=hi))
  for pred in ['mass_sum','mass_mean','gold_sum','gold_mean']:
   for outcome in ['verified_correct_rate','literal_gold_rate','confirmed_error_rate']:
    val,lo,hi,bad=corr_ci(zz['delta_'+pred],zz['delta_'+outcome],rng)
    corr.append(dict(panel=pname,mode=mode,predictor=pred,outcome=outcome,subjects=len(zz),spearman=val,ci_low=lo,ci_high=hi,degenerate_bootstrap_draws=bad))
  for pred in ['mass_sum','mass_mean']:
   for outcome in ['verified_correct_rate','literal_gold_rate']:
    rise=z['delta_'+pred]>1e-9;flat=z['delta_'+outcome]<=0;eligible=z['initial_'+outcome]<1
    patterns.append(dict(panel=pname,mode=mode,score=pred,outcome=outcome,facts=len(z),preference_rise=int(rise.sum()),preference_rise_generation_not_increase=int((rise&flat).sum()),initial_ceiling=int((~eligible).sum()),eligible_below_ceiling=int(eligible.sum()),rise_no_increase_below_ceiling=int((rise&flat&eligible).sum()),late_highscore_at_least_09=int(z.late_mass_sum.ge(.9).sum()),late_highscore_zero_observed_correct=int((z.late_mass_sum.ge(.9)&z['final_'+outcome].eq(0)).sum())))
pd.DataFrame(stage).to_csv(R/'STAGE_SUMMARY.csv',index=False);pd.DataFrame(changes).to_csv(R/'PAIRED_ENDPOINT_CHANGES.csv',index=False)
pd.DataFrame(corr).to_csv(R/'WITHIN_FACT_CHANGE_ASSOCIATIONS.csv',index=False);pd.DataFrame(patterns).to_csv(R/'TRANSITION_DIAGNOSTICS.csv',index=False)
pd.concat(endpoint,ignore_index=True).to_csv(R/'ENDPOINT_FACT_DATA.csv',index=False)
# Retrospectively selected actual-wrong events: exact output verification, named slot only.
events=pd.read_csv(I/'old_matched_errors.csv',keep_default_na=False)
ref=d.set_index('review_id');ev=[];ex=[]
for x in events.to_dict('records'):
 assert x['review_id'] in ref.index
 row=ref.loc[x['review_id']];assert x['original_output']==row.output and x['uid']==row.uid and x['step']==row.step
 if not slot(row.output,x['wrong_answer']):ex.append(dict(**x,exclusion='not a complete first-slot named wrong under fixed diagnostic'));continue
 ev.append(dict(**x,decode_group=row.decode_group,score_prefers_gold=float(x['value'])>0))
e=pd.DataFrame(ev).drop_duplicates(['review_id','wrong_answer','metric']);e.to_csv(R/'ACTUAL_WRONG_MATCHED_EVENTS.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(ex).to_csv(R/'ACTUAL_WRONG_EXCLUDED_EVENTS.csv',index=False,encoding='utf-8-sig')
assert not e.duplicated(['review_id','metric']).any()
e.groupby(['metric','decode_group']).agg(events=('review_id','size'),subjects=('uid','nunique'),gold_preferred=('score_prefers_gold','sum')).reset_index().to_csv(R/'ACTUAL_WRONG_EVENT_SUMMARY.csv',index=False)
inv=pd.read_csv(I/'wrong_pair_inventory.csv',keep_default_na=False);assert len(inv)==88
original_names=set(c.candidate.map(norm));inv['in_fixed85']=inv.generated_wrong_answer.map(norm).isin(original_names);inv.to_csv(R/'ACTUAL_WRONG_CANDIDATE_COVERAGE.csv',index=False)
status=dict(raw_generation_rows=1960,matched_fact_checkpoint_cells=392,missing_generation_cells=58,primary_facts=42,primary_three_stage_outputs=630,
 primary_five_complete_facts=len(national&complete),all_five_complete_facts=len(complete),fixed85_actual_wrong_coverage=int(inv.in_fixed85.sum()),actual_wrong_pairs=88,
 selected_error_event_count=e.review_id.nunique(),original85_summary_max_difference=float(np.max(np.abs(check.mass_sum-check.gold_prob))),
 inference='5000 whole-subject paired bootstrap, descriptive intervals, no new independent confirmation',new_generation=0,new_training=0,
 caution='unknown labels and sampling variability cannot be interpreted as unchanged true generation ability; panels and candidate normalization are restricted')
(R/'ANALYSIS_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
print(json.dumps(status,indent=2));print(pd.DataFrame(changes).query("panel == 'national42_common3' and mode == 'sample'").to_string(index=False))
print(pd.DataFrame(corr).query("panel == 'national42_common3' and mode == 'sample'").to_string(index=False));print(pd.DataFrame(patterns).query("panel == 'national42_common3' and mode == 'sample'").to_string(index=False))
