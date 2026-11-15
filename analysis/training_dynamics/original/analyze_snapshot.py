"""CPU-only, immutable-snapshot C1/C2/C3 analysis. See ANALYSIS_SPEC.md."""
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
import sys
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
P = R.parents[1]
RAW, OUT, INPUT = R/'raw', R/'tables', R/'inputs'
sys.path[:0] = [str(P/'p1_confirmation'), str(P/'p1_audit')]
from verify_analyze import References, source_extractor
from run_p1 import candidate

SEED, NBOOT = 20261129, 5000
CONTRASTS = [('CorrectOnly','NoContext'),('OtherOnly','NoContext'),('Both','NoContext'),('Neutral','NoContext'),('CorrectOnly','OtherOnly')]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv(d, name):
    d.to_csv(OUT/name, index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(INPUT/name, keep_default_na=False)


@lru_cache(maxsize=64)
def boot_indices(n):
    return np.random.default_rng(SEED).integers(0,n,(NBOOT,n))


def interval(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:
        return dict(n=len(a),effect=float(a.mean()) if len(a) else np.nan,ci_low=np.nan,ci_high=np.nan)
    boot=a[boot_indices(len(a))].mean(axis=1)
    return dict(n=len(a),effect=float(a.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))


def signflip(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return np.nan
    signs=np.array(list(itertools.product([-1,1],repeat=len(a))))
    return float(np.mean(np.abs((signs*a).mean(axis=1))>=abs(a.mean())-1e-12))


def holm(group):
    group=group.copy();order=np.argsort(group.p_exact.to_numpy());adjusted=np.full(len(group),np.nan)
    acc=0
    for rank,idx in enumerate(order):
        value=group.p_exact.iloc[idx]
        if np.isfinite(value):
            acc=max(acc,min(1.,value*(len(group)-rank)));adjusted[idx]=acc
    group['p_holm']=adjusted
    return group


def audit_factory():
    refs,extract=References(),source_extractor()
    @lru_cache(maxsize=200000)
    def audit(subject,qid,prop,text):
        span,state=extract(text,subject,prop);tests=[(span,state)]
        if prop=='capital' and text.strip():
            s,t=candidate(text,subject);tests.append((s,'target_relation_assertion' if t=='candidate' else 'nonanswer'))
        rows=[]
        for span,state in tests:
            z=refs.verify(SimpleNamespace(s_uri=qid,prop=prop,subj=subject,output=text),span,state)
            rows.append(dict(audit_label=z[0],audit_reason=z[1],span=span,subject_reference=z[2],answer_reference=z[3],claim=z[4],cross_reference=z[5]))
        return next((r for label in ['wrong_confirmed','correct_verified','type_mismatch','unverified'] for r in rows if r['audit_label']==label),rows[0])
    return refs,audit


def check_inputs():
    manifest=json.loads((RAW/'ANALYSIS_SNAPSHOT_MANIFEST_20261113.json').read_text())
    for f in manifest['files']:
        p=RAW/f['name'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['name']
    cfg=json.loads((RAW/'CAUSAL_CONFIG.json').read_text())
    cfgsha=sha(RAW/'CAUSAL_CONFIG.json');assert cfgsha==(RAW/'CAUSAL_CONFIG.sha256').read_text().split()[0]
    expected={x['id']:x for x in cfg['c3_probes']}
    duplicates=[];integrity=[]
    for f in sorted(RAW.glob('complete_*.json')):
        end=json.loads(f.read_text());tid=end['task']['id']
        c1=json.loads((RAW/f'c1_{tid}.json').read_text());c3=json.loads((RAW/f'c3_{tid}.json').read_text())
        assert c1['config_sha256']==cfgsha and c3['config_sha256']==cfgsha and end['config_sha256']==cfgsha
        assert len(c1['records'])==2730 and len(c3['records'])==400
        assert len({(x['uid'],x['template'],x['mode']) for x in c1['records']})==2730
        assert {x['id'] for x in c3['records']}==set(expected)
        groups={}
        for x in c3['records']:
            for k,v in expected[x['id']].items():assert x[k]==v,(tid,x['id'],k)
            assert len(x['outputs'])==5 and {o['mode'] for o in x['outputs']}=={'greedy','sample101','sample102','sample103','sample104'}
            for score in x['scores']:
                assert len(score['token_ids'])==len(score['token_logprobs'])
                assert np.isclose(sum(score['token_logprobs']),score['sum_logprob'],atol=1e-4)
                assert np.isclose(np.mean(score['token_logprobs']),score['mean_logprob'],atol=1e-4)
                assert np.isfinite(score['sum_logprob'])
            key=(x['prompt_text'],tuple(x['candidates']))
            groups.setdefault(key,[]).append(x)
        assert len(groups)==200 and all(len(v)==2 for v in groups.values())
        maxdiff=0;output_diff=0
        for items in groups.values():
            a,b=items
            maxdiff=max(maxdiff,max(abs(a['scores'][i]['sum_logprob']-b['scores'][i]['sum_logprob']) for i in (0,1)))
            output_diff+=sum(u['token_ids']!=v['token_ids'] for u,v in zip(a['outputs'],b['outputs']))
        duplicates.append(dict(task_id=tid,raw_probes=400,unique_prompts=200,max_sum_logprob_difference=maxdiff,different_generation_duplicates=output_diff))
        integrity.append(dict(task_id=tid,model=end['task']['model'],revision=end['task']['revision'],training_progress=end['task']['training_progress'],primary_three_stage=end['task']['run_c3'],c1_outputs=2730,c3_queries=400,unique_c3_queries=200,c3_open_outputs=2000))
    csv(pd.DataFrame(integrity),'INTEGRITY.csv');csv(pd.DataFrame(duplicates),'DUPLICATE_AUDIT.csv')
    return cfg,{t['id']:t for t in cfg['tasks']},integrity


def c1_analysis(audit):
    records=[]
    for f in sorted(RAW.glob('c1_*.json')):
        for x in json.loads(f.read_text())['records']:
            a=audit(x['subj'],x['s_uri'],x['prop'],x['output'])
            records.append(dict(**x,**a))
    d=pd.DataFrame(records)
    d['fact_id']='wd:'+d.s_uri+'|'+d.prop.map({'capital':'P36','country':'P17'})+'|wd:'+d.o_uri+'|p1_unqualified_target'
    d['correct']=d.audit_label.eq('correct_verified').astype(float)
    d['error_lower']=d.audit_label.eq('wrong_confirmed').astype(float)
    d['unverified']=d.audit_label.eq('unverified').astype(float)
    d['nonanswer']=d.audit_label.eq('nonanswer').astype(float)
    d['type_mismatch']=d.audit_label.eq('type_mismatch').astype(float)
    d['wide_error_upper']=1-d.correct
    d['decode']=np.where(d['mode'].eq('greedy'),'greedy','sample')
    # Token IDs and prompts stay in immutable JSON; compact audit keeps all text and provenance.
    csv(d.drop(columns=['input_ids','output_ids']),'C1_AUDITED_OUTPUTS.csv')
    keys=['model','revision','cohort','prop','decode']
    summary=d.groupby(keys).agg(outputs=('output','size'),subjects=('s_uri','nunique'),correct=('correct','mean'),error_lower=('error_lower','mean'),unverified=('unverified','mean'),nonanswer=('nonanswer','mean'),type_mismatch=('type_mismatch','mean'),wide_error_upper=('wide_error_upper','mean')).reset_index()
    csv(summary,'C1_TIME_SUMMARY.csv')
    csv(d.groupby(['task_id','model','revision','uid','fact_id','subj','s_uri','prop','decode'])[['correct','error_lower','unverified','wide_error_upper']].mean().reset_index(),'C1_FACT_RATES.csv')
    return d


def c3_analysis(cfg,tasks,refs,audit):
    spec=importlib.util.spec_from_file_location('frozen_emission',INPUT/'frozen_emission_rule.py')
    emission=importlib.util.module_from_spec(spec);spec.loader.exec_module(emission)
    fact=read('capital_chunks.csv').set_index('subj').to_dict('index')
    scores=[];opened=[];truth=[]
    for f in sorted(RAW.glob('c3_*.json')):
        j=json.loads(f.read_text());tid=f.stem[3:];task=tasks[tid]
        for x in j['records']:
            c,w=x['scores'];assert c['role']=='correct' and w['role']=='competing'
            target_country=x['correct_country'];fct=fact[target_country]
            target_id=fct['s_uri'] if x['relation']=='capital' else fct['o_uri']
            meta=dict(task_id=tid,model=task['model'],revision=task['revision'],training_progress=task['training_progress'],primary_three_stage=task['run_c3'],probe_id=x['id'],panel_subject=x['panel_subject'],target_country=target_country,pair_cluster=x['pair_cluster'],subject=x['subject'],relation=x['relation'],swapped=x['swapped'],context=x['context_arm'],template=x['template'],correct_answer=c['answer'],competing_answer=w['answer'],fact_id='wd:'+fct['s_uri']+'|P36|wd:'+fct['o_uri']+'|p1_unqualified_target')
            sm=c['sum_logprob']-w['sum_logprob'];mm=c['mean_logprob']-w['mean_logprob']
            scores.append(dict(**meta,correct_tokens=len(c['token_ids']),competing_tokens=len(w['token_ids']),sum_margin=sm,mean_margin=mm,sum_pick=float(sm>0) if sm!=0 else .5,mean_pick=float(mm>0) if mm!=0 else .5,restricted_correct_probability=float(np.exp(c['sum_logprob']-np.logaddexp(c['sum_logprob'],w['sum_logprob']))),ranking_disagreement=int((sm>0)!=(mm>0))))
            for o in x['outputs']:
                status=emission.classify(o['text'],c['answer'],w['answer'])
                a=audit(x['subject'],target_id,x['relation'],o['text'])
                opened.append(dict(**meta,mode=o['mode'],decode='greedy' if o['mode']=='greedy' else 'sample',output=o['text'],closed_status=status,closed_correct=int(status=='correct_first_claim'),closed_competitor=int(status=='verified_target_error'),closed_unknown=int(status=='unverified'),closed_noanswer=int(status=='no_target_answer'),audit_correct=int(a['audit_label']=='correct_verified'),audit_error_lower=int(a['audit_label']=='wrong_confirmed'),audit_unverified=int(a['audit_label']=='unverified'),audit_nonanswer=int(a['audit_label']=='nonanswer'),**a))
            if not x['swapped'] and x['context_arm']=='NoContext' and x['template']==0 and tid==next(iter(sorted(tasks)),''):
                pass
    s,o=pd.DataFrame(scores),pd.DataFrame(opened)
    for (country,rel),g in s[~s.swapped].groupby(['target_country','relation']):
        row=g.iloc[0];fct=fact[country];qid=fct['s_uri'] if rel=='capital' else fct['o_uri']
        z=refs.verify(SimpleNamespace(s_uri=qid,prop=rel,subj=row.subject,output=row.correct_answer),row.correct_answer,'target_relation_assertion')
        truth.append(dict(target_country=country,relation=rel,subject=row.subject,expected=row.correct_answer,status=z[0],reason=z[1],subject_reference=z[2],answer_reference=z[3],cross_reference=z[5]))
    csv(pd.DataFrame(truth),'C3_EXPECTED_FACT_VERIFICATION.csv')
    csv(s,'C3_CANDIDATE_SCORES_ALL.csv');csv(o,'C3_OPEN_OUTPUTS_ALL.csv')
    csv(o[(o.closed_status!='correct_first_claim') | (o.audit_label!='correct_verified')],'C3_FAILURES_AND_UNCERTAINTIES.csv')
    unique=s[~s.swapped].copy();opened_unique=o[~o.swapped].copy()
    keys=['task_id','model','revision','training_progress','primary_three_stage','relation','context','target_country','pair_cluster']
    sc=unique.groupby(keys)[['sum_margin','mean_margin','sum_pick','mean_pick','restricted_correct_probability','ranking_disagreement']].mean().reset_index()
    op=opened_unique.groupby(keys+['decode'])[['closed_correct','closed_competitor','closed_unknown','closed_noanswer','audit_correct','audit_error_lower','audit_unverified','audit_nonanswer']].mean().reset_index()
    country=sc.merge(op,on=keys,validate='one_to_many')
    csv(country,'C3_COUNTRY_METRICS.csv')
    summary=country.groupby(['task_id','model','revision','training_progress','primary_three_stage','relation','context','decode'])[['sum_pick','mean_pick','sum_margin','mean_margin','restricted_correct_probability','ranking_disagreement','closed_correct','closed_competitor','closed_unknown','closed_noanswer','audit_correct','audit_error_lower','audit_unverified','audit_nonanswer']].mean().reset_index()
    csv(summary,'C3_STAGE_SUMMARY.csv')
    # Confidence intervals for levels: same country weight regardless of repeated generations.
    levels=[]
    metrics=['sum_pick','mean_pick','sum_margin','mean_margin','closed_correct','closed_competitor','closed_unknown','audit_correct','audit_error_lower']
    for key,g in country.groupby(['task_id','model','revision','relation','context','decode']):
        meta=dict(zip(['task_id','model','revision','relation','context','decode'],key))
        for metric in metrics:
            for cluster,values in [('country',g[metric]),('pair',g.groupby('pair_cluster')[metric].mean())]:
                levels.append(dict(**meta,metric=metric,cluster=cluster,**interval(values)))
    csv(pd.DataFrame(levels),'C3_LEVEL_CLUSTER_CI.csv')
    context=[]
    for key,g in country.groupby(['task_id','model','revision','relation','decode']):
        meta=dict(zip(['task_id','model','revision','relation','decode'],key))
        for metric in metrics:
            t=g.pivot(index=['target_country','pair_cluster'],columns='context',values=metric)
            for a,b in CONTRASTS:
                delta=t[a]-t[b];pair=delta.groupby(level='pair_cluster').mean()
                for cluster,values in [('country',delta),('pair',pair)]:
                    context.append(dict(**meta,metric=metric,contrast=a+'-'+b,cluster=cluster,p_exact=signflip(pair),**interval(values)))
    contexts=pd.DataFrame(context)
    holm_keys=['task_id','model','revision','relation','decode','metric','cluster']
    contexts=pd.concat([holm(g) for _,g in contexts.groupby(holm_keys)],ignore_index=True)
    csv(contexts,'C3_CONTEXT_CONTRASTS.csv')
    # Preselected earliest/latest and adjacent available phases; all exploratory here.
    times=[]
    for model,g in country.groupby('model'):
        order=g[['task_id','training_progress']].drop_duplicates().sort_values('training_progress').task_id.tolist()
        pairs=list(zip(order,order[1:]))
        if len(order)>2:pairs.append((order[0],order[-1]))
        for early,late in pairs:
            a=g[g.task_id==early];b=g[g.task_id==late]
            joined=a.merge(b,on=['target_country','pair_cluster','relation','context','decode'],suffixes=('_early','_late'),validate='one_to_one')
            for key,h in joined.groupby(['relation','context','decode']):
                for metric in metrics:
                    delta=h[metric+'_late']-h[metric+'_early']
                    for cluster,values in [('country',delta),('pair',h.assign(delta=delta).groupby('pair_cluster').delta.mean())]:
                        times.append(dict(model=model,early=early,late=late,relation=key[0],context=key[1],decode=key[2],metric=metric,cluster=cluster,exploratory=True,**interval(values)))
    csv(pd.DataFrame(times),'C3_TIME_CONTRASTS.csv')
    # Subject binding uses paired actual questions with fixed A/B candidates, not duplicate swap rows.
    binding=[]
    for key,g in unique.groupby(['task_id','model','revision','pair_cluster','relation','context','template']):
        assert len(g)==2 and g.target_country.nunique()==2
        a,b=g.sort_values('target_country').to_dict('records')
        assert a['correct_answer']==b['competing_answer'] and a['competing_answer']==b['correct_answer']
        binding.append(dict(zip(['task_id','model','revision','pair_cluster','relation','context','template'],key)) | dict(country_a=a['target_country'],country_b=b['target_country'],sum_binding=(a['sum_margin']+b['sum_margin'])/2,mean_binding=(a['mean_margin']+b['mean_margin'])/2,sum_both_correct=float(a['sum_pick']==1 and b['sum_pick']==1),mean_both_correct=float(a['mean_pick']==1 and b['mean_pick']==1),sum_answer_flips=float(a['sum_pick']==b['sum_pick']),mean_answer_flips=float(a['mean_pick']==b['mean_pick'])))
    binding=pd.DataFrame(binding);csv(binding,'C3_SUBJECT_BINDING_PAIRS.csv')
    bindsummary=[]
    for key,g in binding.groupby(['task_id','model','revision','relation','context']):
        for metric in ['sum_binding','mean_binding','sum_both_correct','mean_both_correct','sum_answer_flips','mean_answer_flips']:
            bindsummary.append(dict(zip(['task_id','model','revision','relation','context'],key)) | dict(metric=metric,cluster='pair',**interval(g.groupby('pair_cluster')[metric].mean())))
    csv(pd.DataFrame(bindsummary),'C3_SUBJECT_BINDING_CI.csv')
    # Ranking/generation dissociation, one row per country/decode/context.
    gp=opened_unique.groupby(['task_id','probe_id','decode'])[['closed_correct','audit_correct','closed_competitor','closed_unknown']].mean().reset_index()
    joint=unique.merge(gp,on=['task_id','probe_id'],validate='one_to_many')
    for score in ['sum','mean']:
        joint[score+'_candidate_correct_generation_not_confirmed']=joint[score+'_pick']*(1-joint.closed_correct)
        joint[score+'_candidate_wrong_generation_correct']=(1-joint[score+'_pick'])*joint.closed_correct
        joint[score+'_selection_minus_generation']=joint[score+'_pick']-joint.closed_correct
    csv(joint,'C3_RANKING_GENERATION_JOINT.csv')
    diss=[]
    fields=[score+suffix for score in ['sum','mean'] for suffix in ['_candidate_correct_generation_not_confirmed','_candidate_wrong_generation_correct','_selection_minus_generation']]
    per=joint.groupby(['task_id','model','revision','relation','context','decode','target_country','pair_cluster'])[fields].mean().reset_index()
    for key,g in per.groupby(['task_id','model','revision','relation','context','decode']):
        for metric in fields:
            for cluster,values in [('country',g[metric]),('pair',g.groupby('pair_cluster')[metric].mean())]:
                diss.append(dict(zip(['task_id','model','revision','relation','context','decode'],key)) | dict(metric=metric,cluster=cluster,**interval(values)))
    csv(pd.DataFrame(diss),'C3_DISSOCIATION_CI.csv')
    # Do templates and token aggregation change the conclusions?
    templates=unique.groupby(['task_id','model','revision','relation','context','template'])[['sum_pick','mean_pick','sum_margin','mean_margin','ranking_disagreement']].mean().reset_index()
    csv(templates,'C3_TEMPLATE_SCORE_ROBUSTNESS.csv')
    csv(opened_unique.groupby(['task_id','model','revision','relation','context','mode'])[['closed_correct','closed_competitor','closed_unknown','audit_correct']].mean().reset_index(),'C3_DECODE_SEED_ROBUSTNESS.csv')
    swapdiag=s.groupby(['task_id','model','revision','relation','context','swapped'])[['sum_pick','mean_pick','sum_margin','mean_margin']].mean().reset_index()
    csv(swapdiag,'C3_REAL_SWAPPED_DIAGNOSTIC.csv')
    return unique,opened_unique,country,summary


def c2_and_joint(c1,scores,opened,country):
    facts=read('capital_chunks.csv')
    facts['fact_id']='wd:'+facts.s_uri+'|P36|wd:'+facts.o_uri+'|p1_unqualified_target'
    evidence=read('evidence_timeline_90_by_5_updated.csv')
    oldwrong=read('wrong_answer_pair_inventory.csv')
    context=read('context_decisions.csv')
    timeline=read('wrong_relation_timeline_v2_88_by_5.csv')
    roles=read('relation_evidence_roles.csv')
    matching=scores[scores.relation.eq('capital')][['target_country','fact_id','correct_answer','competing_answer']].drop_duplicates()
    matching=matching.merge(facts[['fact_id','uid','obj','s_uri','o_uri']],on='fact_id',how='left',validate='many_to_one')
    matching['c2_target_fact_present']=matching.uid.notna()
    matching['c2_original_evidence_available']=matching.fact_id.isin(evidence.fact_id)
    matching['has_historical_actual_wrong_answer']=matching.fact_id.isin(oldwrong.fact_id)
    matching['competitor_matches_historical_actual_wrong']=matching.apply(lambda r:any(str(x).casefold()==str(r.competing_answer).casefold() for x in oldwrong.loc[oldwrong.fact_id==r.fact_id,'generated_wrong_answer']),axis=1)
    matching['cross_model_training_evidence_available']=False
    matching['scope']='C2 evidence belongs to original model corpus only; not exposure evidence for new families'
    csv(matching,'C1_C2_C3_FACT_MATCH.csv')
    ev=evidence[evidence.fact_id.isin(matching.fact_id)].copy();csv(ev,'C2_MATCHED_ORIGINAL_TIMELINES.csv')
    wt=timeline[timeline.fact_id.isin(matching.fact_id)].copy();csv(wt,'C2_MATCHED_HISTORICAL_WRONG_TIMELINES.csv')
    csv(context[context.fact_id.isin(matching.fact_id)],'C2_MATCHED_CONTEXT_DECISIONS.csv')
    # Cross-model C1 actual wrong answers are linked only on exact conservative identity text.
    wrong=c1[(c1.prop=='capital') & (c1.audit_label=='wrong_confirmed')].copy()
    wrong=wrong.merge(matching[['fact_id','competing_answer']].drop_duplicates(),on='fact_id',how='left')
    wrong['c3_pair_competitor_emitted']=wrong['span'].str.casefold()==wrong['competing_answer'].fillna('').str.casefold()
    wrong['historical_pair_present']=wrong.apply(lambda r:any(str(x).casefold()==str(r.span).casefold() for x in oldwrong.loc[oldwrong.fact_id==r.fact_id,'generated_wrong_answer']),axis=1)
    wrong['historical_entity_pair_present']=wrong.apply(lambda r:any(str(x)==str(r.answer_reference) for x in oldwrong.loc[oldwrong.fact_id==r.fact_id,'wrong_entity']),axis=1)
    csv(wrong.drop(columns=['input_ids','output_ids']),'C2_NEW_MODEL_CONFIRMED_WRONG_EVENTS.csv')
    actual=wrong.groupby(['model','revision','fact_id','subj','span','c3_pair_competitor_emitted','historical_pair_present'],dropna=False).size().reset_index(name='events')
    csv(actual,'C2_NEW_ACTUAL_WRONG_INVENTORY.csv')
    recurring=wrong.groupby(['fact_id','subj','answer_reference']).agg(events=('span','size'),answer_names=('span',lambda x:'|'.join(sorted(set(x)))),models=('model',lambda x:'|'.join(sorted(set(x)))),families=('family',lambda x:'|'.join(sorted(set(x)))),model_count=('model','nunique'),family_count=('family','nunique')).reset_index()
    csv(recurring.sort_values(['family_count','model_count','events'],ascending=False),'C2_RECURRING_ERRORS_ACROSS_MODELS.csv')
    mapped=wrong.merge(oldwrong[['fact_id','wrong_entity','uid']].drop_duplicates(),left_on=['fact_id','answer_reference'],right_on=['fact_id','wrong_entity'],how='inner',suffixes=('','_old'))
    links=mapped[['model','revision','fact_id','subj','span','answer_reference','uid_old']].drop_duplicates().merge(context,left_on=['fact_id','answer_reference'],right_on=['fact_id','wrong_entity'],how='left',suffixes=('','_evidence'))
    links['evidence_corpus']='original-model training corpus, not verified as current-model exposure'
    links['current_model_exposure_verified']=False
    csv(links,'C2_NEW_WRONG_TO_ORIGINAL_CONTEXT_LINKS.csv')
    # Match standalone C1 generation to C3 NoContext scores on exactly the same capital fact.
    base=c1[c1.prop.eq('capital')].groupby(['task_id','model','revision','fact_id','s_uri','decode'])[['correct','error_lower','unverified','wide_error_upper']].mean().reset_index()
    c3=country[(country.relation=='capital') & (country.context=='NoContext')].merge(matching[['target_country','fact_id']].drop_duplicates(),on='target_country',validate='many_to_one')
    common=base.merge(c3,on=['task_id','model','revision','fact_id','decode'],suffixes=('_c1','_c3'),validate='one_to_one')
    csv(common,'C1_C3_COMMON_FACT_PANEL.csv')
    # Country paired change associations are descriptive; N=10 is too small for mechanism claims.
    changes=[]
    for model,g in common.groupby('model'):
        order=g[['task_id','training_progress']].drop_duplicates().sort_values('training_progress').task_id.tolist()
        if len(order)<2:continue
        a,b=order[0],order[-1]
        merged=g[g.task_id==a].merge(g[g.task_id==b],on=['fact_id','target_country','pair_cluster','decode'],suffixes=('_early','_late'),validate='one_to_one')
        for x in merged.to_dict('records'):
            changes.append(dict(model=model,early=a,late=b,fact_id=x['fact_id'],target_country=x['target_country'],pair_cluster=x['pair_cluster'],decode=x['decode'],candidate_sum_change=x['sum_pick_late']-x['sum_pick_early'],candidate_mean_change=x['mean_pick_late']-x['mean_pick_early'],sum_binding_margin_change=x['sum_margin_late']-x['sum_margin_early'],c1_correct_change=x['correct_late']-x['correct_early'],c1_error_change=x['error_lower_late']-x['error_lower_early'],c3_open_correct_change=x['closed_correct_late']-x['closed_correct_early']))
    changes=pd.DataFrame(changes);csv(changes,'C1_C3_COUNTRY_CHANGES.csv')
    change_ci=[];associations=[]
    for key,g in changes.groupby(['model','early','late','decode']):
        meta=dict(zip(['model','early','late','decode'],key))
        for metric in ['candidate_sum_change','candidate_mean_change','sum_binding_margin_change','c1_correct_change','c1_error_change','c3_open_correct_change']:
            for cluster,values in [('country',g[metric]),('pair',g.groupby('pair_cluster')[metric].mean())]:
                change_ci.append(dict(**meta,metric=metric,cluster=cluster,**interval(values)))
        # Associations are descriptive, with each family/stage analyzed separately.
        for outcome in ['c1_error_change','c1_correct_change']:
            x=g.sum_binding_margin_change.to_numpy();y=g[outcome].to_numpy()
            for cluster in ['country','pair']:
                if cluster=='country':idx=boot_indices(len(g))
                else:
                    paired=[np.flatnonzero(g.pair_cluster.to_numpy()==k) for k in sorted(g.pair_cluster.unique())]
                    assert len({len(v) for v in paired})==1
                    idx=np.asarray(paired)[boot_indices(len(paired))].reshape(NBOOT,-1)
                bx=x[idx];by=y[idx];bx=bx-bx.mean(1,keepdims=True);by=by-by.mean(1,keepdims=True)
                denom=np.sqrt((bx*bx).sum(1)*(by*by).sum(1))
                draws=np.divide((bx*by).sum(1),denom,out=np.full(NBOOT,np.nan),where=denom>1e-14)
                draws=draws[np.isfinite(draws)]
                corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-14 and np.std(y)>1e-14 else np.nan
                associations.append(dict(**meta,outcome=outcome,cluster=cluster,subjects=len(g),pearson_r=corr,ci_low=float(np.quantile(draws,.025)) if len(draws) else np.nan,ci_high=float(np.quantile(draws,.975)) if len(draws) else np.nan,valid_bootstraps=len(draws),interpretation='exploratory association; not predictive validation or causal identification'))
    csv(pd.DataFrame(change_ci),'C1_C3_CHANGE_CLUSTER_CI.csv')
    csv(pd.DataFrame(associations),'C1_C3_CHANGE_ASSOCIATIONS.csv')
    coverage=dict(c3_countries=int(matching.target_country.nunique()),matched_original_c2_facts=int(matching.loc[matching.c2_original_evidence_available,'fact_id'].nunique()),c3_competitors_matching_historical_wrong=int(matching.loc[matching.competitor_matches_historical_actual_wrong,'fact_id'].nunique()),matched_C1_C3_facts=int(common.fact_id.nunique()),original_c2_facts=int(evidence.fact_id.nunique()),new_model_confirmed_capital_error_events=len(wrong),new_model_distinct_wrong_name_pairs=len(actual[['fact_id','span']].drop_duplicates()),new_model_distinct_wrong_entity_pairs=len(recurring),new_model_error_events_matching_original_entity_pair=int(wrong.historical_entity_pair_present.sum()),cross_model_exposure_timeline_available=False)
    (OUT/'C2_MATCH_COVERAGE.json').write_text(json.dumps(coverage,indent=2))
    return coverage


def main():
    OUT.mkdir(exist_ok=True)
    cfg,tasks,integrity=check_inputs();print('INPUTS_VERIFIED',len(integrity),flush=True)
    refs,audit=audit_factory();c1=c1_analysis(audit);print('C1_AUDITED',len(c1),flush=True)
    scores,opened,country,summary=c3_analysis(cfg,tasks,refs,audit);print('C3_ANALYZED',len(scores),len(opened),flush=True)
    coverage=c2_and_joint(c1,scores,opened,country);print('MATCHED',json.dumps(coverage),flush=True)
    (OUT/'ANALYSIS_STATUS.json').write_text(json.dumps(dict(completed_checkpoints=len(integrity),c1_outputs=len(c1),c3_raw_queries=len(scores)*2,c3_unique_queries=len(scores),c3_raw_open_outputs=len(opened)*2,c3_unique_open_outputs=len(opened),hypotheses_changed=False,exploratory=True,c2_cross_model_explanation_complete=False,internal_mechanism_proven=False),indent=2))


if __name__=='__main__':
    main()
