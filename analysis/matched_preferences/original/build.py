from pathlib import Path
import json,hashlib,sys,importlib.util,shutil
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;S=R.parent;P=S.parent
if (R/'INPUT_FREEZE.json').exists():raise SystemExit('Inputs already frozen; analyze saved snapshots')
try:import pyarrow
except ImportError:
 lib=P/'figures/_plotdeps/pyarrow';sp=importlib.util.spec_from_file_location('pyarrow',lib/'__init__.py',submodule_search_locations=[str(lib)])
 m=importlib.util.module_from_spec(sp);sys.modules['pyarrow']=m;sp.loader.exec_module(m)
source={
 'classified_records.parquet':S/'c1_taxonomy_v1/classified_records.parquet',
 'gold_summary.csv':S/'c2_bridge_20261114/FACT_CHECKPOINT_BRIDGE.csv',
 'candidates85.csv':S/'c2_bridge_20261114/inputs/candidates.csv',
 'candidates_augmented.csv':S/'c2_bridge_20261114/CANDIDATES_AUGMENTED.csv',
 'scope_registry.csv':S/'c2_c3_design/evidence_timeline_90_by_5.csv',
 'old_matched_errors.csv':S/'c2_bridge_20261114/MATCHED_ERROR_EVENTS.csv',
 'wrong_pair_inventory.csv':S/'c2_wrong_relations/wrong_answer_pair_inventory.csv',
 'scoring_source.py':Path('C:/Users/Administrator/Desktop/jiemi/C2_V2_FLOW_20261113/06_scripts/c2_v2_logprob.py'),
 'generation_source.py':P/'p1_confirmation/generate_original.py',
 'generation_protocol.md':P/'p1_confirmation/PROTOCOL.md',
 'positive_control_status.json':S/'c2_positive_control_20261115/ANALYSIS_STATUS.json'}
snap=R/'inputs';snap.mkdir(exist_ok=True);manifest=[]
for name,path in source.items():
 assert path.exists(),path
 shutil.copyfile(path,snap/name);manifest.append(dict(snapshot=name,original_path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
raw=pd.read_parquet(snap/'classified_records.parquet');assert len(raw)==11370 and raw.review_id.is_unique
assert raw.model.nunique()==1 and raw.model.iloc[0]=='LMEnt-1B-6E'
g=pd.read_csv(snap/'gold_summary.csv');c=pd.read_csv(snap/'candidates85.csv',keep_default_na=False)
assert len(g)==450 and g.uid.nunique()==90 and len(c)==38250
scope=pd.read_csv(snap/'scope_registry.csv')[['uid','scope_grade']].drop_duplicates();assert scope.uid.is_unique
cap=raw[(raw.prop=='capital')&(raw.template=='t1')].copy();assert len(cap)==1960
match=cap.merge(g[['uid','step','subject']],on=['uid','step'],validate='many_to_one',indicator=True)
assert match._merge.eq('both').all();match=match.drop(columns='_merge')
match['expected_prompt']='The capital of '+match.subject+' is'
bad=match[match.prompt.ne(match.expected_prompt)]
bad.to_csv(R/'PROMPT_MISMATCHES.csv',index=False);assert bad.empty
assert match.subj.eq(match.subject).all()
gold_names=c[['uid','gold']].drop_duplicates();assert gold_names.uid.is_unique
check=match[['uid','gold']].drop_duplicates().merge(gold_names,on='uid',suffixes=('_generation','_score'),validate='one_to_one')
assert check.gold_generation.eq(check.gold_score).all()
match.to_csv(R/'CANONICAL_GENERATIONS.csv',index=False,encoding='utf-8-sig')
facts=match[['uid','s_uri','subject','gold','cohort']].drop_duplicates().merge(scope,on='uid',validate='one_to_one')
assert facts.uid.is_unique and len(facts)==90
facts.to_csv(R/'FACT_REGISTRY.csv',index=False)
available=match[['uid','step']].drop_duplicates()
cov=g[['uid','step','subject']].merge(available.assign(generation_available=True),on=['uid','step'],how='left',validate='one_to_one')
cov['generation_available']=cov.generation_available.eq(True)
cov['reason']=np.where(cov.generation_available,'exact canonical prompt; 1greedy+4samples','original cohort has no generation at intermediate checkpoint; no imputation')
cov.to_csv(R/'COVERAGE_GRID.csv',index=False)
cov[~cov.generation_available].to_csv(R/'MISSING_GENERATION_CELLS.csv',index=False)
coverage=cov.groupby('step').agg(scored_cells=('uid','size'),generated_cells=('generation_available','sum')).reset_index();coverage.to_csv(R/'COVERAGE_BY_STAGE.csv',index=False)
inputs=dict(sources=manifest,raw_rows=11370,canonical_capital_rows=1960,excluded_capital_t2_rows=int(((raw.prop=='capital')&(raw.template=='t2')).sum()),excluded_other_relation_rows=int((raw.prop!='capital').sum()),matched_cells=len(available),missing_cells=int((~cov.generation_available).sum()),facts=90,primary_national_facts=int(facts.scope_grade.eq('national_capital_name_concordant').sum()),analysis_plan_sha256=hashlib.sha256((R/'ANALYSIS_PLAN.md').read_bytes()).hexdigest(),new_generation=0,new_training=0)
(R/'INPUT_FREEZE.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8');print(coverage.to_string(index=False));print(json.dumps({k:v for k,v in inputs.items() if k!='sources'},indent=2))
