"""Replay the archived offline C1 evidence gate and join all raw generations.

This is a deterministic machine-assisted replay against saved reference snapshots,
not new human review or a refresh of current-world truth.
"""
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
import ast
import gzip
import hashlib
import importlib.util
import json
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis/training_dynamics'
OUT = ROOT / 'reproduction_outputs/source_verification'
OUT.mkdir(parents=True, exist_ok=True)
PROV = BASE / 'label_provenance'

spec = importlib.util.spec_from_file_location('archived_verifier', PROV / 'verify_analyze.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

# Select only the original pure functions, avoiding the historical GPU/archive
# entry points and the optional pyarrow import in run_p1.py.
tree = ast.parse((PROV / 'run_p1.py').read_text(encoding='utf-8-sig'))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ['normalize', 'candidate']]
ns = {'re': re, 'unicodedata': unicodedata}
exec(compile(tree, str(PROV / 'run_p1.py'), 'exec'), ns)
tree = ast.parse((BASE / 'original/analyze_snapshot.py').read_text(encoding='utf-8-sig'))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'audit_factory']
ns.update(References=verifier.References, source_extractor=verifier.source_extractor,
          SimpleNamespace=SimpleNamespace, lru_cache=lru_cache)
exec(compile(tree, str(BASE / 'original/analyze_snapshot.py'), 'exec'), ns)
refs, audit = ns['audit_factory']()

frozen = pd.read_csv(BASE / 'data/C1_AUDITED_OUTPUTS.csv.gz', keep_default_na=False,
                     low_memory=False)
assert len(frozen) == 319410 and frozen.uid.nunique() == 273
assert frozen.task_id.nunique() == 117
groups = {task: frame for task, frame in frozen.groupby('task_id', sort=False)}
keys = ['task_id', 'uid', 'template', 'mode', 'seed', 'prompt', 'output']
differences = []
counts = {}
matched = 0
for path in sorted((BASE / 'data/raw').glob('*.json.gz')):
    raw = json.loads(gzip.decompress(path.read_bytes()).decode('utf-8-sig'))
    task = raw['task']['id']
    old = groups[task].reset_index(drop=True)
    assert len(raw['records']) == len(old), task
    for index, record in enumerate(raw['records']):
        expected = old.iloc[index]
        assert all(str(record[key]) == str(expected[key]) for key in keys), (task, index, 'raw join')
        result = audit(record['subj'], record['s_uri'], record['prop'], record['output'])
        for key in ['audit_label', 'audit_reason', 'span', 'subject_reference',
                    'answer_reference', 'claim', 'cross_reference']:
            if str(result[key]) != str(expected[key]):
                differences.append({'task_id': task, 'index': index, 'field': key,
                                    'archived': str(expected[key]), 'replayed': str(result[key])})
        matched += 1
    counts[task] = len(old)
    print(f'{task}: joined and replayed {len(old)} rows', flush=True)
assert matched == len(frozen)
pd.DataFrame(differences, columns=['task_id', 'index', 'field', 'archived', 'replayed']).to_csv(
    OUT / 'REPLAY_DIFFERENCES.csv', index=False)
report = {'raw_generations_joined': matched, 'tasks': len(counts), 'facts': 273,
          'replay_field_differences': len(differences), 'passed': not differences,
          'reference_route': 'Frozen offline Wikidata and GeoNames snapshots',
          'human_review_performed': False, 'network_used': False,
          'per_task_rows': counts}
(OUT / 'SOURCE_VERIFICATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert not differences, f'Replay differences retained under {OUT}'
print('All 319410 raw outputs join and replay the frozen offline audit fields exactly.')
