"""Portable CPU entry points for the frozen manuscript analyses.

Original scripts remain byte-identical under analysis/*/original. Adaptations here
change paths, input compression and orchestration, not statistical definitions.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'reproduction_outputs'


def require_replace(text, old, new):
    assert old in text, f'Original source anchor not found: {old[:100]}'
    return text.replace(old, new)


def execute(text, path, log):
    path.write_text(text, encoding='utf-8')
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run([sys.executable, '-X', 'utf8', str(path)],
                                cwd=path.parent, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}; inspect {log}')


def compare(expected, actual, allow_legacy_encoding=False):
    import pandas as pd
    results = []
    for source in sorted(expected.glob('*.csv')):
        target = actual / source.name
        if not target.exists():
            continue
        left = pd.read_csv(source, keep_default_na=False)
        right = pd.read_csv(target, keep_default_na=False)
        try:
            pd.testing.assert_frame_equal(left, right, check_dtype=False,
                                          check_exact=False, rtol=1e-8, atol=1e-9)
        except AssertionError as error:
            # Two archived single-case CSVs were produced by decoding UTF-8
            # JSON as GBK on Windows. Report that exact reversible discrepancy;
            # do not silently alter frozen text or tolerate other differences.
            legacy = False
            changed = []
            if allow_legacy_encoding and left.shape == right.shape and list(left) == list(right):
                changed = [c for c in left if not left[c].equals(right[c])]
                if changed == ['text']:
                    def old_decode(value):
                        try:
                            return value.encode('utf-8').decode('gbk')
                        except (UnicodeError, AttributeError):
                            return value
                    legacy = all(x == y or x == old_decode(y)
                                 for x, y in zip(left.text, right.text))
            results.append({'table': source.name, 'pass': bool(legacy),
                            'strict_text_match': False,
                            'encoding_only_warning': bool(legacy),
                            'changed_text_rows': int((left.text != right.text).sum()) if legacy else None,
                            'detail': ('Archived UTF-8-as-GBK mojibake; every non-text field matches. '
                                       'Recomputed output uses the original UTF-8 JSON.'
                                       if legacy else str(error)[:1600])})
        else:
            results.append({'table': source.name, 'pass': True, 'rows': len(left)})
    return results


def run(phase):
    base = ROOT / 'analysis' / phase
    work = ART / phase
    work.mkdir(parents=True, exist_ok=True)
    original = base / 'original'
    start = time.monotonic()
    if phase == 'training_dynamics':
        for filename in ['c1_abe.py', 'c1_final.py']:
            text = (original / filename).read_text(encoding='utf-8-sig')
            text = require_replace(text,
                "SNAP = sys.argv[1] if len(sys.argv) > 1 else 'cross_model_snapshot_91_20261113'"
                if filename == 'c1_abe.py' else
                "SNAP = sys.argv[1] if len(sys.argv) > 1 else 'cross_model_snapshot_117_20261113'",
                "SNAP = 'cross_model_snapshot_117_20261113'")
            text = require_replace(text, "T = ROOT / 'next_stage' / SNAP / 'tables'",
                                   f'T = Path({str(base / "data")!r})')
            out_line = ("OUT = ROOT / 'next_stage' / 'c1_abe' / SNAP"
                        if filename == 'c1_abe.py' else
                        "OUT = ROOT / 'next_stage' / 'c1_final' / SNAP")
            text = require_replace(text, out_line, f'OUT = Path({str(work)!r})')
            text = require_replace(text, "T / 'C1_AUDITED_OUTPUTS.csv'",
                                   "T / 'C1_AUDITED_OUTPUTS.csv.gz'")
            execute(text, work / filename, work / (filename + '.log'))
    elif phase in ['matched_preferences', 'positive_control']:
        shutil.copytree(base / 'data', work, dirs_exist_ok=True)
        if phase == 'matched_preferences':
            source = (original / 'build.py').read_text(encoding='utf-8-sig')
            coverage = source.split("available=match[['uid','step']].drop_duplicates()", 1)[1].split('inputs=dict(', 1)[0]
            text = ("from pathlib import Path\nimport numpy as np,pandas as pd\n"
                    "R=Path(__file__).resolve().parent\n"
                    "match=pd.read_csv(R/'CANONICAL_GENERATIONS.csv',keep_default_na=False)\n"
                    "g=pd.read_csv(R/'inputs/gold_summary.csv')\n"
                    "match[match.prompt.ne('The capital of '+match.subject+' is')].to_csv(R/'PROMPT_MISMATCHES.csv',index=False)\n"
                    "available=match[['uid','step']].drop_duplicates()" + coverage)
            execute(text, work / 'coverage.py', work / 'coverage.log')
        scripts = ['analyze.py', 'followup_diagnostics.py'] if phase == 'matched_preferences' else ['analyze.py']
        for filename in scripts:
            execute((original / filename).read_text(encoding='utf-8-sig'),
                    work / filename, work / (filename + '.log'))
    elif phase == 'context_selection':
        text = (original / 'final_contrasts.py').read_text(encoding='utf-8-sig')
        text = require_replace(text, "OUT = ROOT / 'next_stage' / 'c3_final'",
                               f'OUT = Path({str(work)!r})')
        text = require_replace(text,
            "PANEL = ROOT / 'next_stage' / 'cross_model_snapshot_117_20261113' / 'raw' / 'FACT_PANEL.csv'",
            f'PANEL = Path({str(base / "data/FACT_PANEL.csv")!r})')
        old = """    files = sorted(cfg['raw'].glob(cfg['prefix'] + '*.parquet'))
    if not files:
        print(f'  {rname}: no parquet files found, skipped')
        continue
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)"""
        new = (f"    csv_root = Path({str(base / 'data')!r})\n"
               "    d = pd.read_csv(csv_root / ('raw_v8.csv.gz' if rname == 'v8_development' else 'raw_v9.csv.gz'), keep_default_na=False)")
        text = require_replace(text, old, new)
        execute(text, work / 'final_contrasts.py', work / 'final_contrasts.log')
    elif phase == 'forward_prediction':
        # Fit the original models, then use the archived vectorised postprocessor.
        # The historical one-file script contains a superseded slow micro-only
        # bootstrap section. The published tables use c2_forward_post.py instead.
        text = (original / 'c2_forward_prediction.py').read_text(encoding='utf-8-sig')
        text = text.split('# Paired subject bootstrap: one draw matrix for all models/metrics.')[0]
        text = require_replace(text,
            'OUT = ROOT / "next_stage" / "C2_FORWARD_PREDICTION_V1_20261113"',
            f'OUT = Path({str(work)!r})')
        text = require_replace(text,
            'SRC = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113" / "Paper2_PhaseI_Repro" / "inputs" / "model_frame_42_by_5.csv"',
            f'SRC = Path({str(base / "data/model_frame_42_by_5.csv")!r})')
        text += "\nnp.save(OUT / 'C2_FORWARD_BOOTSTRAP_DRAWS.npy', np.random.default_rng(20261113).integers(0, 42, size=(20000, 42)))\n"
        execute(text, work / 'fit.py', work / 'fit.log')
        for filename in ['c2_forward_post.py', 'c2_forward_sensitivity.py']:
            execute((original / filename).read_text(encoding='utf-8-sig'),
                    work / filename, work / (filename + '.log'))
    else:
        raise ValueError(phase)
    checks = compare(base / 'expected', work, allow_legacy_encoding=phase == 'positive_control')
    assert checks, f'No output tables compared for {phase}'
    report = {'phase': phase, 'seconds': round(time.monotonic() - start, 2),
              'comparisons': checks, 'passed': all(x['pass'] for x in checks),
              'archived_tables_not_recomputed': sorted(
                  {p.name for p in (base / 'expected').glob('*.csv')} -
                  {x['table'] for x in checks}),
              'scope': 'Recompute saved analyses with the original rules; no GPU or network.'}
    (work / 'COMPARISON.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}), flush=True)
    if not report['passed']:
        raise RuntimeError(f'Output differences retained in {work / "COMPARISON.json"}')
    return report


if __name__ == '__main__':
    phases = ['training_dynamics', 'matched_preferences', 'context_selection',
              'forward_prediction', 'positive_control']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=phases + ['all'])
    args = parser.parse_args()
    reports = [run(p) for p in phases if args.phase in ['all', p]]
    (ART / 'REPRODUCTION_REPORT.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
