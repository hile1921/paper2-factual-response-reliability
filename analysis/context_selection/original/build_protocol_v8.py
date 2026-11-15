#!/usr/bin/env python3
"""C3 v8: length-matched bare mention, to close the one uncontrolled variable in v7.

In v7 the bare-mention arm ("Suva is a city.") was shorter than the relation arm
("The capital of Fiji is Suva."), so "length" remained a rival explanation for why bare
mention produced the most wrong answers.

v8 adds:
  G_matched : the bare mention padded with neutral words until its character length is
              within 2 characters of the relation sentence for the SAME case.
  C_padded  : the relation sentence padded by the same amount, so padding itself is
              controlled rather than assumed inert.

Arms: B_filler, C_true, C_padded, G_short, G_matched, plus A_none.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEAS = ROOT / 'next_stage' / 'c2_capital_feasibility' / 'PAIR_CAPITAL_FEASIBILITY.csv'
OUT = ROOT / 'next_stage' / 'c3_realwrong_v8'
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATES = ["The capital of {s} is", "{s} has its capital at"]
FILL = ["The weather in the region changes with the season.",
        "Trade routes in the area developed over many centuries.",
        "Local languages vary between neighbouring districts."]
PAD = [" of note", " indeed", " as is often said", " in many respects",
       " of considerable renown", " as widely recorded", " in older sources",
       " according to common accounts"]

feas = pd.read_csv(FEAS, keep_default_na=False, low_memory=False)
use = feas[feas.is_capital == True].drop_duplicates(['subject', 'generated_wrong_answer'])
print(f'pairs={len(use)}  subjects={use.subject.nunique()}')


def match_pad(base, target_len):
    """Grow `base` with neutral padding until |len - target| is minimal."""
    best = (base, abs(len(base) - target_len))
    for p in PAD:
        if abs(len(base + p) - target_len) < best[1]:
            best = (base + p, abs(len(base + p) - target_len))
    for p1 in PAD:
        for p2 in PAD:
            if p1 == p2:
                continue
            c = base + p1 + p2
            if abs(len(c) - target_len) < best[1]:
                best = (c, abs(len(c) - target_len))
    return best


probes = []
audit = []
for r in use.itertuples():
    target, gold, wrong, comp = r.subject, r.correct_answer, r.generated_wrong_answer, r.wrong_of
    c_sent = f"The capital of {comp} is {wrong}."
    g_short = f"{wrong} is a city."
    g_matched, diff = match_pad(g_short, len(c_sent))
    c_padded = c_sent + " " + "Of this there is little doubt."
    audit.append(dict(subject=target, wrong_answer=wrong, competitor=comp,
                      len_C=len(c_sent), len_G_short=len(g_short),
                      len_G_matched=len(g_matched), match_diff=diff,
                      G_matched=g_matched, C_padded=c_padded))
    for ti in range(2):
        q = TEMPLATES[ti].format(s=target)
        base = dict(target_country=target, gold=gold, wrong_answer=wrong, wrong_of=comp,
                    template=ti, prompt=q, source_fact_id=r.fact_id, source_uid=r.uid)
        for arm, ctx in [
            ('A_none', ''),
            ('B_filler', '\n'.join(FILL)),
            ('C_true', c_sent),
            ('C_padded', c_padded),
            ('G_short', g_short),
            ('G_matched', g_matched),
        ]:
            probes.append(dict(id=f"v8|{target}|{wrong}|t{ti}|{arm}", experiment='arms',
                               arm=arm, context=ctx, **base))

A = pd.DataFrame(audit)
A.to_csv(OUT / 'LENGTH_MATCH_AUDIT.csv', index=False, encoding='utf-8')
print(f'\nlength matching (G_matched vs C_true):')
print(f'  mean |diff| = {A.match_diff.mean():.2f} chars, max = {A.match_diff.max()}')
print(f'  mean len_C={A.len_C.mean():.1f}  len_G_short={A.len_G_short.mean():.1f}  '
      f'len_G_matched={A.len_G_matched.mean():.1f}')
print(f'  cases with |diff| <= 2: {int((A.match_diff <= 2).sum())}/{len(A)}')

cfg = dict(
    version='c3-realwrong-v8',
    frozen_at='2026-11-13T08:10:00+08:00',
    purpose=('close the length confound between the bare-mention arm and the relation arm '
             'that remained open in v7'),
    n_pairs=int(len(use)), n_subjects=int(use.subject.nunique()), n_probes=len(probes),
    templates=TEMPLATES, filler=FILL, probes=probes,
    arms=dict(A_none='no context', B_filler='three neutral sentences',
              C_true='capital of its real country is the wrong city',
              C_padded='C_true plus a neutral closing clause (padding control)',
              G_short='the wrong city is a city (shorter than C)',
              G_matched='the wrong city is a city, padded to C length'),
    contrasts={'G_matched_minus_C_true': 'bare mention vs relation, length matched',
               'G_short_minus_G_matched': 'effect of padding alone within the bare arm',
               'C_padded_minus_C_true': 'effect of padding alone within the relation arm'},
)
(OUT / 'C3_REALWRONG_V8_PROTOCOL.json').write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'\nprobes={len(probes)}')
from collections import Counter
print(Counter(p['arm'] for p in probes))
