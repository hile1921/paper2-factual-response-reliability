#!/usr/bin/env python3
"""C3 v9: the confirmatory round, on held-out pairs, with the v8 defects fixed.

Fixes relative to v8:
  * held-out pairs only (never used in v5-v8),
  * every context ends with the wrong city, so city position is constant,
  * no sentence fragments: one sentence, no padding appended after a period,
  * the relation arm and the bare arm have the same word count and the same syntactic
    shape (determiner + subject + copula + city),
  * historical and subnational entities are excluded, because "capital of the Ukrainian
    Soviet Socialist Republic" is not a modern capital relation.

Design per pair, all seven words and city-final:
  C_true  "The capital city of {competitor} is {wrong}."
  F_false "The capital city of {other}      is {wrong}."   same shape, false country
  G_bare  "Another city in the region is {wrong}."          no country, no relation
  B_filler three neutral sentences, no city
  A_none  no context
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / 'next_stage' / 'c3_confirm' / 'CONFIRM_CANDIDATES.csv'
PANEL = ROOT / 'next_stage' / 'cross_model_snapshot_117_20261113' / 'raw' / 'FACT_PANEL.csv'
OUT = ROOT / 'next_stage' / 'c3_confirm'

TEMPLATES = ["The capital of {s} is", "{s} has its capital at"]
FILL = ["The weather in the region changes with the season.",
        "Trade routes in the area developed over many centuries.",
        "Local languages vary between neighbouring districts."]

# entities whose "capital" is historical or subnational; they cannot carry a modern relation
EXCLUDE_COUNTRY = {
    'ukrainian soviet socialist republic', 'weimar republic', 'khyber pakhtunkhwa',
    'soviet union', 'czechoslovakia', 'yugoslavia', 'commonwealth of england',
}

cand = pd.read_csv(CAND, keep_default_na=False, low_memory=False)
cand['country_norm'] = cand.country_of_wrong.map(lambda s: re.sub(r'\s+', ' ', str(s).lower()).strip())
before = len(cand)
cand = cand[~cand.country_norm.isin(EXCLUDE_COUNTRY)].copy()
print(f'candidates {before} -> {len(cand)} after excluding historical/subnational entities')
print(f'  pairs={len(cand)}  subjects={cand.subject.nunique()}  '
      f'new subjects={cand[~cand.subject_in_used].subject.nunique()}')

panel = pd.read_csv(PANEL, keep_default_na=False, low_memory=False)
countries = sorted({str(c).strip() for c in panel[panel.prop == 'capital'].subj.unique()
                    if str(c).strip()
                    and re.sub(r'\s+', ' ', str(c).lower()).strip() not in EXCLUDE_COUNTRY})

probes, chosen = [], []
for r in cand.itertuples():
    target, wrong, comp = r.subject, r.wrong_answer, r.country_of_wrong
    gold = panel[(panel.subj == target) & (panel.prop == 'capital')].gold.iloc[0]
    others = [c for c in countries if c not in (target, comp)]
    if not others:
        continue
    # match the false country by WORD count first (so C and F are the same length),
    # then by character length
    other = min(others, key=lambda c: (abs(len(c.split()) - len(comp.split())),
                                       abs(len(c) - len(comp)), c))
    # pad the bare mention by exactly (words(comp) - 1) so that words(G) == words(C)
    k = max(0, len(comp.split()) - 1)
    pad = {0: '', 1: ' wider', 2: ' wider surrounding',
           3: ' wider surrounding nearby', 4: ' wider surrounding nearby coastal',
           5: ' wider surrounding nearby coastal outer'}.get(k, ' wider' * k)
    g_sent = f"Another city in the{pad} region is {wrong}."
    chosen.append(dict(subject=target, wrong_answer=wrong, competitor=comp,
                       false_country=other, new_subject=bool(not r.subject_in_used)))
    for ti in range(2):
        q = TEMPLATES[ti].format(s=target)
        base = dict(target_country=target, gold=gold, wrong_answer=wrong, wrong_of=comp,
                    false_country=other, template=ti, prompt=q,
                    source_uid=r.uid, source_fact_id=r.fact_id)
        for arm, ctx in [
            ('A_none', ''),
            ('B_filler', '\n'.join(FILL)),
            ('C_true', f"The capital city of {comp} is {wrong}."),
            ('F_false', f"The capital city of {other} is {wrong}."),
            ('G_bare', g_sent),
        ]:
            probes.append(dict(id=f"v9|{target}|{wrong}|t{ti}|{arm}", experiment='arms',
                               arm=arm, context=ctx, context_words=len(ctx.split()),
                               city_final=bool(ctx) and ctx.rstrip().endswith(wrong + '.'),
                               **base))

C = pd.DataFrame(chosen)
C.to_csv(OUT / 'CONFIRM_PAIRS.csv', index=False, encoding='utf-8')
print(f'\npairs used: {len(C)}  subjects={C.subject.nunique()}  '
      f'new subjects={C[~C.new_subject].new_subject.sum() if False else int(C.new_subject.sum())}')

checks = pd.DataFrame(probes)
for arm in ('C_true', 'F_false', 'G_bare'):
    a = checks[checks.arm == arm]
    print(f'  {arm:9s} word counts: {sorted(a.context_words.unique())}  '
          f'city-final: {bool(a.city_final.all())}')

cfg = dict(
    version='c3-confirm-v9',
    frozen_at='2026-11-13T08:40:00+08:00',
    purpose=('confirm the bare-mention versus relation effect on held-out pairs with city '
             'position, sentence count and word count held constant and no fragments'),
    n_pairs=int(len(C)), n_subjects=int(C.subject.nunique()),
    n_new_subjects=int(C.new_subject.sum()), n_probes=len(probes),
    primary_contrast='G_bare minus C_true, outcome = designated_wrong',
    stopping_rule=('report the primary contrast per checkpoint condition with a cluster '
                   'permutation test and Holm across the four conditions; if the direction '
                   'is consistent and the intervals exclude zero the effect replicates; '
                   'otherwise report the inconsistency'),
    templates=TEMPLATES, filler=FILL, probes=probes,
    arms=dict(A_none='no context', B_filler='three neutral sentences',
              C_true='the wrong city as the capital of its real country (city final, 7 words)',
              F_false='the wrong city as the capital of a matched unrelated country (7 words)',
              G_bare='the wrong city as another city, no country, no relation (7 words)'),
    excluded_entities=sorted(EXCLUDE_COUNTRY),
)
(OUT / 'C3_CONFIRM_V9_PROTOCOL.json').write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'\nprobes={len(probes)}')
from collections import Counter
print(Counter(p['arm'] for p in probes))
