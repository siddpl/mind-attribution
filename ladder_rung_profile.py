import csv
from collections import defaultdict
import statistics

rows = list(csv.DictReader(open('referent_ladder_results.csv')))
tier_cols = ['experiential', 'first_person', 'human_reference']

BAND = set(str(l) for l in range(8, 25))
filtered = [r for r in rows if r['layer'] in BAND]

by_rung = defaultdict(lambda: defaultdict(list))
for r in filtered:
    key = (r['ladder'], int(r['rung']), r['rung_id'])
    for t in tier_cols:
        by_rung[key][t].append(float(r[f'{t}_separation']))

print("Mean separation (affirm - deny) by rung, layers 8-24, all 4 claims pooled:")
for ladder in ('person', 'scope'):
    print(f"\nLadder: {ladder}")
    keys = sorted([k for k in by_rung if k[0] == ladder], key=lambda k: k[1])
    for key in keys:
        _, rung, rung_id = key
        line = f"  rung {rung} ({rung_id:>20}): "
        for t in tier_cols:
            vals = by_rung[key][t]
            line += f"{t}={statistics.mean(vals):+.5f}  "
        print(line)
