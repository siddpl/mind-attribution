import csv
from collections import defaultdict
import statistics

rows = list(csv.DictReader(open('referent_ladder_results.csv')))
tier_cols = ['experiential', 'first_person', 'human_reference']

by_layer = defaultdict(list)
for r in rows:
    by_layer[r['layer']].append(r)

print("Mean |affirm_score| by layer (magnitude - where is there real signal to work with):")
header = f"{'layer':>6}  " + "  ".join(f"{t:>15}" for t in tier_cols)
print(header)
for layer in sorted(by_layer, key=int):
    line = f"{layer:>6}  "
    for t in tier_cols:
        vals = [abs(float(r[f'affirm_{t}_score'])) for r in by_layer[layer]]
        line += f"{statistics.mean(vals):>15.6f}  "
    print(line)
