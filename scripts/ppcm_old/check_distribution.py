# scripts/check_distribution.py
import json
import os
from collections import defaultdict

base_path = 'data/S-UODAC2020'

with open('data/S-UODAC2020/COCO_Annotations/instances_source.json', 'r') as f:
    data = json.load(f)

type_count = defaultdict(int)
not_found = 0

for img in data['images']:
    fname = img['file_name']
    found = False
    for t in ['type1','type2','type3','type4','type5','type6','type7']:
        if os.path.exists(os.path.join(base_path, t, fname)):
            type_count[t] += 1
            found = True
            break
    if not found:
        not_found += 1

print('=== 이미지 분포 ===')
for t in sorted(type_count.keys()):
    print(f"{t}: {type_count[t]}장")
print(f"못 찾음: {not_found}장")
print(f"총합: {sum(type_count.values())}장")