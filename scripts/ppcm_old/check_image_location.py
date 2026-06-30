# scripts/check_image_location.py
import json
import os

base_path = 'data/S-UODAC2020'

with open('data/S-UODAC2020/COCO_Annotations/instances_source.json', 'r') as f:
    data = json.load(f)

# fine which type folder from first 5 images
for img in data['images'][:5]:
    fname = img['file_name']
    found = False
    for t in ['type1','type2','type3','type4','type5','type6','type7']:
        full_path = os.path.join(base_path, t, fname)
        if os.path.exists(full_path):
            print(f"{fname} → {t}")
            found = True
            break
    if not found:
        print(f"{fname} → 못 찾음")