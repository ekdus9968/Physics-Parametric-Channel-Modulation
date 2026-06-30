import json
import os

base_path = r'C:\Users\thisi\OneDrive\Documents\ERA-marin\PPCM\data\S-UODAC2020'

for type_name in ['type1','type2','type3','type4','type5','type6','type7']:
    ann_path = os.path.join(base_path, type_name, 'annotations', 'instances.json')
    if not os.path.exists(ann_path):
        print(f"{type_name}: 파일 없음")
        continue
    with open(ann_path, 'r') as f:
        data = json.load(f)
    print(f"{type_name}: images={len(data['images'])}, annotations={len(data['annotations'])}")

print("\nCategories:")
with open(os.path.join(base_path, 'type1', 'annotations', 'instances.json'), 'r') as f:
    data = json.load(f)
print(data['categories'])