import json

with open('data/S-UODAC2020/COCO_Annotations/instances_source.json', 'r') as f:
    data = json.load(f)

print('처음 10개 file_name:')
for img in data['images'][:10]:
    print(img['file_name'])
    