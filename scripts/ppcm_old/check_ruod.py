import json

with open('data/RUOD/RUOD_ANN/instances_train.json') as f:
    d = json.load(f)

print('categories:', d['categories'])
print('train images:', len(d['images']))
print('train annotations:', len(d['annotations']))
print('sample filename:', d['images'][0]['file_name'])

with open('data/RUOD/RUOD_ANN/instances_test.json') as f:
    t = json.load(f)

print('test images:', len(t['images']))
print('test sample filename:', t['images'][0]['file_name'])