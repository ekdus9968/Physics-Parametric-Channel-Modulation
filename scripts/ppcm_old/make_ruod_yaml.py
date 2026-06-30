import json
import os
import shutil
from tqdm import tqdm

BASE_PATH = 'data/RUOD'
YOLO_PATH = 'data/RUOD_YOLO'

# RUOD categories
CATEGORIES = {
    1: 'holothurian',
    2: 'echinus',
    3: 'scallop',
    4: 'starfish',
    5: 'fish',
    6: 'corals',
    7: 'diver',
    8: 'cuttlefish',
    9: 'turtle',
    10: 'jellyfish'
}

def coco_to_yolo(ann_file, img_dir, out_img_dir, out_lbl_dir):
    """Convert COCO format to YOLO format."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)

    # Build image info dict
    img_info = {img['id']: img for img in data['images']}

    # Build annotations per image
    ann_per_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_per_img:
            ann_per_img[img_id] = []
        ann_per_img[img_id].append(ann)

    print(f"Converting {len(data['images'])} images...")
    for img in tqdm(data['images']):
        img_id   = img['id']
        fname    = img['file_name']
        W, H     = img['width'], img['height']

        # Copy image
        src = os.path.join(img_dir, fname)
        dst = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

        # Write YOLO label
        lbl_path = os.path.join(out_lbl_dir,
                                fname.replace('.jpg', '.txt'))
        anns = ann_per_img.get(img_id, [])

        with open(lbl_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id'] - 1  # 0-indexed
                x, y, w, h = ann['bbox']

                # Normalize to [0,1]
                cx = (x + w/2) / W
                cy = (y + h/2) / H
                nw = w / W
                nh = h / H

                # Clamp to valid range
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                f.write(f"{cat_id} {cx:.6f} {cy:.6f} "
                        f"{nw:.6f} {nh:.6f}\n")

if __name__ == '__main__':
    # Convert train
    coco_to_yolo(
        ann_file    = os.path.join(BASE_PATH, 'RUOD_ANN',
                                   'instances_train.json'),
        img_dir     = os.path.join(BASE_PATH, 'RUOD_pic', 'train'),
        out_img_dir = os.path.join(YOLO_PATH, 'images', 'train'),
        out_lbl_dir = os.path.join(YOLO_PATH, 'labels', 'train')
    )

    # Convert test
    coco_to_yolo(
        ann_file    = os.path.join(BASE_PATH, 'RUOD_ANN',
                                   'instances_test.json'),
        img_dir     = os.path.join(BASE_PATH, 'RUOD_pic', 'test'),
        out_img_dir = os.path.join(YOLO_PATH, 'images', 'val'),
        out_lbl_dir = os.path.join(YOLO_PATH, 'labels', 'val')
    )

    # Write dataset yaml
    yaml_content = f"""path: {os.path.abspath(YOLO_PATH)}
train: images/train
val: images/val

nc: 10
names: {list(CATEGORIES.values())}
"""
    with open('data/ruod.yaml', 'w') as f:
        f.write(yaml_content)

    print("Done. Saved: data/ruod.yaml")