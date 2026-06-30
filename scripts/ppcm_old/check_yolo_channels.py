# scripts/check_yolo_channels.py
import sys
sys.path.insert(0, '.')

import torch
from ultralytics import YOLO

YOLO_WEIGHTS = 'runs/detect/work_dirs/yolo_ruod_baseline-3/weights/best.pt'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo = YOLO(YOLO_WEIGHTS)
yolo.model.to(DEVICE)
yolo.model.eval()

outputs = {}

def make_hook(idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            outputs[idx] = output.shape
    return hook

layers = yolo.model.model
for i, layer in enumerate(layers):
    if i in [15, 18, 21]:
        layer.register_forward_hook(make_hook(i))

dummy = torch.randn(1, 3, 640, 640).to(DEVICE)
with torch.no_grad():
    yolo.model(dummy)

print("P3 [15]:", outputs.get(15))
print("P4 [18]:", outputs.get(18))
print("P5 [21]:", outputs.get(21))