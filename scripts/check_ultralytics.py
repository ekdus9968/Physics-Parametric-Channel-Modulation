from ultralytics.nn.modules.head import Detect
import torch
# 256ch input, 4 classes, 3 scales
head = Detect(nc=4, ch=(256, 256, 256))
print(head)
# dummy input
p3 = torch.randn(1, 256, 80, 80)
p4 = torch.randn(1, 256, 40, 40)
p5 = torch.randn(1, 256, 20, 20)
head.eval()
out = head([p3, p4, p5])
print('Output type:', type(out))
print('Output shape:', out[0].shape if isinstance(out, (list,tuple)) else out.shape)