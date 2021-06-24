import torch
import time
import numpy as np
from models import *

# torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True
net = TSN(num_class = 101, num_segments = 16, pretrained_parts = "finetune", modality = "RGB")
#.cuda()
# net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,18,4),use_aux=False).cuda()

net.eval()

x = torch.zeros((16,3,244,244))+1
#.cuda() + 1
for i in range(5):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    print("I'm here!",t2-t1)
t_all = []
for i in range(10):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

