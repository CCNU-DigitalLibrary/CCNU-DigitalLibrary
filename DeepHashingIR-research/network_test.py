import torch
from utils.tools import *
from network import *
from torchsummary import summary

import os
import torch.optim as optim
import time
import numpy as np


# hi = torch.rand(1, 2)
# hj = torch.rand(1, 2)
# print(hi)
# print(hj)
#
# inner_product = hi @ hj.t()
# print(inner_product)
# norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
# print(norm)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res = ADHN(48).to(device)
# torchsummary.summary(model, input_size, batch_size=-1, device="cuda") 其中input size 格式为 Channel Height Weight  batch_size默认为-1
summary(res, (3, 224, 224))
