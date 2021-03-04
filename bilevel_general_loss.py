import os
import numpy as np
import random
import torch
import ignite
from torch._C import dtype
from model import EfficientNet
from cifar10 import load_cifar10
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils import print_num_params
from trainer import my_cross_entropy, train_epoch,eval_epoch
import torch.optim as optim
import torch.nn as nn

seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)

model = EfficientNet(num_classes=10, 
                     width_coefficient=1.0, depth_coefficient=1.0, 
                     dropout_rate=0.2)
                    
print_num_params(model)

train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader=load_cifar10(batch_size=64)

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled
torch.backends.cudnn.benchmark = True
device = "cuda"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.001
total_epoch=60
optimizer = optim.SGD(params=model.parameters(),lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

dy=torch.ones([10],dtype=torch.float32,device=device)
ly=torch.zeros([10],dtype=torch.float32,device=device)
#dy.requires_grad=True
ly.requires_grad=True
val_optimizer = optim.SGD(params=[{'params':dy},{'params':ly}],lr=0.0003, momentum=0.9, nesterov=True)

#print(dy.requires_grad,ly)
for i in range(total_epoch):
    
    train_epoch(train_loader,model,my_cross_entropy,optimizer,i,
            val_loader=val_loader,val_optimizer=val_optimizer,val_loss=my_cross_entropy,dy=dy,ly=ly)
    eval_epoch(eval_train_loader,model,my_cross_entropy,i,' train_dataset',dy,ly)
    eval_epoch(test_loader,model,my_cross_entropy,i,' test_dataset',dy,ly)
    print(dy,ly)






