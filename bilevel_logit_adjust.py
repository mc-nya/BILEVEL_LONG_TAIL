import os
import numpy as np
import random
import torch
import ignite
from torch._C import dtype
from models.EfficientNet import EfficientNet
from dataset.cifar10 import load_cifar10 as load_dataset
#from dataset.cifar100 import load_cifar100 as load_dataset
num_classes=10
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.metrics import print_num_params
from core.trainer import loss_adjust_cross_entropy,cross_entropy, logit_adjust_ly, train_epoch,eval_epoch
import torch.optim as optim
import torch.nn as nn

seed = 17
random.seed(seed)
_ = torch.manual_seed(seed)

model = EfficientNet(num_classes=num_classes, 
                     width_coefficient=1.0, depth_coefficient=1.0, 
                     dropout_rate=0.2)
                    
print_num_params(model)

train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples=load_dataset(batch_size=64,train_rho=1.)

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled
torch.backends.cudnn.benchmark = True
device = "cuda"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.01
hp_lr=0.003
total_epoch=60
#optimizer = optim.SGD(params=model.parameters(),lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
optimizer = optim.Adam(params=model.parameters(),lr=lr)
dy=torch.ones([num_classes],dtype=torch.float32,device=device)
ly=torch.zeros([num_classes],dtype=torch.float32,device=device)
#dy.requires_grad=True
#ly.requires_grad=True
#val_optimizer = optim.SGD(params=[{'params':dy},{'params':ly}],lr=0.0003, momentum=0.9, nesterov=True)
val_optimizer = optim.Adam(params=[{'params':dy},{'params':ly}],lr=hp_lr)

if not os.path.exists('./results/logit_adjustment/'):
        os.makedirs('./results/logit_adjustment/')
logfile=open('./results/logit_adjustment/logs.txt',mode='w')
#print(dy.requires_grad,ly)
torch.save(model,'./results/logit_adjustment/init_model.pth')
for i in range(total_epoch+1):
        text,loss,acc=eval_epoch(eval_train_loader,model,cross_entropy,i,' train_dataset',params=[dy,ly],num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,acc=eval_epoch(val_loader,model,cross_entropy,i,' val_dataset',params=[dy,ly],logit_adjust=logit_adjust_ly,num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,acc=eval_epoch(test_loader,model,cross_entropy,i,' test_dataset',params=[dy,ly],logit_adjust=logit_adjust_ly,num_classes=num_classes)
        logfile.write(text+'\n')
        print(dy,ly,'\n')
        logfile.write(str(dy)+str(ly)+'\n\n')
        logfile.flush()

        train_epoch(i, model, train_loader, cross_entropy, optimizer,
                is_out=False, out_loader=val_loader, out_optimizer=val_optimizer,
                out_criterion=cross_entropy, out_logit_adjust=logit_adjust_ly, out_params=[dy,ly],
                ARCH_EPOCH=5,num_classes=num_classes)
torch.save(model,'./results/logit_adjustment/final_model.pth')







