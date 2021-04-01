from models.utils import load_pretrained_weights
import os
import numpy as np
import random
from numpy.lib.scimath import log
import torch
import ignite
from torch._C import dtype

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.metrics import print_num_params
from core.trainer import loss_adjust_cross_entropy,cross_entropy, logit_adjust_ly, train_epoch,eval_epoch
import torch.optim as optim
import torch.nn as nn

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled
torch.backends.cudnn.benchmark = True
device = "cuda"

# seed = 17 # random seed
# random.seed(seed)
# _ = torch.manual_seed(seed)


network_model='ResNet20' # Model, either ResNet20 or Efficient
dataset='Cifar10'  # Dataset, either Cifar10 or Cifar100
lr = 0.01  # inner optim lr
hp_lr=0.0003  # outer optim lr
total_epoch=200 # Total training epoch
batch_size=64 # Training batchsize
train_rho=0.1 # Imbalance ratio : Min/Max
ARCH_EPOCH=0 # The epoch for starting outer opimization

if dataset=='Cifar10':
        from dataset.cifar10 import load_cifar10 as load_dataset
        num_classes=10
elif dataset=='Cifar100':
        from dataset.cifar100 import load_cifar100 as  load_dataset
        num_classes=100

if network_model=='Efficient':
        from models.EfficientNet_NEW import EfficientNet
        model=EfficientNet.from_pretrained('efficientnet-b0',load_weights=False,num_classes=num_classes)
        train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples=load_dataset(batch_size=batch_size,train_rho=train_rho)
elif network_model=='ResNet20':
        from models.ResNet import ResNet20
        model=ResNet20()
        train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples=load_dataset(batch_size=batch_size,train_rho=train_rho,image_size=32)
print_num_params(model)



model = model.to(device)

criterion = nn.CrossEntropyLoss()


# pi=num_train_samples/np.sum(num_train_samples)
# tau=2.5
# print(pi)
# pi=tau*log(pi)
# print(pi)
#optimizer = optim.SGD(params=model.parameters(),lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
#ly=torch.tensor(pi).cuda()
#val_optimizer = optim.SGD(params=[{'params':dy},{'params':ly}],lr=hp_lr, momentum=0.9, nesterov=True)

dy=torch.ones([num_classes],dtype=torch.float32,device=device)
ly=torch.zeros([num_classes],dtype=torch.float32,device=device)
wy=torch.ones([num_classes],dtype=torch.float32,device=device)
dy.requires_grad=True
ly.requires_grad=True
wy.requires_grad=True

train_optimizer = optim.Adam(params=model.parameters(),lr=lr)
val_optimizer = optim.Adam(params=[{'params':dy},{'params':ly},{'params':wy}],lr=hp_lr)
train_lr_scheduler=optim.lr_scheduler.MultiStepLR(train_optimizer,milestones=[80,120,150,180],gamma=0.1)


if not os.path.exists('./results/loss_adjustment/'):
        os.makedirs('./results/loss_adjustment/')
logfile=open('./results/loss_adjustment/logs.txt',mode='w')
#print(dy.requires_grad,ly)
torch.save(model,'./results/loss_adjustment/init_model.pth')
for i in range(total_epoch+1):
        train_epoch(i, model, 
                in_loader=train_loader, in_criterion=loss_adjust_cross_entropy, in_optimizer=train_optimizer,in_params=[dy,ly],
                is_out=True, out_loader=val_loader, out_optimizer=val_optimizer,
                out_criterion=cross_entropy, out_logit_adjust=None, out_params=[dy,ly],
                ARCH_EPOCH=0,num_classes=num_classes)
        text,loss,acc=eval_epoch(eval_train_loader,model,cross_entropy,i,' train_dataset',params=[dy,ly,wy],num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,acc=eval_epoch(val_loader,model,cross_entropy,i,' val_dataset',params=[dy,ly,wy],logit_adjust=None,num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,acc=eval_epoch(test_loader,model,cross_entropy,i,' test_dataset',params=[dy,ly,wy],logit_adjust=None,num_classes=num_classes)
        logfile.write(text+'\n')
        print(dy,ly,'\n')
        logfile.write(str(dy)+str(ly)+'\n\n')
        logfile.flush()
        train_lr_scheduler.step()

        
torch.save(model,'./results/loss_adjustment/final_model.pth')







