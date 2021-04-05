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
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--model', dest='model', default='ResNet20', type=str)
parser.add_argument('--dataset', dest='dataset', default='Cifar10', type=str)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--lr', dest='lr', default=0.01, type=float)
parser.add_argument('--arch_lr', dest='arch_lr', default=0.003, type=float)
parser.add_argument('--log_interval', dest='log_interval', default=50, type=int)
parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=30, type=int)
parser.add_argument('--model_file', dest='model_file', default=None, type=str)
parser.add_argument('--save_path', dest='save_path', default=None, type=str)
parser.add_argument('--epoch', dest='epoch', default=200, type=int)
parser.add_argument('--train_rho', dest='train_rho', default=0.1, type=float)
parser.add_argument('--ARCH_EPOCH', dest='ARCH_EPOCH', default=10, type=int)
parser.add_argument('--ARCH_END', dest='ARCH_END', default=200, type=int)
parser.add_argument('--ARCH_INTERVAL', dest='ARCH_INTERVAL', default=40, type=int)
parser.add_argument('--ARCH_TRAIN_SAMPLE', dest='ARCH_TRAIN_SAMPLE', default=20, type=int)
parser.add_argument('--ARCH_VAL_SAMPLE', dest='ARCH_VAL_SAMPLE', default=20, type=int)
parser.add_argument('--ARCH_EPOCH_INTERVAL', dest='ARCH_EPOCH_INTERVAL', default=1, type=int)
parser.add_argument('--dy', dest='dy', default='True', type=str)
parser.add_argument('--ly', dest='ly', default='True', type=str)
args=parser.parse_args()

args.dy=args.dy=='True'
args.ly=args.ly=='True'
network_model=args.model # Model, either ResNet20 or Efficient
dataset=args.dataset  # Dataset, either Cifar10 or Cifar100
batch_size=args.batch_size # Training batchsize
lr = args.lr  # inner optim lr
arch_lr=args.arch_lr  # outer optim lr
total_epoch=args.epoch # Total training epoch
train_rho=args.train_rho # Imbalance ratio : Min/Max

ARCH_EPOCH=args.ARCH_EPOCH # The epoch for starting outer opimization
ARCH_END=args.ARCH_END # The epoch for ending outer opimization
ARCH_INTERVAL=args.ARCH_INTERVAL # The iteration interval for conduction hyper-parameter update
ARCH_TRAIN_SAMPLE=args.ARCH_TRAIN_SAMPLE # The batches of training samples used for one arch update
ARCH_VAL_SAMPLE=args.ARCH_VAL_SAMPLE # The batches of validation samples used for one arch update
ARCH_EPOCH_INTERVAL=args.ARCH_EPOCH_INTERVAL
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

dy=torch.ones([num_classes],dtype=torch.float32,device=device)#+torch.rand([num_classes],dtype=torch.float32,device=device)*arch_lr
ly=torch.zeros([num_classes],dtype=torch.float32,device=device)
#ly=torch.rand([num_classes],dtype=torch.float32,device=device)*arch_lr
wy=torch.ones([num_classes],dtype=torch.float32,device=device)
dy.requires_grad=args.dy
ly.requires_grad=args.ly
#wy.requires_grad=True

train_optimizer = optim.Adam(params=model.parameters(),lr=lr)
val_optimizer = optim.Adam(params=[{'params':dy},{'params':ly},{'params':wy}],lr=arch_lr)
train_lr_scheduler=optim.lr_scheduler.MultiStepLR(train_optimizer,milestones=[100,120,150],gamma=0.1)
val_lr_scheduler=optim.lr_scheduler.MultiStepLR(val_optimizer,milestones=[60,80,120,150],gamma=0.1)
if args.save_path is None:
        import time
        args.save_path=f'./results/{int(time.time())}'       
if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
logfile=open(f'{args.save_path}/logs.txt',mode='w')
dy_log=open(f'{args.save_path}/dy.txt',mode='w')
ly_log=open(f'{args.save_path}/ly.txt',mode='w')
acc_log=open(f'{args.save_path}/acc.txt',mode='w')
config_log=open(f'{args.save_path}/config.txt',mode='w')
for k,v in vars(args).items():
	config_log.write(str(k)+' '+str(v)+'\n')
config_log.close()
#print(dy.requires_grad,ly)
torch.save(model,f'{args.save_path}/init_model.pth')
for i in range(total_epoch+1):
        
        text,loss,train_acc=eval_epoch(eval_train_loader,model,loss_adjust_cross_entropy,i,' train_dataset',params=[dy,ly,wy],num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,val_acc=eval_epoch(val_loader,model,cross_entropy,i,' val_dataset',params=[dy,ly,wy],logit_adjust=None,num_classes=num_classes)
        logfile.write(text+'\n')
        text,loss,test_acc=eval_epoch(test_loader,model,cross_entropy,i,' test_dataset',params=[dy,ly,wy],logit_adjust=None,num_classes=num_classes)
        logfile.write(text+'\n')
        print(dy,ly,'\n')
        train_epoch(i, model, 
                in_loader=train_loader, in_criterion=loss_adjust_cross_entropy, 
                in_optimizer=train_optimizer,in_params=[dy,ly],
                is_out=(i>=ARCH_EPOCH) and (i<=ARCH_END) and ((i+1)%ARCH_EPOCH_INTERVAL)==0, 
                out_loader=val_loader, out_optimizer=val_optimizer,
                out_criterion=cross_entropy, out_logit_adjust=None, out_params=[dy,ly],
                num_classes=num_classes,
                ARCH_EPOCH=ARCH_EPOCH,ARCH_INTERVAL=ARCH_INTERVAL,
                ARCH_TRAIN_SAMPLE=ARCH_TRAIN_SAMPLE,ARCH_VAL_SAMPLE=ARCH_VAL_SAMPLE)
        logfile.write(str(dy)+str(ly)+'\n\n')
        dy_log.write(f'{dy.detach().cpu().numpy()}\n')
        ly_log.write(f'{ly.detach().cpu().numpy()}\n')
        acc_log.write(f'{train_acc} {val_acc} {test_acc}\n')
        logfile.flush()
        dy_log.flush()
        ly_log.flush()
        acc_log.flush()
        train_lr_scheduler.step()
        val_lr_scheduler.step()
        if i%args.checkpoint_interval==0:
                torch.save(model,f'{args.save_path}/epoch_{i}.pth')
logfile.close()
dy_log.close()
ly_log.close()
acc_log.close()
torch.save(model,f'{args.save_path}/loss_adjustment.pth')







