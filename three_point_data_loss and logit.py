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
parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
parser.add_argument('--lr', dest='lr', default=0.01, type=float)
parser.add_argument('--arch_lr', dest='arch_lr', default=0.03, type=float)
parser.add_argument('--log_interval', dest='log_interval', default=50, type=int)
parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=40, type=int)
parser.add_argument('--model_file', dest='model_file', default=None, type=str)
parser.add_argument('--save_path', dest='save_path', default='Three_point', type=str)
parser.add_argument('--epoch', dest='epoch', default=10, type=int)
parser.add_argument('--train_rho', dest='train_rho', default=0.1, type=float)
parser.add_argument('--ARCH_EPOCH', dest='ARCH_EPOCH', default=0, type=int)
parser.add_argument('--ARCH_END', dest='ARCH_END', default=370, type=int)
parser.add_argument('--ARCH_INTERVAL', dest='ARCH_INTERVAL', default=10, type=int)
parser.add_argument('--ARCH_TRAIN_SAMPLE', dest='ARCH_TRAIN_SAMPLE', default=10, type=int)
parser.add_argument('--ARCH_VAL_SAMPLE', dest='ARCH_VAL_SAMPLE', default=10, type=int)
parser.add_argument('--ARCH_EPOCH_INTERVAL', dest='ARCH_EPOCH_INTERVAL', default=1, type=int)
parser.add_argument('--dy', dest='dy', default='True', type=str)
parser.add_argument('--ly', dest='ly', default='True', type=str)
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int)
args=parser.parse_args()

if args.save_path is None:
        import time
        args.save_path=f'./results/{int(time.time())}'       
if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)



args.dy=args.dy=='True'
args.ly=args.ly=='True'

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
ARCH_EPOCH_INTERVAL=args.ARCH_EPOCH_INTERVAL # 

from dataset.three_point import load_three_point as load_dataset
from models.MLP import MLP
num_classes=3

model_ce=MLP(2,3)
model_logit=MLP(2,3)
model_loss=MLP(2,3)
model_logit.load_state_dict(model_ce.state_dict())
model_loss.load_state_dict(model_ce.state_dict())

train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples=load_dataset(batch_size=batch_size,train_rho=train_rho)


print_num_params(model_ce)


criterion = nn.CrossEntropyLoss()

model_ce = model_ce.to(device)
train_optimizer = optim.SGD(params=model_ce.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
train_lr_scheduler=optim.lr_scheduler.MultiStepLR(train_optimizer,milestones=[280,330],gamma=0.1)
for i in range(args.checkpoint+1,total_epoch+1):
        
        text,loss,train_acc=eval_epoch(eval_train_loader,model_ce,cross_entropy,i,' train_dataset',num_classes=num_classes)
        text,loss,val_acc=eval_epoch(val_loader,model_ce,cross_entropy,i,' val_dataset',logit_adjust=None,num_classes=num_classes)
        text,loss,test_acc=eval_epoch(test_loader,model_ce,cross_entropy,i,' test_dataset',logit_adjust=None,num_classes=num_classes)
        train_epoch(i, model_ce, 
                in_loader=train_loader, in_criterion=cross_entropy, 
                in_optimizer=train_optimizer,
                is_out=False, 
                num_classes=num_classes,
                ARCH_EPOCH=ARCH_EPOCH,ARCH_INTERVAL=ARCH_INTERVAL,
                ARCH_TRAIN_SAMPLE=ARCH_TRAIN_SAMPLE,ARCH_VAL_SAMPLE=ARCH_VAL_SAMPLE)
        train_lr_scheduler.step()

dy_logit=torch.ones([num_classes],dtype=torch.float32,device=device)
ly_logit=torch.zeros([num_classes],dtype=torch.float32,device=device)
dy_logit.requires_grad=True
ly_logit.requires_grad=True

model_logit = model_logit.to(device)
train_optimizer = optim.SGD(params=model_logit.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
val_optimizer = optim.SGD(params=[{'params':dy_logit},{'params':ly_logit}],
                        lr=arch_lr,momentum=0.9,weight_decay=1e-4)
train_lr_scheduler=optim.lr_scheduler.MultiStepLR(train_optimizer,milestones=[280,330],gamma=0.1)
val_lr_scheduler=optim.lr_scheduler.MultiStepLR(val_optimizer,milestones=[280,330],gamma=0.2)
for i in range(args.checkpoint+1,total_epoch+1):
        
        text,loss,train_acc=eval_epoch(eval_train_loader,model_logit,cross_entropy,i,' train_dataset',num_classes=num_classes,logit_adjust=logit_adjust_ly,params=[dy_logit,ly_logit],)
        text,loss,val_acc=eval_epoch(val_loader,model_logit,cross_entropy,i,' val_dataset',params=[dy_logit,ly_logit],logit_adjust=logit_adjust_ly,num_classes=num_classes)
        text,loss,test_acc=eval_epoch(test_loader,model_logit,cross_entropy,i,' test_dataset',params=[dy_logit,ly_logit],logit_adjust=logit_adjust_ly,num_classes=num_classes)
        print(dy_logit,ly_logit,'\n')
        train_epoch(i, model_logit, 
                in_loader=train_loader, in_criterion=cross_entropy, 
                in_optimizer=train_optimizer,in_params=[dy_logit,ly_logit],
                is_out=(i>=ARCH_EPOCH) and (i<=ARCH_END) and ((i+1)%ARCH_EPOCH_INTERVAL)==0, 
                out_loader=val_loader, out_optimizer=val_optimizer,
                out_criterion=cross_entropy, out_logit_adjust=logit_adjust_ly, out_params=[dy_logit,ly_logit],
                out_posthoc=True,
                num_classes=num_classes,
                ARCH_EPOCH=ARCH_EPOCH,ARCH_INTERVAL=ARCH_INTERVAL,
                ARCH_TRAIN_SAMPLE=ARCH_TRAIN_SAMPLE,ARCH_VAL_SAMPLE=ARCH_VAL_SAMPLE)
        train_lr_scheduler.step()
        val_lr_scheduler.step()


dy_loss=torch.ones([num_classes],dtype=torch.float32,device=device)
ly_loss=torch.zeros([num_classes],dtype=torch.float32,device=device)
dy_loss.requires_grad=True
ly_loss.requires_grad=True

model_loss = model_loss.to(device)
train_optimizer = optim.SGD(params=model_loss.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
val_optimizer = optim.SGD(params=[{'params':dy_loss},{'params':ly_loss}],
                        lr=arch_lr,momentum=0.9,weight_decay=1e-4)
train_lr_scheduler=optim.lr_scheduler.MultiStepLR(train_optimizer,milestones=[280,330],gamma=0.1)
val_lr_scheduler=optim.lr_scheduler.MultiStepLR(val_optimizer,milestones=[280,330],gamma=0.2)
for i in range(args.checkpoint+1,total_epoch+1):
        
        text,loss,train_acc=eval_epoch(eval_train_loader,model_loss,loss_adjust_cross_entropy,i,' train_dataset',params=[dy_loss,ly_loss],num_classes=num_classes)
        text,loss,val_acc=eval_epoch(val_loader,model_loss,cross_entropy,i,' val_dataset',params=[dy_loss,ly_loss],logit_adjust=None,num_classes=num_classes)
        text,loss,test_acc=eval_epoch(test_loader,model_loss,cross_entropy,i,' test_dataset',params=[dy_loss,ly_loss],logit_adjust=None,num_classes=num_classes)
        print(dy_loss,ly_loss,'\n')
        train_epoch(i, model_loss, 
                in_loader=train_loader, in_criterion=loss_adjust_cross_entropy, 
                in_optimizer=train_optimizer,in_params=[dy_loss,ly_loss],
                is_out=(i>=ARCH_EPOCH) and (i<=ARCH_END) and ((i+1)%ARCH_EPOCH_INTERVAL)==0, 
                out_loader=val_loader, out_optimizer=val_optimizer,
                out_criterion=cross_entropy, out_logit_adjust=None, out_params=[dy_loss,ly_loss],
                num_classes=num_classes,
                ARCH_EPOCH=ARCH_EPOCH,ARCH_INTERVAL=ARCH_INTERVAL,
                ARCH_TRAIN_SAMPLE=ARCH_TRAIN_SAMPLE,ARCH_VAL_SAMPLE=ARCH_VAL_SAMPLE)
        train_lr_scheduler.step()
        val_lr_scheduler.step()

x_test=[]
for x in np.arange(-3, 3, 0.01):
        for y in np.arange(-3, 3, 0.01):
                x_test.append((x,y))
x_test=torch.from_numpy(np.array(x_test).astype(np.float32)).to(device)

preds=[]
logits_ce=model_ce(x_test)
preds_ce = logits_ce.data.max(1)[1].cpu().numpy()
preds.append(preds_ce)

logits_logit=model_logit(x_test)*dy_logit-ly_logit
preds_logit = logits_logit.data.max(1)[1].cpu().numpy()
preds.append(preds_logit)

logits_loss=model_loss(x_test)
preds_loss = logits_loss.data.max(1)[1].cpu().numpy()
preds.append(preds_loss)
print(preds_ce,preds_logit,preds_loss)

import matplotlib.pyplot as plt
x_test=x_test.cpu().numpy()
colors=['tab:blue','tab:red','tab:green']
fig, ax = plt.subplots()
for i in range(len(preds)):
        plt.cla()
        for label in range(num_classes):
                index=np.where(preds[i]==label)[0]
                print(index)
                ax.scatter(x_test[index,0],x_test[index,1],  c=colors[label],alpha=0.5)

        ax.grid(True)
        plt.savefig(f'{args.save_path}/fig_{i}.png')



