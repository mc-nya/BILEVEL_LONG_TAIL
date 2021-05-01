from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Mapping, List

import homura
import hydra
import torch
from homura import trainers, TensorMap, callbacks, optim, lr_scheduler
from homura.vision import DATASET_REGISTRY


from torch import nn, Tensor, utils
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.data import Subset,Dataset


from policy import Policy
from utils import Config, MODEL_REGISTRY

class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)


class EvalTrainer(trainers.TrainerBase):
    def __init__(self, *args, **kwargs):
        super(EvalTrainer, self).__init__(*args, **kwargs)
        if self.policy is not None:
            self.policy.to(self.device)
            self.policy.eval()

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> Mapping[str, Tensor]:
        # input [-1, 1]
        input, target = data
        if self.policy is not None and self.is_train:
            with torch.no_grad():
                # input: [-1, 1]
                input = self.policy(self.policy.denormalize_(input))
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return TensorMap(loss=loss, output=output)
    
    


    

    
    
class EvalTrainer_Ydepends(trainers.TrainerBase):
    def __init__(self,major,minor,*args, **kwargs):
        super(EvalTrainer_Ydepends, self).__init__(*args, **kwargs)
        
        if self.policy[0] is not None:
            self.policy[0].to(self.device)
            self.policy[0].eval()
        
        if self.policy[1] is not None:
            self.policy[1].to(self.device)
            self.policy[1].eval()

        self.major=major
        self.minor=minor
        print(f"major{self.major},minor{self.minor}")
        
    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> Mapping[str, Tensor]:
        # input [-1, 1]
        input, target = data
        #print(target)
        assert input.size()[0]==target.size()[0]
        dv = torch.device('cuda:0')
        #x_input=torch.empty(input.size(),device=dv)
        #major=self.major#set([0])
        #minor=set([6,7,8,9])
        #minor=self.minor#set([1])
        if self.policy is not None and self.is_train:
            with torch.no_grad():
                # input: [-1, 1]
                #print(self.policy[0].denormalize_(input).size())
                #print(self.policy[0](self.policy[0].denormalize_(input)).size())
                
                #print(self.policy[1].denormalize_(input).size())
                #sprint(self.policy[1](self.policy[1].denormalize_(input)).size())
                
                
                result_p1=self.policy[0](self.policy[0].denormalize_(input))
                result_p2=self.policy[1](self.policy[1].denormalize_(input))
                
                for c in list(self.major):
                    if (target==c).nonzero().view(-1).size()[0]>0:
                        print("target major size",(target==c).nonzero().view(-1).size())
                        input[(target==c).nonzero().view(-1)]=result_p1[(target==c).nonzero().view(-1)]
                        
                if None not in self.minor:
                    for c in list(self.minor):
                        if (target==c).nonzero().view(-1).size()[0]>0:
                            input[(target==c).nonzero().view(-1)]=result_p2[(target==c).nonzero().view(-1)]
                            print("target minor size",(target==c).nonzero().view(-1).size())
                
                
                
                """
                print('oi',input)
                
                print('ot',target)
                
                if (target==0).nonzero().view(-1).size()[0]>0:
                    policy1_input=input[(target==0).nonzero().view(-1)]
                    print("aug 1",(target==0).nonzero().view(-1).size())
                    temp=self.policy[0].denormalize_(policy1_input)
                    policy1_aug=self.policy[0](temp)
                    print(policy1_aug.size())
                elif (target==0).nonzero().view(-1).size()[0]==0:
                    policy1_aug=torch.tensor([],device=dv)
                
                if (target==1).nonzero().view(-1).size()[0]>0:
                    policy2_input=input[(target==1).nonzero().view(-1)]
                    print("aug 2",(target==1).nonzero().view(-1).size())
                    if (target==1).nonzero().view(-1).size()[0]==1:
                        policy2_input=torch.reshape(policy2_input,(1,3,32,32))
                        print("changed aug 2",policy2_input,policy2_input.size())
                    temp=self.policy[1].denormalize_(policy2_input)
                    policy2_aug=self.policy[1](temp)
                    print(policy2_aug.size())
                elif (target==1).nonzero().view(-1).size()[0]==0:
                    policy2_aug=torch.tensor([],device=dv)
                    
                    
                
                
                x_input=torch.cat((policy1_aug,policy2_aug),0)
                print(x_input.size())
                """
                """
                for i,x in enumerate(target):
                    print("input",input[i].size())
                    if x.item()==0:
                        #print(input[i])
                        #print("policy",self.policy[0].denormalize_(input[i]).size())
                        temp=self.policy[0].denormalize_(torch.reshape(input[i],(1,3,32,32)))
                        print("temp",temp.size())
                        input_temp = self.policy[0](torch.reshape(temp,(3,32,32)))
                        #input_temp = self.policy[0](temp)
                    else:
                        temp=self.policy[1].denormalize_(torch.reshape(input[i],(1,3,32,32)))
                        input_temp = self.policy[1](torch.reshape(temp,(3,32,32)))
                        #input_temp = self.policy[0](temp)
                    x_input=torch.cat((x_input,input_temp),0)
                """
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return TensorMap(loss=loss, output=output)    
    

@dataclass
class ModelConfig:
    name: str
    num_chunks: int


@dataclass
class DataConfig:
    name: str
    batch_size: int
    download: bool


@dataclass
class CosineSchedulerConfig:
    mul: float
    warmup: int


@dataclass
class StepSchedulerConfig:
    steps: List[int]
    gamma: float


@dataclass
class OptimConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    scheduler: CosineSchedulerConfig or StepSchedulerConfig


@dataclass
class BaseConfig(Config):
    path: str
    first_append: str
    second_append: str
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig


def train_and_eval(cfg: BaseConfig):
    if cfg.path is None:
        print('cfg.path is None, so FasterAutoAugment is not used')
        policy = None
    else:
        path = Path(hydra.utils.get_original_cwd()) / cfg.path
        assert path.exists()
        policy_weight = torch.load(path, map_location='cpu')
        cfg.model.num_chunks=1
        policy1 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy1_kwargs'])
        policy2 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy2_kwargs'])
        policy1.load_state_dict(policy_weight['policy1'])
        policy2.load_state_dict(policy_weight['policy2'])
    
    #torch.manual_seed(0)
    train_loader, test_loader, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                             drop_last=True,
                                                                             download=cfg.data.download,
                                                                             return_num_classes=True,
                                                                             norm=[transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5))
                                                                                   ],
                                                                             num_workers=4
                                                                             )
    
    
    print(f"num_classes{num_classes} in traning set")
    
    model = MODEL_REGISTRY(cfg.model.name)(num_classes)
    optimizer = optim.SGD(cfg.optim.lr,
                          momentum=cfg.optim.momentum,
                          weight_decay=cfg.optim.weight_decay,
                          nesterov=cfg.optim.nesterov)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       cfg.optim.scheduler.mul,
                                                       cfg.optim.scheduler.warmup)
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),
         callbacks.AccuracyCallback(),
         tqdm]
    policy=[policy1,policy2]
    with EvalTrainer_Ydepends(model=model,
                     optimizer=optimizer,
                     loss_f=F.cross_entropy,
                     callbacks=c,
                     scheduler=scheduler,
                     policy=policy,
                     cfg=cfg.model,
                     major=set([i for i in range(10)]),
                     minor=set([None]),        
                     use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
            trainer.test(test_loader)
    print(f"Min. Error Rate: {1 - max(c[1].history['test']):.3f}")
    
    
    
    
def train_and_eval_init(cfg: BaseConfig):
    if cfg.path is None:
        print('cfg.path is None, so FasterAutoAugment is not used')
        policy = None
    else:
        path = Path(hydra.utils.get_original_cwd()) / cfg.path
        assert path.exists()
        #policy_weight = torch.load(path, map_location='cpu')
        #cfg.model.num_chunks=1
        #policy1 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy1_kwargs'])
        #policy2 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy2_kwargs'])
        #policy1.load_state_dict(policy_weight['policy1'])
        #policy2.load_state_dict(policy_weight['policy2'])
    
    #torch.manual_seed(0)
    train_loader, test_loader, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                             drop_last=True,
                                                                             download=cfg.data.download,
                                                                             return_num_classes=True,
                                                                             norm=[transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5))
                                                                                   ],
                                                                             num_workers=4
                                                                             )
    
    
    print(f"num_classes{num_classes} in traning set")
    
    model = MODEL_REGISTRY(cfg.model.name)(num_classes)
    optimizer = optim.SGD(cfg.optim.lr,
                          momentum=cfg.optim.momentum,
                          weight_decay=cfg.optim.weight_decay,
                          nesterov=cfg.optim.nesterov)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       cfg.optim.scheduler.mul,
                                                       cfg.optim.scheduler.warmup)
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),
         callbacks.AccuracyCallback(),
         tqdm]
    #policy=[policy1,policy2]
    with EvalTrainer(model=model,
                     optimizer=optimizer,
                     loss_f=F.cross_entropy,
                     callbacks=c,
                     scheduler=scheduler,
                     policy=None,
                     cfg=cfg.model,  
                     use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
            trainer.test(test_loader)
    print(f"Min. Error Rate: {1 - max(c[1].history['test']):.3f}")
    
    
    
    
    
def load_cifar10(train_size=4000,train_rho=0.01,image_size=None,batch_size=128,num_workers=4,path='./data',num_list=None):
    torch.manual_seed(0) 
    np.random.seed(0)
    train_transform = Compose([
        transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
        #Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        #transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    
    #[transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                          #[transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                           #transforms.RandomHorizontalFlip()]

    test_transform = Compose([
        #Resize(image_size, BICUBIC),    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_dataset = CIFAR10(root=path, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=path, train=False, transform=test_transform, download=True)
    train_x,train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    #test_x, test_y = test_dataset.data, test_dataset.targets
    num_train_samples=[]
    num_val_samples=[]
    train_mu=train_rho**(1./9.)
    #val_mu=val_rho**(1./9.)
    for i in num_list:
        num_train_samples.append(round(train_size*(train_mu**i)))
        #num_val_samples.append(round(val_size*(val_mu**i)))
    train_index=[]
    #val_index=[]
    #print(train_x,train_y)
    print(f"class: {str(num_list)},num_train_samples: {str(num_train_samples)}")
    for i,c in enumerate(num_list):
        train_index.extend(np.where(train_y==c)[0][:num_train_samples[i]])
        #val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])
        #index.extend()
    np.random.shuffle(train_index)
    #random.shuffle(val_index)
    
    train_data,train_targets=train_x[train_index],train_y[train_index]
    #val_data,val_targets=train_x[val_index],train_y[val_index]
    
    train_dataset = CustomDataset(train_data,train_targets,train_transform)
    #val_dataset = CustomDataset(val_data,val_targets,train_transform)
    #train_eval_dataset = CustomDataset(train_data,train_targets,test_transform)
    #val_eval_dataset = CustomDataset(val_data,val_targets,test_transform)
    
    #RS=utils.data.RandomSampler(train_dataset, replacement=True, num_samples=256*128)
    #batch_sampler_x = utils.data.BatchSampler(RS, batch_size, drop_last=True)
    
    train_loader = DataLoader(train_dataset, num_workers=num_workers, 
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False, 
                             pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, 
                            #shuffle=True, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True)

    #eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                #shuffle=False, drop_last=False, pin_memory=True)
    #eval_val_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                #shuffle=False, drop_last=False, pin_memory=True)
    
    #if not val_size:
    return train_loader,test_loader,len(num_list)    
    
    
    
    
def train_and_eval_seperate(cfg: BaseConfig):
    if cfg.path is None:
        print('cfg.path is None, so FasterAutoAugment is not used')
        policy = None
    else:
        path1=cfg.path+cfg.first_append
        path1 = Path(hydra.utils.get_original_cwd()) / path1
        assert path1.exists()
        policy_weight1 = torch.load(path1, map_location='cpu')
        #cfg.model.num_chunks=1
        policy1 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight1['policy_kwargs'])
        #policy2 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy_kwargs'])
        policy1.load_state_dict(policy_weight1['policy'])
        
        
        path2=cfg.path+cfg.second_append
        path2 = Path(hydra.utils.get_original_cwd()) / path2
        assert path2.exists()
        policy_weight2 = torch.load(path2, map_location='cpu')
        #cfg.model.num_chunks=1
        policy2 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight2['policy_kwargs'])
        #policy2 = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy_kwargs'])
        policy2.load_state_dict(policy_weight2['policy'])
        #policy2.load_state_dict(policy_weight['policy2'])
    
    policy=[policy1,policy2]
    train_loader, test_loader, num_classes=load_cifar10(num_list=[0,1,2,3,4,5,6,7,8,9])
    #torch.manual_seed(0)
    #train_loader, test_loader, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                             #drop_last=True,
                                                                             #download=cfg.data.download,
                                                                             #return_num_classes=True,
                                                                             #norm=[transforms.ToTensor(),
                                                                                   #transforms.Normalize(
                                                                                       #(0.5, 0.5, 0.5),
                                                                                       #(0.5, 0.5, 0.5))
                                                                                   #],
                                                                             #num_workers=4
                                                                             #)
    
    
    print(f"num_classes{num_classes} in traning set")
    
    model = MODEL_REGISTRY(cfg.model.name)(num_classes)
    optimizer = optim.SGD(cfg.optim.lr,
                          momentum=cfg.optim.momentum,
                          weight_decay=cfg.optim.weight_decay,
                          nesterov=cfg.optim.nesterov)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       cfg.optim.scheduler.mul,
                                                       cfg.optim.scheduler.warmup)
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),
         callbacks.AccuracyCallback(),
         tqdm]
    #policy=[policy1,policy2]
    with EvalTrainer_Ydepends(model=model,
                     optimizer=optimizer,
                     loss_f=F.cross_entropy,
                     callbacks=c,
                     scheduler=scheduler,
                     policy=policy,
                     cfg=cfg.model,
                     major=set([i for i in range(5)]),
                     minor=set([i for i in range(5,10)]),        
                     use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
            trainer.test(test_loader)
    print(f"Min. Error Rate: {1 - max(c[1].history['test']):.3f}")


@hydra.main('config/train.yaml')
def main(cfg: BaseConfig):
    print(cfg.pretty())
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        return train_and_eval_seperate(cfg)
        #return train_and_eval_init(cfg)


if __name__ == '__main__':
    main()
