import pathlib
from dataclasses import dataclass
from typing import Tuple, Mapping, Any
from torchvision.transforms import Compose
from torch.utils.data import Subset,Dataset
import homura
import hydra
import torch
from homura import trainers, TensorMap, optim, callbacks
from homura.vision import DATASET_REGISTRY
from torch import nn, Tensor, utils
from torch.nn import functional as F
from torchvision import transforms

from policy import Policy
from utils import Config, MODEL_REGISTRY
import numpy as np
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torch.utils.data import DataLoader

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

class Discriminator(nn.Module):
    def __init__(self,
                 base_module: nn.Module
                 ):
        super(Discriminator, self).__init__()
        self.base_model = base_module
        num_features = self.base_model.fc.in_features
        num_class = self.base_model.fc.out_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_class)
        self.discriminator = nn.Sequential(nn.Linear(num_features, num_features),
                                           nn.ReLU(),
                                           nn.Linear(num_features, 1))

    def forward(self,
                input: Tensor
                ) -> Tuple[Tensor, Tensor]:
        x = self.base_model(input)
        return self.classifier(x), self.discriminator(x).view(-1)


class AdvTrainer(trainers.TrainerBase):
    # acknowledge https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> Mapping[str, Tensor]:
        # input: [-1, 1]
        input, target = data
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]
        loss, d_loss, a_loss = self.wgan_loss(n_input, n_target, a_input, a_target)

        return TensorMap(loss=loss, d_loss=d_loss, a_loss=a_loss)

    def wgan_loss(self,
                  n_input: Tensor,
                  n_target: Tensor,
                  a_input: Tensor,
                  a_target: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.model['main'].requires_grad_(True)
        self.model['main'].zero_grad()
        # real images
        output, n_output = self.model['main'](n_input)
        loss = self.cfg.cls_factor * F.cross_entropy(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)

        # augmented images
        with torch.no_grad():
            # a_input [-1, 1] -> [0, 1]
            
            a_input = self.model['policy'].denormalize_(a_input)
            augmented = self.model['policy'](a_input)
            #print(a_input.size())
        _, a_output = self.model['main'](augmented)
        #print(a_output.size())
        d_a_loss = a_output.mean()
        #print(d_a_loss.size())
        d_a_loss.backward(ones)
        gp = self.cfg.gp_factor * self.gradient_penalty(n_input, augmented)
        #print(gp.size())
        gp.backward()
        self.optimizer['main'].step()
        #exit(0)
        # train policy
        self.model['main'].requires_grad_(False)
        self.model['policy'].zero_grad()
        _output, a_output = self.model['main'](self.model['policy'](a_input))
        _loss = self.cfg.cls_factor * F.cross_entropy(_output, a_target)
        _loss.backward(retain_graph=True)
        a_loss = a_output.mean()
        a_loss.backward(-ones)
        self.optimizer['policy'].step()

        return loss + _loss, -d_n_loss + d_a_loss + gp, -a_loss

    def gradient_penalty(self,
                         real: Tensor,
                         fake: Tensor
                         ) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.model['main'](interpolated)
        grad = torch.autograd.grad(outputs=output, inputs=interpolated, grad_outputs=torch.ones_like(output),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def state_dict(self
                   ) -> Mapping[str, Any]:
        policy: Policy = self.accessible_model['policy']
        return {'policy': policy.state_dict(),
                'policy_kwargs': dict(num_sub_policies=policy.num_sub_policies,
                                      temperature=policy.temperature,
                                      operation_count=policy.operation_count),
                'epoch': self.epoch,
                'step': self.step}

    def save(self,
             path: str
             ) -> None:
        if homura.is_master():
            path = pathlib.Path(path)
            path.mkdir(exist_ok=True, parents=True)
            with (path / f'{self.epoch}.pt').open('wb') as f:
                torch.save(self.state_dict(), f)

                
                
                
class AdvTrainer_YCustom(trainers.TrainerBase):
    # acknowledge https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def __init__(self,major,minor, *args, **kwargs):
        super(AdvTrainer_YCustom, self).__init__(*args, **kwargs)
        self.major=major
        self.minor=minor
        
        print(f"major{self.major},minor{self.minor}")
    
    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> Mapping[str, Tensor]:
        # input: [-1, 1]
        input, target = data
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]
        loss, d_loss, a_loss = self.wgan_loss(n_input, n_target, a_input, a_target)
        

        return TensorMap(loss=loss, d_loss=d_loss, a_loss=a_loss)

    def wgan_loss(self,
                  n_input: Tensor,
                  n_target: Tensor,
                  a_input: Tensor,
                  a_target: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.model['main'].requires_grad_(True)
        self.model['main'].zero_grad()
        # real images
        output, n_output = self.model['main'](n_input)
        loss = self.cfg.cls_factor * F.cross_entropy(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)
        dv = torch.device('cuda:0')
        # augmented images
        #major=set([0,1,2,3,4,5])
        #major=set([0])
        #minor=set([6,7,8,9])
        #minor=set([1])
        with torch.no_grad():
            # a_input [-1, 1] -> [0, 1]
            augmented=torch.tensor([],device=dv)
            a_output=torch.tensor([],device=dv)
            a_input1=torch.tensor([],device=dv)
            a_input2=torch.tensor([],device=dv)
            #_output1=torch.tensor([],device=dv)
            #_output2=torch.tensor([],device=dv)
            temp=torch.tensor([],device=dv)
            ctr1,ctr2=0,0
            for i,x in enumerate(a_target):
                if x.item() in self.major:
                    ctr1+=1
                    a_input1_sub = self.model['policy1'].denormalize_(torch.reshape(a_input[i],(1,3,32,32)))
                    augmented1 = self.model['policy1'](torch.reshape(a_input1_sub,(1,3,32,32)))
                    temp1, a_output1 = self.model['main'](augmented1)
                    a_input1=torch.cat((a_input1,a_input1_sub),0)
                    augmented=torch.cat((augmented,augmented1),0)
                    a_output=torch.cat((a_output,a_output1),0)
                    temp=torch.cat((temp,temp1),0)
                else:
                    ctr2+=1
                    a_input2_sub = self.model['policy2'].denormalize_(torch.reshape(a_input[i],(1,3,32,32)))
                    augmented2 = self.model['policy2'](torch.reshape(a_input2_sub,(1,3,32,32)))
                    temp2, a_output2 = self.model['main'](augmented2)
                    a_input2=torch.cat((a_input2,a_input2_sub),0)
                    augmented=torch.cat((augmented,augmented2),0)
                    a_output=torch.cat((a_output,a_output2),0)
                    temp=torch.cat((temp,temp2),0)
            print(f"major{ctr1},minor{ctr2}")            
        #augmented=torch.cat((augmented1,augmented2),0)
        #a_output=torch.cat((a_output1,a_output2),0)
        from torch.autograd import Variable
        #print(a_output.size())
        a_output=Variable(a_output, requires_grad=True)
        #a_output=torch.reshape(a_output,(a_output.size()[0],1))
        #print(a_output.size())
        d_a_loss = a_output.mean()
        #print(d_a_loss)
        d_a_loss.backward(ones)
        #print(d_a_loss.size())
        gp = self.cfg.gp_factor * self.gradient_penalty(n_input, augmented)
        #print(gp)
        gp.backward()
        self.optimizer['main'].step()

        # train policy
        self.model['main'].requires_grad_(False)
        
        _output=torch.zeros_like(temp,device=dv)
        if len(a_input1)>0:
            self.model['policy1'].zero_grad()
            _output1, a_output1 = self.model['main'](self.model['policy1'](a_input1))
            index=[]
            for c in list(self.major):
                index=index+(a_target==c).nonzero().view(-1).tolist()
            index.sort()
            _output[index]=_output1
        if len(a_input2)>0:
            self.model['policy2'].zero_grad()
            _output2, a_output2 = self.model['main'](self.model['policy2'](a_input2))
            index=[]
            for c in list(self.minor):
                index=index+(a_target==c).nonzero().view(-1).tolist()
            index.sort()
            _output[index]=_output2
        
       # assert (target==0).nonzero().view(-1).size()[0]==len(a_input1) and (target==1).nonzero().view(-1).size()[0]==len(a_input2)
        
        
        #_output[(target==0).nonzero().view(-1)]=_output1
        #print("class 0 num",(target==0).nonzero().view(-1).size())
        #_output[(target==1).nonzero().view(-1)]=_output1
        #print("class 1 num",(target==1).nonzero().view(-1).size())
        #,_output2),0)
        
        
        _loss = self.cfg.cls_factor * F.cross_entropy(_output, a_target)
        _loss.backward(retain_graph=True)
        a_loss = a_output.mean()
        a_loss.backward(-ones)
        self.optimizer['policy1'].step()
        self.optimizer['policy2'].step()

        return loss + _loss, -d_n_loss + d_a_loss + gp, -a_loss

    def gradient_penalty(self,
                         real: Tensor,
                         fake: Tensor
                         ) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.model['main'](interpolated)
        grad = torch.autograd.grad(outputs=output, inputs=interpolated, grad_outputs=torch.ones_like(output),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def state_dict(self
                   ) -> Mapping[str, Any]:
        policy1: Policy = self.accessible_model['policy1']
        policy2: Policy = self.accessible_model['policy2']
        return {'policy1': policy1.state_dict(),
                'policy2': policy2.state_dict(),
                'policy1_kwargs': dict(num_sub_policies=policy1.num_sub_policies,
                                      temperature=policy1.temperature,
                                      operation_count=policy1.operation_count),
                'policy2_kwargs': dict(num_sub_policies=policy2.num_sub_policies,
                                      temperature=policy2.temperature,
                                      operation_count=policy2.operation_count),
                'epoch': self.epoch,
                'step': self.step}

    def save(self,
             path: str
             ) -> None:
        if homura.is_master():
            path = pathlib.Path(path)
            path.mkdir(exist_ok=True, parents=True)
            with (path / f'{self.epoch}.pt').open('wb') as f:
                torch.save(self.state_dict(), f)
                
                
                
                
                

@dataclass
class ModelConfig:
    cls_factor: float
    gp_factor: float
    temperature: float
    num_sub_policies: int
    num_chunks: int
    operation_count: int


@dataclass
class DataConfig:
    name: str
    train_size: int
    batch_size: int

    cutout: bool
    download: bool


@dataclass
class OptimConfig:
    epochs: int

    main_lr: float
    policy_lr: float


@dataclass
class BaseConfig(Config):
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig


def search(cfg: BaseConfig
           ):
    train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                   train_size=cfg.data.train_size,
                                                                   drop_last=True,
                                                                   download=cfg.data.download,
                                                                   return_num_classes=True,
                                                                   num_workers=4)
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    #data_obj={'train_loader':train_loader,'num_classes':num_classes}
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),  # classification loss
         callbacks.metric_callback_by_name('d_loss'),  # discriminator loss
         callbacks.metric_callback_by_name('a_loss'),  # augmentation loss
         tqdm]
    with AdvTrainer(model,
                    optimizer,
                    F.cross_entropy,
                    callbacks=c,
                    cfg=cfg.model,
                    use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
        trainer.save(pathlib.Path(hydra.utils.get_original_cwd()) / 'policy_weights' / cfg.data.name)
        
      

    
    
    
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
    
    RS=utils.data.RandomSampler(train_dataset, replacement=True, num_samples=256*128)
    batch_sampler_x = utils.data.BatchSampler(RS, batch_size, drop_last=True)
    
    train_loader = DataLoader(train_dataset, num_workers=num_workers, 
                            batch_sampler=batch_sampler_x,
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
    #else:
    #return train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples
    
    
    
    
    
    
    
def data_maker():
    major_train,_,major_class_num=load_cifar10(num_list=[0,1,2,3,4])
    minor_train,_,minor_class_num=load_cifar10(num_list=[5,6,7,8,9])
    
    
    

        
        
def search1(cfg: BaseConfig
           ):
    #train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                  # train_size=cfg.data.train_size,
                                                                   #drop_last=True,
                                                                   #download=cfg.data.download,
                                                                   #return_num_classes=True,
                                                                   #num_workers=4)
                        
    train_loader, _, num_classes=load_cifar10(num_list=[0,1,2,3,4])
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),  # classification loss
         callbacks.metric_callback_by_name('d_loss'),  # discriminator loss
         callbacks.metric_callback_by_name('a_loss'),  # augmentation loss
         tqdm]
    with AdvTrainer(model,
                    optimizer,
                    F.cross_entropy,
                    callbacks=c,
                    cfg=cfg.model,
                    use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
        trainer.save(pathlib.Path(hydra.utils.get_original_cwd()) / 'policy_weights' / cfg.data.name /'policy1')
        
        
        
def search2(cfg: BaseConfig
           ):
    #train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                   #train_size=cfg.data.train_size,
                                                                   #drop_last=True,
                                                                   #download=cfg.data.download,
                                                                   #return_num_classes=True,
                                                                   #num_workers=4)
    train_loader, _, num_classes=load_cifar10(num_list=[5,6,7,8,9])
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),  # classification loss
         callbacks.metric_callback_by_name('d_loss'),  # discriminator loss
         callbacks.metric_callback_by_name('a_loss'),  # augmentation loss
         tqdm]
    with AdvTrainer(model,
                    optimizer,
                    F.cross_entropy,
                    callbacks=c,
                    cfg=cfg.model,
                    use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
        trainer.save(pathlib.Path(hydra.utils.get_original_cwd()) / 'policy_weights' / cfg.data.name /'policy2')
        
        

        
        
def search_Y(cfg: BaseConfig
           ):
    #train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                  #train_size=cfg.data.train_size,
                                                                   #drop_last=True,
                                                                   #download=cfg.data.download,
                                                                   #return_num_classes=True,
                                                                   #num_workers=4)
    #torch.manual_seed(0)                    
    train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                    train_size=cfg.data.train_size,
                                                                   drop_last=True,
                                                                   download=cfg.data.download,
                                                                   return_num_classes=True,
                                                                   num_workers=4)
    
    #num_classes=2
    
    print(f"num_classes{num_classes} in search Y dependent")
    
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy1': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks),
             'policy2': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
    """
    
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
        
    """    
    
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy1': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999)),
                 'policy2': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    
    
    """
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
                 #'policy2': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    """
    data_objs={'train_loader':train_loader,'num_classes':num_classes}
    tqdm = callbacks.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.LossCallback(),  # classification loss
         callbacks.metric_callback_by_name('d_loss'),  # discriminator loss
         callbacks.metric_callback_by_name('a_loss'),  # augmentation loss
         tqdm]
    with AdvTrainer_YCustom(model=model,
                    optimizer=optimizer,
                    loss_f=F.cross_entropy,
                    callbacks=c,
                    cfg=cfg.model,
                    major=set([i for i in range(10)]),
                    minor=set([None]), 
                    use_cuda_nonblocking=True) as trainer:
        for _ in tqdm:
            trainer.train(train_loader)
        trainer.save(pathlib.Path(hydra.utils.get_original_cwd()) / 'policy_weights' / cfg.data.name)
        torch.save(data_objs, '/home/csgrad/ychan/dda/faster_autoaugment/dataloader.pth')
        
        
        

@hydra.main('config/search.yaml')
def main(cfg: BaseConfig):
    print(cfg.pretty())
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        search1(cfg)
        search2(cfg)
        #return search_Y(cfg)


if __name__ == '__main__':
    main()
