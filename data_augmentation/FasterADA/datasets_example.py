import copy
import inspect
import pathlib
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.jit.annotations import List
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torch.utils.data import Subset,Dataset

from homura import is_distributed, Registry, get_environ


# to enable _split_dataset
def _svhn_getitem(self,
                  index: int):
    img, target = self.data[index], int(self.targets[index])
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.transform is not None:
        img = self.transform(img)
    return img, target


datasets.SVHN.__getitem__ = _svhn_getitem


# Dataset(root, train, transform, download) is expected
class ImageNet(datasets.ImageFolder):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        assert not download, "Download dataset by yourself!"
        root = pathlib.Path(root) / ('train' if train else 'val')
        super(ImageNet, self).__init__(root, transform=transform)
        import warnings

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class OriginalSVHN(datasets.SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        super(OriginalSVHN, self).__init__(root, split="train" if train else "test", transform=transform,
                                           download=download)
        self.targets = self.labels


class ExtraSVHN(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        if train:
            return (datasets.SVHN(root, split='train', transform=transform, download=download) +
                    datasets.SVHN(root, split='extra', transform=transform, download=download))
        else:
            return OriginalSVHN(root, train=False, transform=transform, download=download)


@dataclass
class VisionSet:
    """ Dataset abstraction for vision datasets.
    """

    tv_class: Type[datasets.VisionDataset]
    root: str or pathlib.Path
    num_classes: int
    default_norm: List
    default_train_da: Optional[List] = None
    default_test_da: Optional[List] = None
    collate_fn: Optional[Callable] = None

    def __post_init__(self):
        # _ is trainer
        args = {'root', 'train', 'transform', 'download'}
        _, *args_init = inspect.getfullargspec(self.tv_class.__init__).args
        _, *args_new = inspect.getfullargspec(self.tv_class.__new__).args
        if not (args <= set(args_init) or args <= set(args_new)):
            raise RuntimeError(f"tv_class is expected to have signiture of DataSet(root, train, transform, download),"
                               f"but {self.tv_class} has arguments of {args_init} instead.")
        self.root = pathlib.Path(self.root).expanduser()
        self.default_train_da = self.default_train_da or []
        self.default_test_da = self.default_test_da or []
        self._train_set = None
        self._train_loader = None
        self._val_set = None
        self._val_loader = None
        self._test_set = None
        self._test_loader = None
        print(f"self.num_classes{self.num_classes}")
        #_, *args = inspect.getfullargspec(self.tv_class).args

        #if not ({'root', 'train', 'transform', 'download'} <= set(args)):
            #raise RuntimeError(f"dataset DataSet(root, train, transform, download) is expected, "
                               #f"but {self.tv_class} has arguments of {set(args)} instead.")
        #self.root = pathlib.Path(self.root).expanduser()
        #if self.default_train_da is None:
            #self.default_train_da = []
        #if self.default_test_da is None:
            #self.default_test_da = []

    def get_dataset(self,
                    train_size: Optional[int] = None,
                    test_size: Optional[int] = None,
                    val_size: Optional[int] = None,
                    train_da: Optional[List] = None,
                    test_da: Optional[List] = None,
                    norm: Optional[List] = None,
                    download: bool = False,
                    *,
                    pre_default_train_da: Optional[List] = None,
                    post_default_train_da: Optional[List] = None,
                    post_norm_train_da: Optional[List] = None
                    ) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, Optional[datasets.VisionDataset]]:

          
        assert (download or self.root.exists()), "root does not exist"
        if train_da is None:
            train_da = list(self.default_train_da)
        if test_da is None:
            test_da = list(self.default_test_da)
        if norm is None:
            norm = list(self.default_norm)

        def unpack_optional_list(x: Optional[List]) -> List:
            return [] if x is None else x

        pre_default_train_da = unpack_optional_list(pre_default_train_da)
        post_default_train_da = unpack_optional_list(post_default_train_da)
        post_norm_train_da = unpack_optional_list(post_norm_train_da)

        train_transform = transforms.Compose(pre_default_train_da + train_da + post_default_train_da
                                             + norm + post_norm_train_da)
        train_set = self.tv_class(self.root, train=True, transform=train_transform, download=download)
        train_set.data,train_set.targets=self.change_2_long_tail(train_set,num_classes=self.num_classes)
        if train_size is not None and train_size > len(train_set):
            raise ValueError(f'train_size should be <={len(train_set)}')

        test_transform = transforms.Compose(test_da + norm)
        test_set = self.tv_class(self.root, train=False, transform=test_transform, download=download)
        test_set.data,test_set.targets=self.change_2_long_tail_decode_test(test_set,num_classes=self.num_classes)
        if test_size is not None and test_size > len(test_set):
            raise ValueError(f'test_size should be <={len(test_set)}')

        val_set = None
        if val_size is not None and val_size > 0:
            if train_size is not None and (train_size + val_size) > len(train_set):
                raise ValueError(f'train_set+val_size should be <={len(train_set)}')

            train_set, val_set = self._split_dataset(train_set, val_size)
            val_set.transform = test_transform

        return train_set, test_set, val_set

    def get_dataloader(self,
                       batch_size: int,
                       train_da: Optional[List] = None,
                       test_da: Optional[List] = None,
                       norm: Optional[List] = None,
                       train_size: Optional[int] = None,
                       test_size: Optional[int] = None,
                       val_size: Optional[int] = None,
                       download: bool = False,
                       num_workers: int = 0,
                       non_training_bs_factor=2,
                       drop_last: bool = False,
                       pin_memory: bool = True,
                       return_num_classes: bool = False,
                       test_batch_size: Optional[int] = None,
                       pre_default_train_da: Optional[List] = None,
                       post_default_train_da: Optional[List] = None,
                       post_norm_train_da: Optional[List] = None
                       ) -> (Tuple[DataLoader, DataLoader]
                             or Tuple[DataLoader, DataLoader, DataLoader]):

        """ Get data loaders for registered vision datasets. homura expects
         datasets are in `~/.torch/data/DATASET_NAME`.
        Link path if necessary, e.g. `ln -s /original/path $HOME/.torch`. Datasets can be registered
        using `homura.vision.register_dataset`

        """

        train_set, test_set, val_set = self.get_dataset(train_size, test_size, val_size,
                                                        train_da, test_da, norm, download,
                                                        pre_default_train_da=pre_default_train_da,
                                                        post_default_train_da=post_default_train_da,
                                                        post_norm_train_da=post_norm_train_da)
        if test_batch_size is None:
            test_batch_size = non_training_bs_factor * batch_size

        samplers = [None, None, None]
        if is_distributed():
            import homura

            dist_sampler_kwargs = dict(num_replicas=homura.get_world_size(),
                                       rank=homura.get_global_rank())
            samplers[0] = DistributedSampler(train_set, **dist_sampler_kwargs)
            samplers[2] = DistributedSampler(test_set, **dist_sampler_kwargs)
        else:
            samplers[0] = RandomSampler(train_set, True)

        shared_kwargs = dict(drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                             collate_fn=self.collate_fn)
        train_loader = DataLoader(train_set, batch_size, sampler=samplers[0], **shared_kwargs)
        test_loader = DataLoader(test_set, test_batch_size, sampler=samplers[2], **shared_kwargs)

        ret = [train_loader, test_loader]

        if val_set is not None:
            if is_distributed():
                samplers[1] = DistributedSampler(val_set, **dist_sampler_kwargs)
            val_loader = DataLoader(val_set, test_batch_size, sampler=samplers[1], **shared_kwargs)
            ret.append(val_loader)

        if return_num_classes:
            ret.append(self.num_classes)

        return tuple(ret)

    __call__ = get_dataloader

    @staticmethod
    def _split_dataset(train_set: datasets.VisionDataset,
                       val_size: int) -> (datasets.VisionDataset, datasets.VisionDataset):
        # split train_set to train_set and val_set
        assert len(train_set) >= val_size
        indices = torch.randperm(len(train_set))
        valset = copy.deepcopy(train_set)
        train_set.data = [train_set.data[i] for i in indices[val_size:]]
        train_set.targets = [train_set.targets[i] for i in indices[val_size:]]

        valset.data = [valset.data[i] for i in indices[:val_size]]
        valset.targets = [valset.targets[i] for i in indices[:val_size]]

        return train_set, valset
    
    @staticmethod
    def change_2_long_tail(train_set,
                           train_size=5000,
                           train_pho=0.01,
                           #val_size=1000,
                           #val_pho=1,
                           #batch_size=128,
                           num_workers=4,
                           #path='./data',
                           num_classes=10):
        
        torch.manual_seed(0) 
        np.random.seed(0)
        train_x,train_y=train_set.data,np.asarray(train_set.targets)
        num_train_samples=[]
        #num_val_samples=[]
        #train_pho=1
        #val_pho=1
        #num_classes=2
        #train_size=4000
        #val_size=1000
        train_mu=train_pho**(1./9.)
        #train_mu=1#0.01
        print(f"train_mu{train_mu},train_pho{train_pho}")
        #val_mu=val_pho**(1./9.)
        for i in range(num_classes):
            num_train_samples.append(round(train_size*(train_mu**i)))
            #num_val_samples.append(round(val_size*(val_mu**i)))
        print("num_train_samples",num_train_samples)
        train_index=[]
        #val_index=[]
        #print(train_x,train_y)
        #print(num_train_samples)
        for i in range(num_classes):
            train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
            #val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])
            #index.extend()
        #head=set([8,9])
        #tail=set([0,1])
        #print(train_index)
        random.shuffle(train_index)
        #random.shuffle(val_index)
    
        train_data,train_targets=train_x[train_index],train_y[train_index]
        #val_data,val_targets=train_x[val_index],train_y[val_index]
        #train_dataset = CustomDataset(train_data,train_targets)
        return train_data,train_targets
    
    @staticmethod
    def change_2_long_tail_decode_test(test_set,
                                       num_classes=10):
        
        test_x,test_y=test_set.data,np.asarray(test_set.targets)
        test_index=[]
        prev=0
        for i in range(num_classes):
            test_index.extend(np.where(test_y==i)[0])
            print("num_classes",i,len(test_index)-prev)
            prev=len(test_index)
            #print(
            
        test_data,test_targets=test_x[test_index],test_y[test_index]
        return test_data,test_targets
        
        
        
        
        
        
        
        
    
    
import random    

DATASET_REGISTRY = Registry('vision_datasets', type=VisionSet)

class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        #self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #img = Image.fromarray(img)

        #if self.transform is not None:
            #img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)


"""    
import random
long_tail=True
train_dataset = CIFAR10(root="~/.torch/data/cifar10", train=True, download=True)
train_x,train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
#test_x, test_y = test_dataset.data, test_dataset.targets
if long_tail:
        num_train_samples=[]
        num_val_samples=[]
        train_pho=0.1
        val_pho=1
        num_classes=10
        train_size=4000
        val_size=1000
        train_mu=train_pho**(1./9.)
        val_mu=val_pho**(1./9.)
        for i in range(num_classes):
            num_train_samples.append(round(train_size*(train_mu**i)))
            num_val_samples.append(round(val_size*(val_mu**i)))
        train_index=[]
        val_index=[]
        #print(train_x,train_y)
        print(num_train_samples,num_val_samples)
        for i in range(num_classes):
            train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
            val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])
            #index.extend()
        head=set([8,9])
        tail=set([0,1])
        random.shuffle(train_index)
        random.shuffle(val_index)
    
        train_data,train_targets=train_x[train_index],train_y[train_index]
        val_data,val_targets=train_x[val_index],train_y[val_index]
        train_dataset = CustomDataset(train_data,train_targets)
        
"""        
        
    
        
        

DATASET_REGISTRY.register_from_dict(
    {'cifar10': VisionSet(datasets.CIFAR10, "~/.torch/data/cifar10", 10,
                          [transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                          [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                           transforms.RandomHorizontalFlip()]),

     'cifar100': VisionSet(datasets.CIFAR100, "~/.torch/data/cifar100", 100,
                           [transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                           [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                            transforms.RandomHorizontalFlip()]),

     'SVHN': VisionSet(OriginalSVHN, "~/.torch/data/svhn", 10,
                       [transforms.ToTensor(),
                        transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                       [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
                       ),

     'imagenet': VisionSet(ImageNet, get_environ('IMAGENET_ROOT', '~/.torch/data/imagenet'), 1_000,
                           [transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                           [transforms.RandomResizedCrop(
                               224), transforms.RandomHorizontalFlip()],
                           [transforms.Resize(256), transforms.CenterCrop(224)]),

     }
)
