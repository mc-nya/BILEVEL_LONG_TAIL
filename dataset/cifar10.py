from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset,Dataset
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader
import numpy as np
import random
def load_cifar10(train_size=4000,train_rho=0.01,val_size=1000,val_rho=1,image_size=224,batch_size=128,num_workers=4,path='./data',num_classes=10):
    train_transform = Compose([
        RandomCrop(32,padding=4),
        Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = Compose([
        Resize(image_size, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_dataset = CIFAR10(root=path, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=path, train=False, transform=test_transform, download=True)
    train_x,train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    #test_x, test_y = test_dataset.data, test_dataset.targets
    num_train_samples=[]
    num_val_samples=[]
    train_mu=train_rho**(1./9.)
    val_mu=val_rho**(1./9.)
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
    random.shuffle(train_index)
    random.shuffle(val_index)
    
    train_data,train_targets=train_x[train_index],train_y[train_index]
    val_data,val_targets=train_x[val_index],train_y[val_index]
    
    train_dataset = CustomDataset(train_data,train_targets,train_transform)
    val_dataset = CustomDataset(val_data,val_targets,train_transform)
    train_eval_dataset = CustomDataset(train_data,train_targets,test_transform)
    val_eval_dataset = CustomDataset(val_data,val_targets,test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=True, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)
    eval_val_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)

    return train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples

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
#load_cifar10()