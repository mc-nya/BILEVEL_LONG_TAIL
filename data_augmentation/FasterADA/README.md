# Faster AutoAugment (For this project, this readme is different from original author)

This is the official *re*implementation of FasterAutoAugment ([hataya2020a](https://arxiv.org/abs/1911.06987).)

## Requirements

* `Python>=3.8`  # Developed on 3.8. It may work with earlier versions.
* `PyTorch==1.5.0` # Required by `kornia`
* `torchvision==0.6.0`
* `kornia==0.3.1`
* `homura==2020.07` # `pip install -U git+https://github.com/moskomule/homura@v2020.07`
* `hydra==0.11` 

If there are issue with the visionset in the TraceBack Error information, you can refer here:

https://github.com/moskomule/dda/issues/7


### Search

```
python search.py [data.name={cifar10,cifar100,svhn}] [...]
```

This script will save the obtained policy at `policy_weight/DATASET_NAME/EPOCH.pt`.

### Train

```
python train.py path=PATH_TO_POLICY_WEIGHT [data.name={cifar10,cifar100,svhn}] [model.name={wrn28_2,wrn40_2,wrn28_10}]  [...]
```

When `path` is not specified, training is executed without policy, which can be used as a baseline.

Or you can refer to the shell script for batch running in slurm

## Notice

The codebase here is not exactly the same as the one used in the paper. 
For example, this codebase does not include the support for `DistributedDataParallel` and the custom `CutOut` kernel. 

The data loader here is the simple version of setting up experiements, if you want to be aligened with the original library setting as the author, you need to go to
```
envs/{your_env_name}/lib/{your_python_version_at_the_env}/site-packages/homura/vision/data/datasets.py to modify it (if you used conda).  
```
For example, adding under class VisionSet:
```
@staticmethod
def change_2_long_tail(train_set,
                           train_size=5000,
                           train_pho=0.01,
                           val_size=1000,
                           val_pho=1,
                           batch_size=128,
                           num_workers=4,
                           path='./data',
                           num_classes=10):

```

And change the get_dataloader method, detailed part you can read the example code in the folder, datasets_example.py

## Other:
From now, the individualized ADA used two methods:  
Option 1 : at every iteration seperate the data by class to different policy and train it.  
Option 2 : seperated majority, minority at first, and upsampling if needed, then traing two of them at two different batch.  
