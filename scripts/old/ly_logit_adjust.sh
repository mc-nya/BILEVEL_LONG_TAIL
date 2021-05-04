source ~/.bashrc
conda activate pytorch
python bilevel_logit_adjust.py --dataset Cifar100 --lr 0.1 --arch_lr 0.05 --save_path ./results/ly_logit_cifar100 --batch_size=128 --ly True --dy False --train_rho 0.01