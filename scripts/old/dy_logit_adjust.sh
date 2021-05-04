source ~/.bashrc
conda activate pytorch
python bilevel_logit_adjust.py --dataset Cifar100 --lr 0.1 --arch_lr 0.05 --save_path ./results/dy_logit_cifar100 --batch_size=128 --ly False --dy True --train_rho 0.01