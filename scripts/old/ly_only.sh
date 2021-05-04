source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.1 --arch_lr 0.05 --save_path ./results/ly_only_loss_cifar10_2 --dataset Cifar10 --ly True --dy False --train_rho 0.01