source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.1 --arch_lr 0.00
1 --save_path ./results/dy_ly_loss_cifar10 --dataset Cifar10 --ly True --dy True --train_rho 0.01