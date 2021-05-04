source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.1 --arch_lr 0.0001 --save_path ./results/dy_ly_loss_cifar100 --dataset Cifar100 --ly True --dy True --train_rho 0.01