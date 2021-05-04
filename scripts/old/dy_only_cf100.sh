source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 80 --lr 0.1 --arch_lr 0.001 --save_path ./results/dy_only_loss_cifar100_1 --dataset Cifar100 --ly False --dy True --train_rho 0.01