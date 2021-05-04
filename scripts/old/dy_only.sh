source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 400 --lr 0.1 --arch_lr 0.0001 --save_path ./results/dy_only_loss_cifar10_2 --dataset Cifar10 --ly False --dy False --train_rho 0.01