source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 401 --lr 0.1 --arch_lr 0.01 --save_path ./results/ly_only_logit_adjust_loss_cifar100_2 --dataset Cifar100 --ly False --dy False --train_rho 0.01