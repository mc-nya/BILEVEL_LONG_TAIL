source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 120 --lr 0.1 --arch_lr 0.01 --save_path ./results/loss_adjust_ly_pretrain --dataset Cifar10 --ly True --dy True --train_rho 0.01