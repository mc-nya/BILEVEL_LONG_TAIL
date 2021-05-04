source ~/.bashrc
conda activate pytorch
python plain_training.py --dataset Cifar100 --lr 0.1 --batch_size 128 --arch_lr 0.003 --epoch 300 --save_path ./results/plain_cifar100 --train_rho 0.01