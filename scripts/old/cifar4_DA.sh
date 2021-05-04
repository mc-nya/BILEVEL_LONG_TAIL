source ~/.bashrc
conda activate pytorch
python cifar4_DA.py --dataset Cifar4 --ARCH_EPOCH 80 --dy False --ly True --augmentation weak --lr 0.1 --arch_lr 0.01 --batch_size 128 --arch_lr 0.003 --epoch 200 --save_path ./results/aug_cifar4_logit --train_rho 0.01