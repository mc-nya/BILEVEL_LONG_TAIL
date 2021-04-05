source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 50 --ARCH_END 140 --ARCH_EPOCH_INTERVAL 5 --lr 0.1 --arch_lr 0.1 --save_path ./results/dy_ly_late --ly True --dy True --train_rho 0.1