source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --ARCH_EPOCH 10 --lr 0.1 --arch_lr 0.1 --save_path ./results/dy_ly_new --ly True --dy True --train_rho 0.1