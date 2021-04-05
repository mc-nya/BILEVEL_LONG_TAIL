source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.01 --arch_lr 0.003 --save_path ./results/plain --ly False --dy False --train_rho 0.1 --ARCH_EPOCH 300