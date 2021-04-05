source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.1 --arch_lr 0.1 --save_path ./results/dy_only_neg --ly False --dy True --train_rho 0.1