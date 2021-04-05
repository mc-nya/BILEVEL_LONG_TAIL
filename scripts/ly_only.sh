source ~/.bashrc
conda activate pytorch
python bilevel_loss_adjust.py --lr 0.1 --arch_lr 0.1 --save_path ./results/ly_only_neg --ly True --dy False --train_rho 0.1