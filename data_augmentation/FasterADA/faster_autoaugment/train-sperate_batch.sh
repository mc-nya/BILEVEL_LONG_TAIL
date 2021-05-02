#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12-02:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=Mtra_FasterADA
#SBATCH --output=output_model_wrn40_2_10class_seperate.txt
#SBATCH --mail-user=ychan199@ucr.edu
#SBATCH --mail-type=FAIL
source ~/.bashrc
conda activate py38
python search.py data.name=cifar10 data.download=true 
python train.py path=/home/csgrad/ychan/dda/faster_autoaugment/policy_weights/cifar10 +first_append=/policy1/19.pt +second_append=/policy2/19.pt data.name=cifar10 model.name=wrn40_2 data.download=true
