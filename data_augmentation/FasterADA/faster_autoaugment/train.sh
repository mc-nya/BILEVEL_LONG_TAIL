#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12-02:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=Mtra_FasterADA
#SBATCH --output=output_model_wrn40_2_10class_withoutDA.txt
#SBATCH --mail-user=ychan199@ucr.edu
#SBATCH --mail-type=FAIL
source ~/.bashrc
conda activate py38
python train.py path=/home/csgrad/ychan/dda/faster_autoaugment/policy_weights/cifar10/19.pt data.name=cifar10 model.name=wrn40_2 data.download=true
