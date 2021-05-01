#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12-02:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=2way_FasterADA
#SBATCH --output=output_test_2way_sampling.txt
#SBATCH --mail-user=ychan199@ucr.edu
#SBATCH --mail-type=FAIL
source ~/.bashrc
conda activate py38
python search.py data.name=cifar10 data.download=true
