#!/bin/bash
#SBATCH --job-name=enhi
#SBATCH --partition=short
##   SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time 0-06:00:00
#SBATCH --output=train_enhi1.log
#SBATCH --mail-type=END

module load use.own
module add sentencepiece/0.0
module add python/3.7.0
module add pytorch/1.1.0

set -x;
set -o;

python3 exp1.py 
