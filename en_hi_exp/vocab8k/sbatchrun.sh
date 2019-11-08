#!/bin/bash
#SBATCH --job-name=enhi
#SBATCH --partition=long
#SBATCH --account=upendra
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time 05-00:00:00
#SBATCH --output=exp1.log
#SBATCH --mail-type=END

module load use.own
module add sentencepiece/0.0
module add python/3.7.0
module add pytorch/1.1.0
module add pytorch/fairseq/0.7.2

set -x;
set -o;

python3 exp1.py

rsync -zavh /ssd_scratch/cvit/upendra/tr_experiments/en_hi_exp/vocab8k/exp1/checkpoints/checkpoint_{best,last}.pt .
