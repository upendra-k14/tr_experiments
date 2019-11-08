#!/bin/bash
#SBATCH --reservation non-deadline-queue
#SBATCH --job-name=enfr
#SBATCH --partition=long
#SBATCH --account=upendra
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=10G
#SBATCH --time 02-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH --output=exp1.log
#SBATCH --mail-type=END

module load use.own
module add sentencepiece/0.0
module add python/3.7.0
module add pytorch/1.1.0
module add pytorch/fairseq/0.7.2

set -x;
set -o;

function _export {
    # ssh upendra@ada "mkdir -p /share1/upendra/tr_experiments/hien/vocab8k/checkpoints"
    rsync -zavh /ssd_scratch/cvit/upendra/tr_experiments/hi_en_exp/vocab8k/exp1/checkpoints/checkpoint_{best,last}.pt .
}

trap "_export" SIGHUP

# conda deactivate
python3 exp1.py

_export


