#!/bin/bash

#SBATCH --job-name="pre"
#SBATCH --output=/home/mcogo/scratch/exp2_not_aug/out/jobs/pre.log
#SBATCH --error=/home/mcogo/scratch/exp2_not_aug/out/jobs/pre.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp2/pre_run.sh"
