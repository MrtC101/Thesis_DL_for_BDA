#!/bin/bash

#SBATCH --job-name="final"
#SBATCH --output=/home/mcogo/scratch/exp5/out/jobs/final.log
#SBATCH --error=/home/mcogo/scratch/exp5/out/jobs/final.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00
##SBATCH --nodelist=ivb09

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp5/final_run.sh"
