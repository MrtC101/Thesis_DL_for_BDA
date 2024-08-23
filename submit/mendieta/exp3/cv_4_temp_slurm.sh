#!/bin/bash

#SBATCH --job-name="cv_4"
#SBATCH --output=/home/mcogo/scratch/exp3_aug/out/jobs/cv_4.log
#SBATCH --error=/home/mcogo/scratch/exp3_aug/out/jobs/cv_4.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=18
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00
##SBATCH --nodelist=ivb09

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp3/cv_4_run.sh"
