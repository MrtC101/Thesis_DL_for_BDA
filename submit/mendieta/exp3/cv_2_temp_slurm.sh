#!/bin/bash

#SBATCH --job-name="cv_2"
#SBATCH --output=/home/mcogo/scratch/exp3_aug/out/jobs/cv_2.log
#SBATCH --error=/home/mcogo/scratch/exp3_aug/out/jobs/cv_2.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00
##SBATCH --nodelist=ivb09

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp3/cv_2_run.sh"
