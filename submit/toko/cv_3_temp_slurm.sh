#!/bin/bash

#SBATCH --job-name="cv_3"
#SBATCH --output="/home/mcogo/scratch/not_aug/out/jobs/cv_3/out.log"
#SBATCH --error="/home/mcogo/scratch/not_aug/out/jobs/cv_3/out.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=ivb09

/bin/bash -c "/home/mcogo/scratch/submit/toko/cv_3_run.sh"
