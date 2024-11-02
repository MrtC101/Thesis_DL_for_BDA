#!/bin/bash

#SBATCH --job-name="post"
#SBATCH --output=/home/mcogo/scratch/exp8_a/out/jobs/post.log
#SBATCH --error=/home/mcogo/scratch/exp8_a/out/jobs/post.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=short
#SBATCH --time=00:25:00
##SBATCH --nodelist=ivb09

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp8_a/post_run.sh"
