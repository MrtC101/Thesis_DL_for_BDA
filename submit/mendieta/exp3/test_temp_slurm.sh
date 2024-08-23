#!/bin/bash

#SBATCH --job-name="run_on_test"
#SBATCH --output=/home/mcogo/scratch/exp3_aug/out/jobs/run_on_test.log
#SBATCH --error=/home/mcogo/scratch/exp3_aug/out/jobs/run_on_test.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multi
#SBATCH --time=2-00:00:00

/bin/bash -c "/home/mcogo/scratch/submit/mendieta/exp3/test_run.sh"
