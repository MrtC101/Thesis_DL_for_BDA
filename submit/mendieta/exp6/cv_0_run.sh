#!/bin/bash

source /home/mcogo/scratch/submit/mendieta/exp6/cv_0_temp_env.sh

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate develop

# Run the training script
for file in ${FILE_LIST[@]}; do
python "${file}"
done

conda deactivate
