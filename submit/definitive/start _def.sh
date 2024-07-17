#!/bin/bash

folder_path="/home/mcogo/scratch"
train_py="${folder_path}/src/run_definitive_traning.py"
output_file="${folder_path}/out/time.txt"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate develop

# src path var
export PROJ_PATH="$folder_path"

# Start timer
start=$(date +%s)

# Run the training script
python "$train_py"

# End timer
end=$(date +%s)
conda deactivate

# Calculate and print execution time
execution_time=$((end - start))
echo "($start,$end) Execution time was ${execution_time} seconds." > "$output_file"

