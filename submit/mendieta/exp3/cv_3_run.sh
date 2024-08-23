#!/bin/bash

source /home/mcogo/scratch/submit/mendieta/exp3/cv_3_temp_env.sh

output_file="$OUT_PATH/time.txt"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate develop

# Start timer
start=$(date +%s)

# Run the training script
for file in ${FILE_LIST[@]}; do
    python "${file}"
done

# End timer
end=$(date +%s)
conda deactivate

# Calculate and print execution time
execution_time=$((end - start))
echo "($start,$end) Execution time was ${execution_time} seconds." > "$output_file"
