#!/bin/bash
conda init
conda activate production
start=`date +%s`
python /original_siames/src/train_pipeline.py 
end=`date +%s`
conda deactivate
echo "($start,$end) Execution time was `expr $end - $start` seconds." > /original_siames/out/time.txt
