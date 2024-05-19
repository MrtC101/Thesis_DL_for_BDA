#!/bin/bash
folder_path="/home/mrtc101/Desktop/tesina/repo/my_siames"
conda init
conda activate nlrc
start=`date +%s`
train_py=$folder_path/src/train_pipeline.py
python  $train_py
end=`date +%s`
conda deactivate
echo "($start,$end) Execution time was `expr $end - $start` seconds." > $folder_path/out/time.txt
