#!/bin/bash
#conda init
#conda activate nlrc
train(){
    # borrar extra
    python /original_siames/data/delete_extra.py /original_siames/public_datasets/xBD/raw/train
    # crear mascaras
    python /original_siames/data/create_label_masks.py /original_siames/public_datasets/xBD/raw/train -b 2
    # crear splits
    python /original_siames/data/split.py  /original_siames/public_datasets/xBD /original_siames/public_datasets/xBD/raw/train/labels
    # crear chips
    python /original_siames/data/make_smaller_tiles.py
    # calcular stdv
    python /original_siames/data/compute_mean.py /original_siames/public_datasets/xBD/
    # crear shards
    python /original_siames/data/make_data_shards.py
    # ejecutar train
    python /original_siames/train/train.py
    # ejecutar test
    python /original_siames/inference/inference.py --output_dir /original_siames/out/ --data_img_dir /original_siames/public_datasets/xBD/final_mdl_all_disaster_splits/ --data_inference_dict /original_siames/constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json --data_mean_stddev /original_siames/constants/splits/all_disaster_mean_stddev_tiles_0_1.json --label_map_json /original_siames/constants/class_lists/xBD_label_map.json --model /original_siames/models/model_best.pth.tar --experiment_name try
}
start=`date +%s`
train > /original_siames/out/console.txt
end=`date +%s`
#conda deactivate
echo "($start,$end) Execution time was `expr $end - $start` seconds." > /original_siames/out/time.txt
