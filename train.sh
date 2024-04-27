#!/bin/bash
conda activate nlrc
# borrar extra
python ./data/delete_extra.py public_datasets/xBD/raw/train
# crear mascaras
python ./data/create_label_masks.py public_datasets/xBD/raw/train -b 2
# crear splits
python ./data/split.py  public_datasets/xBD public_datasets/xBD/raw/train/labels
# crear chips
python ./data/make_smaller_tiles.py
# calcular stdv
python ./data/compute_mean.py public_datasets/xBD/
# crear shards
python ./data/make_data_shards.py
# ejecutar train
#python ./train/train.py
# ejecutar test
python ./inference/inference.py --output_dir out/ --data_img_dir public_datasets/xBD/final_mdl_all_disaster_splits/ --data_inference_dict constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json --data_mean_stddev constants/splits/all_disaster_mean_stddev_tiles_0_1.json --label_map_json constants/class_lists/xBD_label_map.json --model models/model_best.pth.tar --experiment_name try
conda deactivate