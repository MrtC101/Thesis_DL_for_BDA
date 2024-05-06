from utils.files.common import read_json, dump_json
from utils.files.datasets import TilesDataset
from collections import defaultdict
from shapely import wkt
import argparse
from tqdm import tqdm
import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.logger import get_logger
l = get_logger("Compute data from images")


def compute_mean_stddev(pre_img, post_img):
    mean = {}
    for time, img in zip(["pre", "post"], [pre_img, post_img]):
        mean[time] = {}
        norm_img = img / 255.0
        mean[time]["mean"] = {
            "R": norm_img[:, :, 0].mean(),
            "G": norm_img[:, :, 1].mean(),
            "B": norm_img[:, :, 2].mean(),
        }
        mean[time]["stdv"] = {
            "R": norm_img[:, :, 0].std(),
            "G": norm_img[:, :, 1].std(),
            "B": norm_img[:, :, 2].std(),
        }
    return mean


def count_buildings(pre_json, post_json):
    count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for time, file in zip(["pre", "post"], [pre_json, post_json]):
        for coord in file['features']['xy']:
            feature_type = coord['properties']['feature_type']
            if feature_type != 'building':
                count[time]["not_building"][feature_type] += 1

            damage_class = coord['properties'].get('subtype', 'no-subtype')
            feat_shape = wkt.loads(coord['wkt'])

            count[time]["bld_area"][damage_class] += feat_shape.area
            count[time]["bld_class"][damage_class] += 1

    return count


def create_data_dicts(split_json_path, data_dict_folder):

    assert os.path.isfile(split_json_path), f"{split_json_path} is not a file."
    assert split_json_path.split(
        ".")[1] == "json", f"{split_json_path} is not a json file."

    splits = read_json(split_json_path)

    mean = defaultdict(lambda: {})
    count = defaultdict(lambda: {})

    for set in ["train", "val"]:
        curr_tileset = TilesDataset(splits[set])
        l.info(f'Counting {set} set with length: {len(curr_tileset)}')
        for data_i in tqdm(curr_tileset):
            dis_id = data_i["dis_id"]
            tile_id = data_i["tile_id"]
            mean[dis_id][tile_id] = compute_mean_stddev(
                data_i["pre_img"], data_i["post_img"])
            count[dis_id][tile_id] = count_buildings(
                data_i["pre_json"], data_i["post_json"])

    mean_path = os.path.join(data_dict_folder, "all_tiles_mean_stdev.json")
    count_path = os.path.join(
        data_dict_folder, "all_tiles_building_count_and_area.json")
    dump_json(mean_path, mean)
    dump_json(count_path, count)

    l.info(f'Total counting by each disaster.')
    count = read_json(count_path)
    c_by_d = {}
    for zone_id, zone in tqdm(count.items()):
        c_by_d[zone_id] = {
            "pre": {
                "bld_area": defaultdict(int),
                "bld_class": defaultdict(int)
            },
            "post": {
                "bld_area": defaultdict(int),
                "bld_class": defaultdict(int)
            }
        }
        for tile_id, tile in zone.items():
            for time_id, time in tile.items():
                for m_id, mesure in time.items():
                    for val_id, value in mesure.items():
                        c_by_d[zone_id][time_id][m_id][val_id] += value

    mean_disaster_path = os.path.join(
        data_dict_folder, "all_tiles_mean_stdev_by_disaster.json")
    dump_json(mean_disaster_path, c_by_d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
    parser.add_argument(
        'split_json_path',
        type=str,
        help=('Path to the file that train and val split sets. ')
    )
    parser.add_argument(
        'data_dicts_path',
        type=str,
        help=('Path to save files created.')
    )
    args = parser.parse_args()
    create_data_dicts(args.split_json_path, args.data_dicts_path)
