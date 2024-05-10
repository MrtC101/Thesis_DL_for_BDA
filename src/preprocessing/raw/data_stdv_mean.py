import os
import sys
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))
from utils.common.logger import get_logger
l = get_logger("data_stdv_mean")


from utils.common.files import read_json, dump_json, is_json, is_file, is_dir
from utils.datasets.raw_datasets import RawDataset
from collections import defaultdict
from shapely import wkt, Polygon, geometry
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

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
            feature_type = coord['properties']['feature_type'][0]
            if feature_type != 'building':
                count[time]["not_building"][feature_type] += 1

            damage_class = coord['properties'].get('subtype', ['no-subtype'])[0]
            
            feat_shape = wkt.loads(coord['wkt']).tolist()

            if("bld_area" not in count[time].keys()):
                count[time]["bld_area"] = defaultdict(float)

            count[time]["bld_area"][damage_class] += feat_shape[0].area
            count[time]["bld_class"][damage_class] += 1

    return count


def count_by_disaster(count: dict):
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
    return c_by_d


def create_data_dicts(splits_json_path : str,out_path : str):
    is_dir(out_path)
    dicts_path = os.path.join(out_path,"dataset_statistics")    
    os.makedirs(dicts_path,exist_ok=True)

    mean = defaultdict(lambda: {})
    count = defaultdict(lambda: {})

    for split_name in ["train", "val"]:
        dataset = RawDataset(split_name=split_name,splits_json_path=splits_json_path)
        loader = DataLoader(dataset,2)
        l.info(f'Counting {split_name} subset with length: {len(loader)}')
        for dis_id,tile_id,data_i in tqdm(loader):
            mean[dis_id][tile_id] = compute_mean_stddev(data_i["pre_image"], data_i["post_image"])
            count[dis_id][tile_id] = count_buildings(data_i["pre_json"], data_i["post_json"])

    mean_path = os.path.join(dicts_path, "all_tiles_mean_stdev.json")
    count_path = os.path.join(dicts_path, "all_tiles_count_area.json")
    dump_json(mean_path, mean)
    dump_json(count_path, count)

    l.info(f'Total counting by each disaster.')
    c_by_d = count_by_disaster(count=count)
    mean_disaster_path = os.path.join(dicts_path, "all_tiles_mean_stdev_by_disaster.json")
    dump_json(mean_disaster_path, c_by_d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates 3 json files that counts building classes, pixel mean and stdv for each image in xBD dataset.')
    parser.add_argument(
        'split_json_path',
        type=str,
        help=('Path to the json file with train,val and test sets.')
    )
    parser.add_argument(
        'out_path',
        type=str,
        help=('Path to save all json files created.')
    )
    args = parser.parse_args()
    create_data_dicts(args.split_json_path, args.out_path)
