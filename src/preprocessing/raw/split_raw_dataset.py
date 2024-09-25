# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import math
from tqdm import tqdm
import random
import pandas as pd
from collections import defaultdict
from utils.common.pathManager import FilePath
from utils.pathManagers.rawManager import RawPathManager
from utils.loggers.console_logger import LoggerSingleton
log = LoggerSingleton()


def get_buildings(tiles_dict: dict) -> pd.DataFrame:
    """Iterates over all label json files and returns a `pd.Dataframe` with all buildings."""
    buildings = []
    for tile_id, tile_dict in tiles_dict.items():
        file_path = FilePath(tile_dict["post"]["json"])
        label_json = file_path.read_json()
        bld_list = label_json['features']['xy']
        if not len(bld_list) > 0:
            buildings.append([tile_id, "no-buildings"])
        for i, building in enumerate(bld_list):
            dmg_label = building['properties'].get(
                'subtype', 'no-damage')
            buildings.append([tile_id, i, dmg_label])
    df = pd.DataFrame(buildings, columns=["tile_id", "list_id", "label"])
    df.set_index(["tile_id"], inplace=True)
    return df


all_clases = ['destroyed', 'major-damage', 'minor-damage', 'no-damage',
              'un-classified', 'no-buildings']


def get_tiles_count(building_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a pd.Dataframe with the count of buildings per class inside each tile."""
    tile_df = building_df.value_counts(["tile_id", "label"]).unstack()
    tile_df[tile_df.isna()] = 0
    tile_df = tile_df.astype(int)
    for col in all_clases:
        if col not in tile_df.columns:
            tile_df[col] = 0
    return tile_df[all_clases]


def create_test_split(tiles_dict: dict, test_n: int) -> dict:
    """Returns a dictionary of test tiles, randomly selected from `tiles_dict`.

    Returns:
        dict: A dictionary containing randomly chosen tiles from `tiles_dict`.
    """

    sampled_keys = random.sample(list(tiles_dict.keys()), test_n)
    sample = {key: tiles_dict[key] for key in sampled_keys}
    for key in sampled_keys:
        tiles_dict.pop(key)
    return sample


def create_train_split(tiles_dict, dis_id, train_n):
    """ Creates a train split using the building count per tile."""
    building_df = get_buildings(tiles_dict)
    tiles_df = get_tiles_count(building_df)

    if train_n < len(tiles_df):
        prop_df = tiles_df.sum(axis=0)
        log.info(f"{dis_id} original set of len {len(tiles_df)} " +
                 f"with original proportion for each {prop_df}")

        prop_df = (prop_df / len(tiles_df) * train_n).apply(math.ceil)
        train_keys, count_df = select_train_tiles(building_df, tiles_df, prop_df)
        sample = {key: tiles_dict[key] for key in train_keys}
        log.info(f"{dis_id} train split of len {len(tiles_df)} " +
                 f"with proportion for each label {count_df}")
    else:
        sample = tiles_dict

    return sample


def select_train_tiles(building_df, tiles_df, prop_df):
    """Selects tiles to add to split train iteratively trying to balance the dataset."""
    train_keys = set()
    train_count = prop_df * 0
    for label in prop_df.sort_values().keys():
        filter_df = building_df[building_df["label"] == label].copy()
        while (train_count <= prop_df).all() and len(filter_df) > 0:
            bld = filter_df.sample(n=1, replace=False)
            tile_id = bld.index[0]
            filter_df.drop(tile_id, inplace=True)
            count = tiles_df.loc[tile_id]
            new_train_count = train_count + count
            if tile_id not in train_keys and (new_train_count <= prop_df).all():
                train_keys.add(tile_id)
                train_count = new_train_count
    for label in prop_df.sort_values().keys():
        if (train_count.loc[label] == 0 and prop_df.loc[label] > 0):
            bld = building_df[building_df["label"] == label].sample(n=1)
            tile_id = bld.index[0]
            train_keys.add(tile_id)
            count = tiles_df.loc[tile_id]
            train_count = train_count + count
    return train_keys, train_count


def save_splits(splits_dict, out_path):
    """Save the dictionary in out_path"""
    split_path = out_path.join("splits").create_folder()
    split_file = split_path.join("raw_splits.json")
    split_file.save_json(splits_dict)
    return split_file


def save_proportions(splits_dict: dict, out_path: FilePath):
    """Computes and saves the corresponding proportional weights computed from building count"""
    tiles = {f"{dis_id}-{tile_id}": tile
             for dis_id, disaster in splits_dict["train"].items()
             for tile_id, tile in disaster.items()}
    building_df = get_buildings(tiles)
    tiles_df = get_tiles_count(building_df)
    weights_df = tiles_df.sum().sum() / tiles_df.sum()
    weights_df[weights_df.isna()] = 0.0
    weights = {label: w for label, w in weights_df.items()
               if label in ['destroyed', 'major-damage', 'minor-damage', 'no-damage']}
    path = out_path.join("..", "out")
    path.join("train_weights.json").save_json(weights)


def stratified_split_dataset(xbd_path, data_path, split_prop,
                             disasters_of_interest, total_tiles):
    """
    Splits dataset into stratified train and test sets.

    Parameters:
    raw_path (str): Path to the raw dataset.
    out_path (str): Path to save the split dataset.
    split (dict): Dictionary with 'train' and 'test' split proportions.
    n (int): Number of tiles per disaster.

    Returns:
    str: Path to the JSON file with the splits.
    """
    disasters = RawPathManager.load_paths(xbd_path, disasters_of_interest)
    test_n = math.ceil(total_tiles * split_prop["test"])
    train_n = math.floor(total_tiles * split_prop["train"])

    splits_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for dis_id, tiles_dict in disasters.items():
        splits_dict["test"][dis_id] = create_test_split(
            tiles_dict, test_n)
        splits_dict["train"][dis_id] = create_train_split(
            tiles_dict, dis_id, train_n)
        msg = f"{dis_id} length {len(tiles_dict)}," + \
            " desired length {total_tiles}, "
        tot_len = 0
        for key, set in splits_dict.items():
            tot_len += len(set[dis_id])
            msg += f"{key} {len(set[dis_id])} "  # For loggin strings
        log.info(msg + f", stratified length {tot_len}")

    save_proportions(splits_dict, data_path)
    return save_splits(splits_dict, data_path)


def leave_only_n(data_path: FilePath, n: int) -> None:
    """Pick n random disaster tiles and deletes all the rest of files from
    disaster Dataset.

    Args:
        data_path: Path to the directory to the xBD dataset that contains
        subsets of xBD dataset.
        n: number of disasters that will be left in the folder.

    Example:
        >>> leave_only_n("data/xBD/raw",45)
    """
    data = RawPathManager.load_paths(data_path)
    tot_tiles = [tile for tiles in data.values()
                 for tile in tiles.values()]
    total_tiles = len(tot_tiles)
    if (total_tiles > n):
        log.info(
            f"Deleting {total_tiles-n} tiles of {total_tiles} total tiles.")

        ids = set(random.sample(range(total_tiles), total_tiles-n))
        for id in tqdm(ids):
            disaster = tot_tiles[id]
            for tile in disaster.values():
                for file in tile.values():
                    FilePath(file).remove()

        log.info(f"Files {total_tiles-n}  removed. {n} tiles left.")
    else:
        log.info(
            f"There are {total_tiles} of {n} requiere tiles. Skipped...")
    log.info(f"There are {total_tiles-n} total tiles." +
             f"From {len(list(data.keys()))} disasters. ")
    return


def split_dataset(raw_path: FilePath, out_path: FilePath,
                  splits: dict) -> FilePath:
    """Creates a dictionary representing the splits of the xBD dataset.
        The dictionary is saved in a new json file\
        `raw_splits.json` inside `out_path/splits` folder.
        The splits are made based on `splits` dictionary.

    Args:
        raw_path: Path to the folder containing subsets from the xBD dataset.
        out_path: Path to the folder where the new JSON file will be saved.
        splits: Dictionary of pairs (key,value) where the key is the split
        set's name and it's value is the proportion of the dataset.

    Return:
        FilePath: Path to the new `raw_splits.json` file.

    Example:
        >>> split_dataset("data/xBD/raw","data/xBD")
    """

    # creates splits folder
    split_path = out_path.join("splits")
    split_path.create_folder()

    # loads raw dataset
    data_dict = RawPathManager.load_paths(raw_path)
    tiles_file = split_path.join("all_disaster.json")
    tiles_file.save_json(data_dict)

    assert sum([val for val in splits.values()]) == 1.0, \
        "splits proportions must sum 1.0"

    splits_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for disaster_name, tiles_dict in tqdm(data_dict.items()):

        tiles_ids = list(tiles_dict.keys())
        random.shuffle(tiles_ids)

        if (len(tiles_ids) < len(splits)):
            raise Exception(f"{disaster_name} disaster number must be more\
                             than {len(splits)}.")

        # works fine if len(tiles_ids) >= 10 for each disaster
        ids = {}
        last_top = 0
        for i, (name, proportion) in enumerate(splits.items()):
            current_top = last_top + \
                math.floor(proportion * len(tiles_ids))
            if i < len(splits) - 1:
                ids[name] = sorted(tiles_ids[last_top:current_top])
            else:
                ids[name] = sorted(tiles_ids[last_top:])
            last_top = current_top

        msg = f"{disaster_name} length {len(tiles_ids)}, "
        for set in splits.keys():
            for id in ids[set]:
                splits_dict[set][disaster_name][id] = tiles_dict[id]
            msg += f"{set} {len(ids[set])} "  # For loggin strings

        log.info(msg)

    split_file = split_path.join("raw_splits.json")
    split_file.save_json(splits_dict)
    return split_file
