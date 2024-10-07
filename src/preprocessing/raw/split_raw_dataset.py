# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from utils.common.pathManager import FilePath
from utils.pathManagers.rawManager import RawPathManager
from utils.loggers.console_logger import LoggerSingleton
log = LoggerSingleton()


def get_tiles_df(tiles_dict: dict) -> pd.DataFrame:
    """Iterates over all label json files and returns a `pd.Dataframe` with
    all count of buildings per tiles."""
    labels = ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"]

    tiles = []
    for tile_id, tile_dict in tiles_dict.items():
        # Open json label file
        file_path = FilePath(tile_dict["post"]["json"])
        label_json = file_path.read_json()
        bld_list = label_json['features']['xy']

        # count buildings
        label_count = np.zeros(len(labels), dtype=np.uint16)
        for building in bld_list:
            i = labels.index(building['properties'].get('subtype'))
            label_count[i] += 1
        # Append row
        tiles.append([tile_id] + label_count.tolist())
    cols = ["tile_id"] + labels
    df = pd.DataFrame(tiles, columns=cols)
    df.set_index(["tile_id"], inplace=True)
    return df


def stratified_sample(tiles_keys_list, tiles_df, prop, curr_prop):
    """Ensures that all splits has a propotional number of buildings from each class"""
    avaliable_tiles: pd.DataFrame = tiles_df.loc[tiles_keys_list, :]
    tiles_ids = []
    for lab in avaliable_tiles.columns[0:4]:
        candidates_df: pd.DataFrame = avaliable_tiles[avaliable_tiles[lab] > 0]
        n_labels = math.floor(len(candidates_df) * (prop / curr_prop))
        log.info(f"{len(candidates_df)} candidates | {n_labels} taken for {lab}")
        lab_tiles_ids = list(candidates_df.sample(n_labels).index)
        avaliable_tiles = avaliable_tiles.drop(candidates_df.index)
        tiles_ids += lab_tiles_ids
    return tiles_ids


def split_disaster(tiles_dict: dict, split_props: dict) -> dict:
    """split a distater dict of tiles into the corresponding split proportion"""
    tiles_df = get_tiles_df(tiles_dict)
    tiles_keys_list: list = list(tiles_dict.keys())
    split = {}
    curr_prop = 1.0
    for split_name, prop in split_props.items():
        log.info(f"Split {split_name}")
        split_n = math.floor(len(tiles_keys_list) * (prop / curr_prop))
        if split_n < len(tiles_keys_list):
            curr_tiles_keys = stratified_sample(tiles_keys_list, tiles_df, prop, curr_prop)
            # curr_tiles_keys = random.sample(tiles_keys_list, split_n)
            split[split_name] = {f"{tile_id}": tiles_dict[tile_id] for tile_id in curr_tiles_keys}
            for key in split[split_name]:
                tiles_keys_list.remove(key)
            curr_prop -= prop
        else:
            split[split_name] = {f"{tile_id}": tiles_dict[tile_id] for tile_id in tiles_keys_list}
    return split


def stratified_split_dataset(xbd_path: FilePath, data_path: FilePath,
                             split_props: dict, disasters_of_interest: tuple) -> FilePath:
    """Splits the xBD dataset with the corresponding proportion from `split_props`,
        only including tiles from disasters from the disasters_of interest tuple.

Args:
    xbd_path (FilePath): Ruta al directorio que contiene el conjunto de datos xBD completo.
    data_path (FilePath): Ruta al directorio donde se guardará el archivo JSON con los
                            datos divididos.
    split_props (dict): Diccionario que contiene los nombres de los subconjuntos como claves
                        (por ejemplo, 'train', 'val', 'test') y sus proporciones como valores.
                        Las proporciones deben sumar 1.
    disasters_of_interest (tuple): Tupla de strings que contiene los IDs o nombres de los
                                    desastres que deben ser incluidos en la división del
                                    conjunto de datos.

Returns:
    FilePath: Ruta al archivo JSON generado con la división de los datos.
"""
    disasters = RawPathManager.load_paths(xbd_path, disasters_of_interest)
    splits_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for dis_id, tiles_dict in disasters.items():
        log.info(f"Splitting {dis_id}")
        splits_from_disaster = split_disaster(tiles_dict, split_props)
        for split_name, tiles in splits_from_disaster.items():
            splits_dict[split_name][dis_id] = tiles
    log_splits_count(splits_dict, split_props)
    return save_splits(splits_dict, data_path)


def save_splits(splits_dict: dict, data_path: dict):
    """Save the dictionary in out_path"""
    # creates splits folder
    split_path = data_path.join("splits")
    split_path.create_folder()
    raw_split_json_path = split_path.join("raw_splits.json")
    raw_split_json_path.save_json(splits_dict)
    return raw_split_json_path


def log_splits_count(splits_dict: dict, split_props: dict) -> None:
    """Method to log the corresponding tile count of each split and its disasters."""
    tot_len = 0
    for split_name in split_props.keys():
        msg = f"{split_name} size:\n"
        tot_len = 0
        for dis_id, set in splits_dict[split_name].items():
            msg += f"{dis_id} {len(set)} tiles\n"
            tot_len += len(set)
        msg += f"Total tiles: {tot_len}"
        log.info(msg)
