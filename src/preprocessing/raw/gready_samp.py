# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from collections import defaultdict
import math
from tqdm import trange, tqdm
from utils.common.pathManager import FilePath
import pandas as pd
import numpy as np

from utils.loggers.console_logger import LoggerSingleton
from utils.pathManagers.rawManager import RawPathManager

log = LoggerSingleton()


def normalize_array(x):
    """Normalize a numpy array to the range [0, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)


def find_best_sample(non_empty_df: pd.DataFrame, dmg_size: int, labels: list,
                     balance_imp: float) -> pd.DataFrame:
    """
    Iteratively explores all possible slices of contiguous rows of length `dmg_size` from a
    DataFrame sorted by imbalance.
    To find the best sample, it combines two functions weighted by `balance_imp`:
    one expressing the total number of buildings in each slice and the other expressing the
    imbalance of damage classes present in `labels`.
    This aggregation allows for identifying the point where the total number of buildings is
    maximized and the class imbalance is minimized, returning a sample with a trade-off between
    total buildings and damage class imbalance.

    Args:
        non_empty_df (pd.DataFrame): The DataFrame containing rows without empty entries.
        dmg_size (int): The size of the slice of contiguous rows to explore.
        labels (list): The list of column names representing the damage labels for imbalance
        calculation.
        balance_imp (float): Importance weight of class imbalance vs total number of buildings.
            - If greater than 0.5, prioritizes solutions with lower class imbalance.
            - If equal to 0.5, balances both objectives equally.
            - If less than 0.5, prioritizes solutions with more total buildings, possibly with
            higher imbalance.

    Returns:
        pd.DataFrame: The sample from the DataFrame that best balances building count and damage
        class imbalance.
    """
    bld_tot_f = []
    imbalance_f = []
    log.info("Exploring posible data samples...")
    for i in trange(0, len(non_empty_df)-dmg_size, desc="Searching best sample"):
        sample = non_empty_df.iloc[i:i+dmg_size]
        bld_tot = sample.sum(axis=0, numeric_only=True).sum()
        arr = sample[labels].sum(axis=0, numeric_only=True).to_numpy()
        imbalance = mean_square_error(np.array([arr]))[0]
        bld_tot_f.append(bld_tot)
        imbalance_f.append(imbalance)
    log.info(f"{len(imbalance_f)} possible samples explored.")
    norm_bld_tot = normalize_array(np.array(bld_tot_f))
    norm_imbalance = normalize_array(np.array(imbalance_f))
    beta = 100 * balance_imp  # Peso para f1
    alpha = 100 - beta   # Peso para f2
    combined_func = alpha * norm_bld_tot - beta * norm_imbalance
    norm_g = normalize_array(combined_func)
    id = norm_g.argmax()
    log.info(f"Selected sample from {id} to {id + dmg_size} index.")
    return non_empty_df.iloc[id:id+dmg_size]


def best_sample_search(non_empty_df: pd.DataFrame, empty_df: pd.DataFrame, total_img: int,
                       labels: list, balance_imp: float = 0.7,
                       empty_prop:  float = 0.2) -> pd.DataFrame:
    """Finds and build the final sample that is balanced and has the number of images requested.

    Args:
        non_empty_df: sorted with greedy approach Dataframe of xBD images.
        empty_df: Dataframe of xBD images without buildings.
        total_img: total number of images for the final sample to build.
        labels: labels to take in count to balance the dataset.
        balance_imp: importance factor to trade of between balance a nd total number of buildings.
        empty_prop: proportion of the final sample that must bee images without buildings.
    Return:
        pd.Dataframe: with all the corresponding images inside the balance sample.
    """
    log.info("Searching for the best sample.")
    if total_img > len(non_empty_df):
        raise ValueError(
            "El tamaño de la muestra no puede ser mayor que la longitud del DataFrame.")

    empty_size = math.floor(total_img * empty_prop)
    dmg_size = total_img - empty_size

    dmg_sample = find_best_sample(non_empty_df, dmg_size, labels, balance_imp)

    # sampling the selected slice of instances.
    dmg_sample = dmg_sample.reset_index()
    proportions = dmg_sample.value_counts("dis_id")
    proportions = proportions / proportions.sum()
    img_num_by_dis = (empty_size * proportions).apply(lambda x: round(x))

    # proportional sample of images without damaged buildings
    empty_sample = empty_df.reset_index(drop=True)
    empty_sample = empty_df.groupby(["dis_id"]).apply(lambda x: x.sample(n=img_num_by_dis[x.name],
                                                                         replace=True),
                                                      include_groups=False)
    empty_sample: pd.DataFrame = empty_sample.reset_index()
    empty_sample.drop(columns=["level_1"], inplace=True)
    return pd.concat([dmg_sample, empty_sample], axis=0)


def mean_square_error(matrix_arr: np.ndarray) -> np.ndarray:
    """Returns the error for each row in the np.array matrix"""
    matrix_arr: np.ndarray = matrix_arr.astype(float)
    matrix_arr -= matrix_arr.mean(axis=1)[:, np.newaxis]
    matrix_arr = matrix_arr**2
    return matrix_arr.sum(axis=1) / matrix_arr.shape[1]


def get_best_candidate(candidates_df: pd.DataFrame, total_arr: np.ndarray) -> pd.DataFrame:
    """Returns the row with less MSE value"""
    cand_matrix: np.ndarray = candidates_df.to_numpy(copy=True)
    cand_matrix += total_arr
    points = mean_square_error(cand_matrix) - mean_square_error(np.array([total_arr]))
    return candidates_df.iloc[points.argmin()].name


def sort_by_imbalance(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Implementation of a greedy strategy to iteratively select the image that has the least impact
    on the imbalance of the current damage label distribution. The result is a sorted DataFrame
    where contiguous rows are arranged to minimize the imbalance ratio between all classes.
    (This property is true until certain row in the Dataframe because this method only sorts the
    dataset and do not drop any row.)

    Args:
        df (pd.DataFrame): The DataFrame with damage label information by each image.
        labels (list): List of damage label column names to be used for calculating the imbalance.

    Returns:
        pd.DataFrame: A DataFrame sorted such that the imbalance between damage labels is minimized
        across contiguous rows.
    """
    index = []
    df = df.set_index(["dis_id", "tile_id"])
    curr_sample = df[labels].copy()
    total_per_label = np.zeros(len(labels), dtype=np.int64)
    log.info("Sorting index...")
    for _ in trange(len(df), desc='Sorting index'):
        best_id = get_best_candidate(curr_sample, total_per_label)
        total_per_label += curr_sample.loc[best_id].to_numpy(copy=True)
        curr_sample.drop(index=best_id, inplace=True)
        index.append(best_id)
    return df.loc[index]


def create_bld_dmg_dataframe(xbd_path: FilePath, labels: list) -> pd.DataFrame:
    """
    This method iterates over each 'labels' folder and creates a pandas DataFrame
    containing the count of buildings by damage type for each image in the xBD dataset.

    Args:
        xbd_path: string path to the xBD dataset folder.
        labels: List of string damage labels to be count.

    Returns:
        pd.Dataframe: The number of buildings of each image by damage label.
    """
    tiles = []
    # iterates over labels directories
    log.info("Reading label files")
    for split_folder in tqdm(xbd_path.get_folder_paths(), desc="Reading label files"):
        folder_path = split_folder.join("labels")
        for json_name in folder_path.get_files_names():
            name_splits = json_name.split("_")
            dis_id, tile_id, time_prefix = name_splits[:3]
            # only count post disaster images.
            if time_prefix == "post":
                label_dict = folder_path.join(json_name).read_json()
                label_count = pd.Series(data=np.zeros(5, dtype=np.uint16), index=labels)
                for building in label_dict['features']['xy']:
                    dmg_label = building['properties'].get('subtype')
                    if dmg_label in label_count.index:
                        label_count[dmg_label] += 1

                row = [dis_id, tile_id] + label_count.tolist()
                tiles.append(row)

    dmg_by_tile_df = pd.DataFrame(tiles, columns=["dis_id", "tile_id"] + labels)
    log.info(f"{len(dmg_by_tile_df)} label json files read.")
    return dmg_by_tile_df


def split_empty_images(dmg_by_tile_df, columns):
    """Splits the DataFrame by images with buildings and empty images."""
    non_empty_idx = dmg_by_tile_df[columns].sum(axis=1) != 0
    non_empty_df = dmg_by_tile_df.loc[non_empty_idx]
    empty_df = dmg_by_tile_df.loc[~non_empty_idx]
    return non_empty_df, empty_df


def greedy_sampling(xbd_path: FilePath, img_num: int) -> FilePath:
    """Returns the corresponding balanced sample obtained with a greedy approach."""
    log.info("Balanced Sampling...")
    labels = ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"]
    dmg_by_tile_df: pd.DataFrame = create_bld_dmg_dataframe(xbd_path, labels)
    non_empty_df, empty_df = split_empty_images(dmg_by_tile_df, labels[:4])
    non_empty_df = sort_by_imbalance(non_empty_df, labels[:4])
    sample = best_sample_search(non_empty_df, empty_df, img_num, labels[:4])
    return sample


def greedy_split_dataset(xbd_path: FilePath, data_path: FilePath, img_total: int,
                         split_prop: dict) -> FilePath:
    """
    Performs a greedy selection of img_total images from the xBD dataset.
    Then splits the selected images into training and testing sets according to the specified
    proportions. Saves the split into a json inside data_path.

    Args:    sample.sum(axis=0, numeric_only=True)
        xbd_path (FilePath): Path to the xBD dataset directory.
        data_path (FilePath): Path to the directory where the split files will be saved.
        img_total (int): Total number of images to be selected for the sample.
        split_prop (dict): Dictionary with the dataset split proportions.
        e.g., {'train': 0.8, 'test': 0.2}.

    Returns:
        FilePath: Path to the JSON file containing the train/test image splits.
    """
    # loads raw dataset
    data_dict = RawPathManager.load_paths(xbd_path, ("all"))

    # Sampling
    sample = greedy_sampling(xbd_path, img_total)

    log.info(f"Building distribution: \n{sample.sum(axis=0, numeric_only=True)}")

    splits_dfs = get_split_dataframes(sample, split_prop)
    splits_dict = build_split_dict(splits_dfs, data_dict)

    # creates splits folder
    split_path = data_path.join("splits")
    split_path.create_folder()
    balanced_file = split_path.join("raw_splits.json")
    balanced_file.save_json(splits_dict)
    return balanced_file


def get_split_dataframes(sample: pd.DataFrame, props: dict) -> dict:
    """Creates a dictionary of dataframes with splits from the sampled xBD dataset
    with the corresponding name and proportion from de dictionary props."""
    ids_df = sample[["dis_id", "tile_id"]]
    ids_df = ids_df.reset_index(drop=True)

    splits: dict = {}
    tot_prop: float = 0.0
    remaining_ids_df = ids_df.copy()
    remaining_prop = 1.0
    for split, prop in list(props.items()):
        tot_prop += prop
        if (prop / remaining_prop) < 1:
            split_ids = remaining_ids_df.sample(frac=prop / remaining_prop)
            remaining_ids_df = remaining_ids_df.drop(split_ids.index)
            remaining_prop -= prop
        else:
            split_ids = remaining_ids_df
        splits[split] = split_ids

    if tot_prop != 1.0:
        raise Exception("Split proportions must sum 1")

    # log lengths
    msg = f"Total imgs {len(ids_df)}:\n"
    for split_name, split_df in list(splits.items()):
        msg += f"{split_name} length {len(split_df)} \n"
    log.info(msg)

    return splits


def build_split_dict(splits_dfs: dict, data_dict: dict) -> dict:
    """
        Iterates over the xBD dataset and builds a dictionary representation
        of all splits using the sampled splits_dfs.
    """
    split_keys_dict = {split_name: list(split_df.itertuples(index=False, name=None))
                       for split_name, split_df in splits_dfs.items()}
    # iterates over all disasters in xBD
    splits_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for dis_id, tiles_dict in data_dict.items():
        # iterates over all tiles in the disaster
        for tile_id in tiles_dict.keys():
            tup = (dis_id, tile_id)
            # iterates over all calculated splits
            for split_name, split_keys in split_keys_dict.items():
                # same tile can be over sampled in other splits
                if tup in split_keys:
                    for i in range(1, split_keys.count(tup)+1):
                        new_tile_id = tile_id
                        if i > 1:
                            # create a new id is needed
                            new_tile_id = str(i) + 'a' + tile_id
                        splits_dict[split_name][dis_id][new_tile_id] = tiles_dict[tile_id]

    return splits_dict
