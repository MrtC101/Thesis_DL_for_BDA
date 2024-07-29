from joblib import Parallel, delayed
from tqdm import tqdm
from utils.common.pathManager import FilePath
from training.cross_validation_pipeline import k_cross_validation
from utils.loggers.console_logger import LoggerSingleton

def _start_k_fold(folds: int, index: int,
                  config: dict, paths: dict) -> tuple[int, float]:
    """
    Run k-fold cross-validation for a given configuration.

    Args:
        folds (int): Number of folds for cross-validation.
        index (int): Index of the current configuration.
        config (dict): Configuration dictionary.
        paths (dict): Dictionary of paths used in the process.

    Returns:
        tuple[int, float]: Index and score of the configuration.
    """
    out_dir = FilePath(paths['out_dir']).join(f'config-{index}')
    out_dir.create_folder()
    paths['out_dir'] = out_dir
    config['configuration_num'] = index
    score = k_cross_validation(folds, config, paths)
    return index, score


def parameter_search(folds: int, param_list: dict, paths_dict: dict, parallel=False) -> dict:
    """
    Perform hyperparameter search using k-fold cross-validation.

    Args:
        folds (int): Number of folds for cross-validation.
        configs (dict): Base configuration dictionary.
        paths_dict (dict): Dictionary of paths used in the process.

    Returns:
        dict: Best configuration dictionary.
    """
    log_out = FilePath(paths_dict['out_dir']).join(
        "hyperparameter_console_logs")
    log = LoggerSingleton("HYPERPARAMETER_SEARCH", log_out)
    log.info(f"Cantidad de configuraciones: {len(param_list)}.")

    if parallel:
        results = Parallel(n_jobs=-1)(delayed(_start_k_fold)
                                    (folds, i, config, paths_dict)
                                    for i, config in tqdm(param_list))
    else:
        results = [_start_k_fold(folds, i, config, paths_dict)
                for i, config in tqdm([param_list])]

    best_index, best_acc = min(results, key=lambda x: x[1])

    log.info(f"Configuration number {best_index} " +
            f"with a validation loss of {best_acc:.4f}")