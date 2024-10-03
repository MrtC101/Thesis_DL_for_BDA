
from typing import Dict, Any
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
from training.model_train.train_manager import train_model
from training.model_train.utils import TrainDataLoader, set_threads
from utils.common.pathManager import FilePath
from utils.common.timeManager import measure_time
from utils.datasets.train_dataset import TrainDataset
from utils.loggers.console_logger import LoggerSingleton


@measure_time
def k_cross_validation(k: int, config: Dict[str, Any], paths: Dict[str, Any]):
    """Perform k-fold cross-validation training with given configuration.

    Args:
        k (int): Number of folds for cross-validation.
        configs (dict): Configuration parameters including
        paths (dict): Paths used in the process, including:
            - 'out_dir': Path for the output directory
            - other
    """
    if k < 2:
        raise Exception("Folds must be greater than 1")

    # Logger
    out_dir = FilePath(paths['out_dir'])
    exp_i = config['configuration_num']
    log = LoggerSingleton(f"config-{exp_i}", folder_path=out_dir)
    config['tot_epochs'] = config['hps_epochs']

    set_threads()

    # Load Datasets
    xBD_train = TrainDataset('train', paths['split_json'], paths['mean_json'])
    log.info(f'xBD_disaster_dataset train length: {len(xBD_train)}')

    # K-fold Cross Validation model evaluation
    KF = KFold(n_splits=k, shuffle=True)
    for fold, (train_idx, val_idx) in tqdm(enumerate(KF.split(xBD_train)), total=k, desc="Fold"):
        log.info(f"{fold} iteration {k} Cross Validation")

        train_loader = TrainDataLoader(
            xBD_train,
            batch_size=config['batch_size'],
            num_workers=config['batch_workers'],
            sampler=SubsetRandomSampler(train_idx)
        )

        val_loader = TrainDataLoader(
            xBD_train,
            batch_size=config['batch_size'],
            num_workers=config['batch_workers'],
            sampler=SubsetRandomSampler(val_idx)
        )

        paths['out_dir'] = out_dir.join(f"{k}-fold_{fold}")
        train_model(config, paths, train_loader, val_loader, save_all_checkpoints=False)
        log.info(f"{fold+1} fold FINISHED.")
    log.info(f"Configuration number {exp_i} finished.")
