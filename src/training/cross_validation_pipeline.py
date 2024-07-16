
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
from training.train_pipeline import set_threads, start_train
from utils.common.pathManager import FilePath
from utils.datasets.train_dataset import TrainDataset
from utils.loggers.console_logger import LoggerSingleton


def k_cross_validation(k: int, configs: dict[str, any],
                       paths: dict[str, any]) -> float:
    """
    Perform k-fold cross-validation training of the model.

    Args:
        k (int): Number of folds for cross-validation.
        configs (dict): Configuration parameters including
        paths (dict): Paths used in the process, including:
            - 'out_dir': Path for the output directory
            - other
    Returns:
        float: Mean accuracy score over all folds.
    """
    # Logger
    out_dir = FilePath(paths['out_dir'])
    exp_i = configs['configuration_num']
    log = LoggerSingleton(f"CONFIG-{exp_i}", folder_path=out_dir)
    set_threads(configs['torch_threads'], configs['torch_op_threads'])

    # Load Datasets
    xBD_train = TrainDataset('train', paths['split_json'], paths['mean_json'])
    log.info(f'xBD_disaster_dataset train length: {len(xBD_train)}')

    scores = []
    # K-fold Cross Validation model evaluation
    KF = KFold(n_splits=k, shuffle=True)
    for fold, (train_idx, val_idx) in tqdm(enumerate(KF.split(xBD_train)),
                                           total=k, desc="Fold"):
        log.info(f"{fold} iteration {k} Cross Validation")
        paths['out_dir'] = out_dir.join(f"{k}-fold_{fold}")
        score = start_train(configs, paths, xBD_train,
                            train_sampler=SubsetRandomSampler(train_idx),
                            val_sampler=SubsetRandomSampler(val_idx))
        scores.append(score)

    mean_acc_score = (sum(scores) / k)
    log.info(f"Score mean {mean_acc_score}")
    return mean_acc_score
