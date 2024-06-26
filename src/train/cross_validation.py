from tqdm import trange
from utils.common.logger import LoggerSingleton


def k_cross_validation(k,model_config, path_config):
    log = LoggerSingleton()
    #dataset_path,statistics_path = preprocess(**pre_config)


    dataset_path = "/home/mrtc101/Desktop/tesina/repo/base_siames/data/xBD/splits/sliced_splits.json"
    statistics_path = "/home/mrtc101/Desktop/tesina/repo/base_siames/data/xBD/dataset_statistics/all_tiles_mean_stddev.json"
    for i in trange(0,k):
        log.info(f"{i} iteration {k} Cross Validation")
        path_config["dataset_path"] = dataset_path
        path_config["statistics_path"] = statistics_path
        log_Title("training and validating model")
        train_model(model_config, path_config)

    log_Title("model over test")
    dataset_path,statistics_path = preprocess(**pre_config)
    test_model(model_config, path_config)
    return 