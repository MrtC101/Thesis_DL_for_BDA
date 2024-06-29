import os
import sys
from os.path import join

# Environment variables
os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from train.train_pipeline import k_cross_validation, train_definitive
from postprocessing.postprocess_pipeline import postprocess
from preprocessing.preprocessing_pipeline import preprocess
from utils.common.logger import LoggerSingleton
from sklearn.model_selection import ParameterGrid


def create_params(dataset_path, mean_std_json_path, configs):
    # Define los parámetros que deseas optimizar
    hiperparameter_config = {
        'init_learning_rate': [0.0005],  # dmg: 0.005, #UNet: 0.01,
        'tot_epochs': [1] ,  # 1500,
        'batch_size': [25]
    }
    param_combinations = list(ParameterGrid(hiperparameter_config))
    config_list = []
    for i, hiper in enumerate(param_combinations):
        path_config = {
            'exp_folder_path': os.path.join(os.environ["OUT_PATH"],f'config-{i}'),
            "split_json_path": dataset_path,
            "statistics_json_path": mean_std_json_path,
            'label_map_json': join(os.environ["DATA_PATH"], "constants","xBD_label_map.json"),
            'starting_checkpoint_path': None
        }
        current_config = dict(
            list(configs.items()) +
            list(hiper.items()) + 
            list(path_config.items())
        )
        config_list.append(current_config)
    return config_list

def SequentialSearch(param_list):
    best = (0.0,-1) 
    for i, current_config in enumerate(param_list):
        current_config['configuration_num'] = i
        score = k_cross_validation(2,current_config)
        if best[1] < score:
            best = (i,score)
    best_index = best[0]
    best_params = param_list[best_index]
    log = LoggerSingleton()
    log.info(f'Configuración numero {best[0]} con una pérdida\
              de validación de {best[1]:.4f}')
    return best_params
    
from joblib import Parallel, delayed
import torch

def ParalelSearch(param_list):
    # torch hardware configurations
    log = LoggerSingleton()
    log.info(f'Using PyTorch version {torch.__version__}.')
    torch.set_num_threads(configs['torch_threads'])
    log.info(f" Number of threads for TorchScripts: {torch.get_num_threads()}")
    torch.set_num_interop_threads(configs['torch_op_threads'])
    log.info(f"Number of threads for PyTorch internal operation: {torch.get_num_interop_threads()}")

    def start_k_fold(i,current_config):
        current_config['configuration_num'] = i
        score = k_cross_validation(2,current_config)
        return (i,score)
    
    results = Parallel(n_jobs=-1)(delayed(start_k_fold)(i,current_config) \
                                  for i, current_config in enumerate(param_list))
    best_index, best_acc = min(results, key=lambda x: x[1])
    log.info(f'Configuración numero {best_index} con una pérdida\
              de validación de {best_acc:.4f}')
    best_params = param_list[best_index]
    return best_params
    
def parameter_search(configs : dict):
    log = LoggerSingleton("Parameter Serach", 
                          folder_path=join(os.environ["OUT_PATH"], "hiper_console_logs"))
    #split_raw_json_path, split_sliced_json_path, mean_std_json_path = preprocess(**pre_config)
    split_raw_json_path = "/home/mrtc101/Desktop/tesina/repo/hiper_siames/data/xBD/splits/raw_splits.json"
    split_sliced_json_path = "/home/mrtc101/Desktop/tesina/repo/hiper_siames/data/xBD/splits/sliced_splits.json"
    mean_std_json_path = "/home/mrtc101/Desktop/tesina/repo/hiper_siames/data/xBD/dataset_statistics/all_tiles_mean_stddev.json"
    param_list = create_params(split_sliced_json_path, mean_std_json_path, configs)
    
    #best_params = ParalelSearch(param_list)
    
    best_params = SequentialSearch(param_list)
    
    # Train definitive model
    pred_path = os.path.join(os.environ['OUT_PATH'],"definitive_model")
    definitive_acc_score = train_definitive(pred_path, best_params)
    log.info(f"Accuracy for the final model : {definitive_acc_score}")
    
    save_path = os.path.join(os.environ['OUT_PATH'],"predictions")
    postprocess(split_raw_json_path, pred_path, best_params['label_map_json'], save_path)
    

if __name__ == "__main__":
    # Configuration dictionary for paths used during model training
    weights_config = {
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
    }
    hardware_config ={
        'device': 'cpu',
        'torch_threads': 12,
        'torch_op_threads': 12,
        'batch_workers': 0,
        'new_optimizer' : False
    }
    visual_config = {
        'num_chips_to_viz': 2,
        'labels_dmg': [ 1, 2, 3, 4],
        'labels_bld': [1],
    }
    pre_config = {
        "disaster_num": 20,
        "border_width": 1,
    }
    post_config = {
        
    }
    configs = dict(
        list(weights_config.items()) +
        list(hardware_config.items()) +
        list(visual_config.items()) +
        list(pre_config.items())
                   )
    parameter_search(configs)