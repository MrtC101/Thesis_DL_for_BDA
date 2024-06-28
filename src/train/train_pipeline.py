# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""Train Pipeline

This module establishes the sequence of execution of
scripts inside `preprocessing`, `train` and `test`
packages, to implement the data preprocessing,
model training and testing pipeline for this project.
"""
import os
import sys
from os.path import join
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm

# Environment variables
"""
os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = join(os.environ["PROJ_PATH"], "src")
os.environ["DATA_PATH"] = join(os.environ["PROJ_PATH"], "data")
os.environ["OUT_PATH"] = join(os.environ["PROJ_PATH"], "out")
"""

# Append path for project packages
if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from train.training import train_model
from utils.datasets.train_dataset import SplitDataset
from utils.common.logger import LoggerSingleton

def train_definitive(output_path, configs : dict[str]):
    """Train the model with the hole training dataset"""

    log = LoggerSingleton("DEFINITIVE MODEL", folder_path=join(output_path, "console_logs"))

    #Load Dataset    
    xBD_train = SplitDataset('train', configs['split_json_path'], configs['statistics_json_path'])
    log.info('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))    
    xBD_test = SplitDataset('test', configs['split_json_path'], configs['statistics_json_path'])
    log.info('xBD_disaster_dataset test length: {}'.format(len(xBD_test)))
    
    # Load dataloaders
    train_loader = DataLoader(xBD_train,
                            batch_size=configs['batch_size'],
                            shuffle=True,
                            num_workers=configs['batch_workers'],
                            pin_memory=False
                            )
    val_loader = DataLoader(xBD_train,
                            batch_size=configs['batch_size'],
                            shuffle=True,
                            num_workers=configs['batch_workers'],
                            pin_memory=False)
    test_loader = DataLoader(xBD_test,
                            batch_size=configs['batch_size'],
                            num_workers=configs['batch_workers'],
                            pin_memory=False)
    
    score = train_model(train_loader,val_loader,output_path,configs,test_loader=test_loader)

    return score

def k_cross_validation(k,configs : dict[str]):
    """ Do a k-fold cross validation training of the model.
        Args:
            configs (dict): All the configuration parameters:
                'init_learning_rate': Initial learning rate.
                'epochs': Number of epochs.
                'batch_size': Batch size for data loading.
                'weights_seg': List of weights for segmentation classes.
                'weights_damage': List of weights for damage classes.
                'weights_loss': List of weights for different loss components.
                'device': Device to use ('cpu' or 'cuda').
                'torch_thread' : Threads numbers for torch backend
                'torch_op_threads' : Thread number for torch operations.
                'batch_workers' : Workers number for data loaders.
                'labels_dmg': List of damage class labels.
                'labels_bld': List of building class labels.
                'num_chips_to_viz': Number of chips to visualize.
                'disaster_num': Number of disasters to leave on dataset
                'border_width': Border width for new mask creation.
                'exp_folder_path': path for the current out folder,
                'split_json_path': Path to the JSON file with splits of patches.
                'statistics_json_path': Path to the JSON file with mean and standard deviation.
                'label_map_json': Path to the JSON file with label mappings.
                'starting_checkpoint_path': Path to the checkpoint to resume from.
                'configuration_num': Number of the set of parameters for this experiment.
                'new_optimizer' : Boolean that indicates if the optimizer saved in 
                checkpoint is ignored
    """
    # FIRST declaration of the logger used throw pipeline
    log = LoggerSingleton("Training Pipeline", 
                         folder_path=os.path.join(configs['exp_folder_path'], "console_logs"))
    
    # torch hardware configurations
    log.info(f'Using PyTorch version {torch.__version__}.')
    torch.set_num_threads(configs['torch_threads'])
    log.info(f" Number of threads for TorchScripts: {torch.get_num_threads()}")
    torch.set_num_interop_threads(configs['torch_op_threads'])
    log.info(f"Number of threads for PyTorch internal operation: {torch.get_num_interop_threads()}")

    # Define the K-fold Cross Validator
    KF = KFold(n_splits=k, shuffle=True)
    xBD_train = SplitDataset('train', configs['split_json_path'], configs['statistics_json_path'])
    log.info('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))    
    
    scores = []
    # K-fold Cross Validation model evaluation
    for fold, (train_idx, val_idx) in tqdm(enumerate(KF.split(xBD_train)),total=k,desc="Fold"):
        log.info(f"{fold} iteration {k} Cross Validation")
        CV_folder_path = os.path.join(configs['exp_folder_path'],
                                      f"{k}-fold_{fold}")
        f"exp_{configs['configuration_num']}_{k}-fold_{fold}"
        # Load datasets
        train_loader = DataLoader(xBD_train,
                                batch_size=configs['batch_size'],
                                sampler=SubsetRandomSampler(train_idx),
                                num_workers=configs['batch_workers'],
                                pin_memory=False
                                )
        val_loader = DataLoader(xBD_train,
                                batch_size=configs['batch_size'],
                                sampler=SubsetRandomSampler(val_idx),
                                num_workers=configs['batch_workers'],
                                pin_memory=False)
        
        score = train_model(train_loader,val_loader,CV_folder_path,configs)
        scores.append(score)

    mean_acc_score = (sum(scores) / k) 
    log.info(f"Score mean {mean_acc_score}")
    return mean_acc_score