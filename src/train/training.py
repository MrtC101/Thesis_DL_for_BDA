# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("training")

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from time import localtime, strftime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.siames.end_to_end_Siam_UNet import SiamUnet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.metrics.common import AverageMeter
from utils.common.files import read_json, dump_json
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer
from utils.metrics.train_metrics import compute_metrics
        
def logging_wrapper(logger,phase):
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = args[6]
            epochs = args[7]
            epoch = args[8]

            if(phase == "train"):
                logger.add_scalar( 'learning_rate', optimizer.param_groups[0]["lr"], epoch)
                
            l.info(f'Model training for epoch {epoch}/{epochs}')        
            start_time = datetime.now()
            
            result = func(*args,**kwargs)
            
            duration = datetime.now() - start_time
            logger.add_scalar(f'time_{phase}', duration.total_seconds(), epoch)

            return result
        return wrapper
    return decorator
     
def bucle(logger ,phase, loader, device, model, epoch, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer = None):
    losses = AverageMeter()
    loss_seg_pre = AverageMeter()
    loss_seg_post = AverageMeter()
    loss_dmg = AverageMeter()

    for batch_idx, data in enumerate(tqdm(loader)):                         
        x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
        x_post = data['post_image'].to(device=device)  
        y_seg = data['building_mask'].to(device=device)  
        y_cls = data['damage_mask'].to(device=device)

        if(phase=="train"):
            model.train()
            optimizer.zero_grad()
        elif(phase=="val"):
            model.eval()  # put model to evaluation mode

        scores = model(x_pre, x_post)

        # modify damage prediction based on UNet arm
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        for c in range(0,scores[2].shape[1]):
            scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)

        loss = weights_loss[0]*criterion_seg_1(scores[0], y_seg) + weights_loss[1]*criterion_seg_2(scores[1], y_seg) + weights_loss[2]*criterion_damage(scores[2], y_cls)
        loss_seg_pre = criterion_seg_1(scores[0], y_seg)
        loss_seg_post = criterion_seg_2(scores[1], y_seg)
        loss_dmg = criterion_damage(scores[2], y_cls)

        losses.update(loss.item(), x_pre.size(0))
        loss_seg_pre.update(loss_seg_pre.item(), x_pre.size(0))
        loss_seg_post.update(loss_seg_post.item(), x_pre.size(0))
        loss_dmg.update(loss_dmg.item(), x_pre.size(0))

        if(phase=="train"):
            loss.backward()  # compute gradients
            optimizer.step()

        # compute predictions & confusion metrics
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
        preds_cls = torch.argmax(softmax(scores[2]), dim=1)

        confusion_mtrx_df_dmg = compute_confusion_mtrx([], epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
        confusion_mtrx_df_bld = compute_confusion_mtrx([], epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])
    
    logger.add_scalars(f'loss_{phase}', {'_total':losses_tr.avg, '_seg_pre': loss_seg_pre.avg, '_seg_post': loss_seg_post.avg, '_dmg': loss_dmg.avg}, epoch)

    if(phase == "train"):
        prepare_for_vis(sample_train_ids, logger_train, model, 'train', epoch, device, softmax)
        prepare_for_vis(sample_val_ids, logger_val, model, 'val', epoch, device, softmax)

    return confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses

@logging_wrapper(logger_train,"training")
def train(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, logger_train, logger_val, sample_train_ids, sample_val_ids, device):
    """
    Train the model on dataset of the loader
    """
    confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses = bucle(logger_train,"train",loader, device, model, epoch, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer)
    # logger image viz        
    step_tr += 1
    return model, optimizer, step_tr, confusion_mtrx_df_dmg, confusion_mtrx_df_bld

@logging_wrapper(logger_val,"validation")
def validation(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epochs, epoch, logger_val, device):
    with torch.no_grad():
        confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses = bucle(logger_val,"val",loader, device, model, epoch, criterion_seg_1, criterion_seg_2, criterion_damage)
    return confusion_mtrx_df_dmg, confusion_mtrx_df_bld, losses.avg

def resume_model(model,starting_checkpoint_path):
    if starting_checkpoint_path and os.path.isfile(starting_checkpoint_path):
        l.info('Loading checkpoint from {}'.format(starting_checkpoint_path))
        optimizer, starting_epoch, best_acc = model.resume_from_checkpoint()
        l.info(f'Loaded checkpoint, starting epoch is {starting_epoch}, best f1 is {best_acc}')
    else:
        l.info('No valid checkpoint is provided. Start to train from scratch...')
        optimizer, starting_epoch, best_acc = model.resume_from_scratch()
    return optimizer, starting_epoch, best_acc

def output_directories(out_dir,exp_name):
    # set up directories (TrainPathManager?)
    exp_dir = os.path.join(out_dir,exp_name)  
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    
    evals_dir = os.path.join(exp_dir, 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    return checkpoint_dir, logger_dir, evals_dir, config_dir


def train_model(train_config,path_config):

    # setup output directories
    checkpoint_dir,logger_dir,evals_dir,config_dir = output_directories(path_config['out_dir'])
    dump_json(os.path.join(config_dir,'train_config.txt') , train_config)
    dump_json(os.path.join(config_dir,'path_config.txt') , path_config)
    
    # initialize logger instances
    global logger_train,logger_val, logger_test
    logger_train = SummaryWriter(log_dir=logger_dir)
    logger_val = SummaryWriter(log_dir=logger_dir)
    logger_test= SummaryWriter(log_dir=logger_dir)

    # Visualize data
    #global viz, labels_set_dmg, labels_set_bld
    label_map = read_json(path_config['label_map_json'])
    viz = RasterLabelVisualizer(label_map=label_map)

    # torch device
    l.info(f'Using PyTorch version {torch.__version__}.')
    device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
    l.info(f'Using device: {device}.')

    #data
    ## Load datasets
    dataset = ShardDataset()
    xBD_train = dataset.load_dataset("train")
    xBD_val = dataset.load_dataset("val")

    train_loader = DataLoader(xBD_train, batch_size=train_config['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(xBD_val, batch_size=train_config['batch_size'], shuffle=False, num_workers=8, pin_memory=False)

    ## Labels
    labels_set_dmg = train_config['labels_dmg']
    labels_set_bld = train_config['labels_bld']
    
    l.info('Log image samples')
    l.info('Get sample chips from train set...')
    sample_train_ids = xBD_train.get_sample_images(which_set='train')    
    l.info('Get sample chips from val set...')
    sample_val_ids = xBD_val.get_sample_images(which_set='val')

    # Training config setup

    ## define model
    model = SiamUnet().to(device=device)
    l.info(model.model_summary())

    ## resume from a checkpoint if provided
    starting_checkpoint_path = path_config['starting_checkpoint_path']
    optimizer, starting_epoch, best_acc = resume_model(model,starting_checkpoint_path)

    ## define loss functions and weights on classes
    global weights_loss, mode
    mode = train_config['mode']
    weights_seg_tf = torch.FloatTensor(train_config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(train_config['weights_damage'])
    weights_loss = train_config['weights_loss']

    ## loss functions    
    criterion_seg_1 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2000, verbose=True)
    
    ## epochs
    step_tr = 1
    epoch = starting_epoch
    epochs = train_config['epochs']

    # Metrics 
    cols = ['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy']
    eval_results_tr_dmg = pd.DataFrame(columns=cols)
    eval_results_tr_bld = pd.DataFrame(columns=cols)
    eval_results_val_dmg = pd.DataFrame(columns=cols)
    eval_results_val_bld = pd.DataFrame(columns=cols)

    while (epoch <= epochs):
        # train phase
        model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld = train(train_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, logger_train, logger_val, sample_train_ids, sample_val_ids, device)
        l.info(f'Compute actual metrics for model evaluation based on training set ...')        
        eval_results_tr_dmg, eval_results_tr_bld = compute_metrics("train",logger_train,eval_results_tr_dmg, eval_results_tr_bld,model, epoch, labels_set_dmg, labels_set_bld, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld)

        # val phase
        confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val = validation(val_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epochs, epoch, logger_val)
        scheduler.step(losses_val) # decay Learning Rate
        l.info(f'Compute actual metrics for model evaluation based on validation set ...')
        eval_results_val_dmg, eval_results_val_bld, val_acc_avg, is_best = compute_metrics("val",logger_val,eval_results_val_dmg, eval_results_val_bld,epoch, labels_set_dmg, labels_set_bld,confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld)
        
        l.info(f'Saved checkpoint for epoch {epoch}. Is it the highest f1 checkpoint so far: {is_best}\n')
        model.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1_avg': val_acc_avg,
            'best_f1': best_acc
            }, is_best, checkpoint_dir)  
              
        epoch += 1

    logger_train.flush()
    logger_train.close()
    logger_val.flush()
    logger_val.close()
    l.info('Done')
    model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld = train(train_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, logger_train, logger_val, sample_train_ids, sample_val_ids, device)
        

if __name__ == "__main__":
    train_config = {
        'labels_dmg': [0, 1, 2, 3, 4],
        'labels_bld': [0, 1],
        'weights_seg': [1, 15],
        'weights_damage': [1, 35, 70, 150, 120],
        'weights_loss': [0, 0, 1],
        'mode': 'dmg',
        'init_learning_rate': 0.0005,#dmg: 0.005, #UNet: 0.01,           
        'device': 'cpu',
        'epochs': 1500,
        'batch_size': 32,
        'num_chips_to_viz': 1
    }
    path_config = {
        'experiment_name': 'train_UNet', #train_dmg
        'out_dir': '/home/mrtc101/Desktop/tesina/repo/my_siames/out',
        'data_dir_shards': '/original_siames/public_datasets/xBD/xBD_sliced_augmented_20_alldisasters_final_mdl_npy/',
        'disaster_splits_json': '/original_siames/constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json',
        'disaster_mean_stddev': '/original_siames/constants/splits/all_disaster_mean_stddev_tiles_0_1.json',
        'label_map_json': '/original_siames/constants/class_lists/xBD_label_map.json',
        'starting_checkpoint_path': '/original_siames/nlrc_outputs/UNet_all_data_dmg/checkpoints/checkpoint_epoch120_2021-06-30-10-28-49.pth.tar',
        'shard_no': 0
    }
    train_model(train_config,path_config)