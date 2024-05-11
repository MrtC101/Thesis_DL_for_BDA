# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
if(os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.logger import get_logger
l = get_logger("train")


import os
import sys
import math
import torch
import shutil
import logging
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from torchvision import transforms
from time import localtime, strftime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#print(os.path.join(os.getcwd(),"data"))
sys.path.append("./")
from data.raster_label_visualizer import RasterLabelVisualizer
from models.end_to_end_Siam_UNet import SiamUnet
from utils.train_utils import AverageMeter
from utils.train_utils import load_json_files, dump_json_files
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset_shard_load import DisasterDataset

config = {'labels_dmg': [0, 1, 2, 3, 4],
          'labels_bld': [0, 1],
          'weights_seg': [1, 15],
          'weights_damage': [1, 35, 70, 150, 120],
          'weights_loss': [0, 0, 1],
          'mode': 'dmg',
          'init_learning_rate': 0.0005,#dmg: 0.005, #UNet: 0.01,           
          'device': 'cpu',
          'epochs': 1500,
          'batch_size': 32,
          'num_chips_to_viz': 1,
          'experiment_name': 'train_UNet', #train_dmg
          'out_dir': '/original_siames/nlrc_outputs/',
          'data_dir_shards': '/original_siames/public_datasets/xBD/xBD_sliced_augmented_20_alldisasters_final_mdl_npy/',
          'shard_no': 0,
          'disaster_splits_json': '/original_siames/constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json',
          'disaster_mean_stddev': '/original_siames/constants/splits/all_disaster_mean_stddev_tiles_0_1.json',
          'label_map_json': '/original_siames/constants/class_lists/xBD_label_map.json',
          'starting_checkpoint_path': '/original_siames/nlrc_outputs/UNet_all_data_dmg/checkpoints/checkpoint_epoch120_2021-06-30-10-28-49.pth.tar'}

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='{asctime} {levelname} {message}',
                    style='{',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Using PyTorch version {torch.__version__}.')
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}.')

def main():

    global viz, labels_set_dmg, labels_set_bld
    global xBD_train, xBD_val
    global train_loader, val_loader, test_loader
    global weights_loss, mode

    xBD_train, xBD_val = load_dataset()

    train_loader = DataLoader(xBD_train, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(xBD_val, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=False)

    label_map = load_json_files(config['label_map_json'])
    viz = RasterLabelVisualizer(label_map=label_map)

    labels_set_dmg = config['labels_dmg']
    labels_set_bld = config['labels_bld']
    mode = config['mode']

    eval_results_tr_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_tr_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])

    # set up directories    
    checkpoint_dir = os.path.join(config['out_dir'], config['experiment_name'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger_dir = os.path.join(config['out_dir'], config['experiment_name'], 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    
    evals_dir = os.path.join(config['out_dir'], config['experiment_name'], 'evals')
    os.makedirs(evals_dir, exist_ok=True)

    config_dir = os.path.join(config['out_dir'], config['experiment_name'], 'configs')
    os.makedirs(config_dir, exist_ok=True)
    dump_json_files(os.path.join(config_dir,'config.txt') , config)


    # define model
    model = SiamUnet().to(device=device)
    model_summary(model)

    
    # resume from a checkpoint if provided
    starting_checkpoint_path = config['starting_checkpoint_path']
    if starting_checkpoint_path and os.path.isfile(starting_checkpoint_path):
        logging.info('Loading checkpoint from {}'.format(starting_checkpoint_path))
        checkpoint = torch.load(starting_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        #don't load the optimizer settings so that a newly specified lr can take effect
        if mode == 'dmg':
            print_network(model)
            model = freeze_model_param(model)
            print_network(model)

            # monitor model
            logger_model = SummaryWriter(log_dir=logger_dir)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger_model.add_histogram(tag, value.data.cpu().numpy(), global_step=0)
            
            reinitialize_Siamese(model)
            
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger_model.add_histogram(tag, value.data.cpu().numpy(), global_step=1)

            logger_model.flush()
            logger_model.close()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['init_learning_rate'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['init_learning_rate'])

        starting_epoch = checkpoint['epoch'] + 1  # we did not increment epoch before saving it, so can just start here
        best_acc = checkpoint.get('best_f1', 0.0)
        logging.info(f'Loaded checkpoint, starting epoch is {starting_epoch}, '
                     f'best f1 is {best_acc}')
    else:
        logging.info('No valid checkpoint is provided. Start to train from scratch...')
        optimizer = torch.optim.Adam(model.parameters(), lr=config['init_learning_rate'])
        starting_epoch = 1
        best_acc = 0.0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2000, verbose=True)

    # define loss functions and weights on classes
    weights_seg_tf = torch.FloatTensor(config['weights_seg'])
    weights_damage_tf = torch.FloatTensor(config['weights_damage'])
    weights_loss = config['weights_loss']
    
    criterion_seg_1 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)

    # initialize logger instances
    logger_train = SummaryWriter(log_dir=logger_dir)
    logger_val = SummaryWriter(log_dir=logger_dir)
    logger_test= SummaryWriter(log_dir=logger_dir)

    
    logging.info('Log image samples')
    logging.info('Get sample chips from train set...')
    sample_train_ids = get_sample_images(which_set='train')    
    logging.info('Get sample chips from val set...')
    sample_val_ids = get_sample_images(which_set='val')

    epoch = starting_epoch
    step_tr = 1
    epochs = config['epochs']

    while (epoch <= epochs):

        ###### train
        logger_train.add_scalar( 'learning_rate', optimizer.param_groups[0]["lr"], epoch)
        logging.info(f'Model training for epoch {epoch}/{epochs}')
        train_start_time = datetime.now()
        model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld = train(train_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, logger_train, logger_val, sample_train_ids, sample_val_ids, device)
        train_duration = datetime.now() - train_start_time
        logger_train.add_scalar('time_training', train_duration.total_seconds(), epoch)

        logging.info(f'Compute actual metrics for model evaluation based on training set ...')

        # damage level eval train
        eval_results_tr_dmg = compute_eval_metrics(epoch, labels_set_dmg, confusion_mtrx_df_tr_dmg, eval_results_tr_dmg)
        eval_results_tr_dmg_epoch = eval_results_tr_dmg.loc[eval_results_tr_dmg['epoch'] == epoch,:]
        f1_harmonic_mean = 0
        for metrics in ['f1']:
            for index, row in eval_results_tr_dmg_epoch.iterrows():
                if int(row['class']) in labels_set_dmg[1:]:
                    logger_train.add_scalar( 'tr_dmg_class_' + str(int(row['class'])) + '_' + str(metrics), row[metrics], epoch)
                    if metrics == 'f1':
                        f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
        f1_harmonic_mean = 4.0/f1_harmonic_mean
        logger_train.add_scalar( 'tr_dmg_harmonic_mean_f1', f1_harmonic_mean, epoch)

        
        # bld level eval train
        eval_results_tr_bld = compute_eval_metrics(epoch, labels_set_bld, confusion_mtrx_df_tr_bld, eval_results_tr_bld)
        eval_results_tr_bld_epoch = eval_results_tr_bld.loc[eval_results_tr_bld['epoch'] == epoch,:]
        for metrics in ['f1']:
            for index, row in eval_results_tr_bld_epoch.iterrows():
                if int(row['class']) in labels_set_dmg[1:]:
                    logger_train.add_scalar( 'tr_bld_class_' + str(int(row['class'])) + '_' + str(metrics), row[metrics], epoch)

        
        ####### validation
        logging.info(f'Model validation for epoch {epoch}/{epochs}')
        eval_start_time = datetime.now()
        confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val = validation(val_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epochs, epoch, logger_val)
        eval_duration = datetime.now() - eval_start_time
        # decay Learning Rate
        scheduler.step(losses_val)
        logger_val.add_scalar('time_validation', eval_duration.total_seconds(), epoch)
        logging.info(f'Compute actual metrics for model evaluation based on validation set ...')

        # damage level eval validation
        eval_results_val_dmg = compute_eval_metrics(epoch, labels_set_dmg, confusion_mtrx_df_val_dmg, eval_results_val_dmg)
        eval_results_val_dmg_epoch = eval_results_val_dmg.loc[eval_results_val_dmg['epoch'] == epoch,:]
        f1_harmonic_mean = 0
        for metrics in ['f1']:
            for index, row in eval_results_val_dmg_epoch.iterrows():
                if int(row['class']) in labels_set_dmg[1:]:
                    logger_val.add_scalar( 'val_dmg_class_' + str(int(row['class'])) + '_' + str(metrics), row[metrics], epoch)
                    if metrics == 'f1':
                        f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
        f1_harmonic_mean = 4.0/f1_harmonic_mean
        logger_val.add_scalar( 'val_dmg_harmonic_mean_f1', f1_harmonic_mean, epoch)

        # bld level eval validation
        eval_results_val_bld = compute_eval_metrics(epoch, labels_set_bld, confusion_mtrx_df_val_bld, eval_results_val_bld)
        eval_results_val_bld_epoch = eval_results_val_bld.loc[eval_results_val_bld['epoch'] == epoch,:]
        for metrics in ['f1']:
            for index, row in eval_results_val_bld_epoch.iterrows():
                if int(row['class']) in labels_set_bld[1:]:
                    logger_val.add_scalar( 'val_bld_class_' + str(int(row['class'])) + '_' + str(metrics), row[metrics], epoch)
                

        ####### compute average accuracy across all classes to select the best model
        val_acc_avg = f1_harmonic_mean
        is_best = val_acc_avg > best_acc
        best_acc = max(val_acc_avg, best_acc)
        
        logging.info(f'Saved checkpoint for epoch {epoch}. Is it the highest f1 checkpoint so far: {is_best}\n')
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1_avg': val_acc_avg,
            'best_f1': best_acc}, is_best, checkpoint_dir)        
    
        # log execution time for this epoch
        logging.info((f'epoch training duration is {train_duration.total_seconds()} seconds;'
                     f'epoch evaluation duration is {eval_duration.total_seconds()} seconds'))

        epoch += 1

    logger_train.flush()
    logger_train.close()
    logger_val.flush()
    logger_val.close()

    
    logging.info('Done')

    return

def train(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, logger_train, logger_val, sample_train_ids, sample_val_ids, device):
    """
    Train the model on dataset of the loader
    """
    confusion_mtrx_list_tr_dmg = []
    confusion_mtrx_list_tr_bld = []

    losses_tr = AverageMeter()
    loss_seg_pre = AverageMeter()
    loss_seg_post = AverageMeter()
    loss_dmg = AverageMeter()

    for batch_idx, data in enumerate(tqdm(loader)): 
                         
        x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
        x_post = data['post_image'].to(device=device)  
        y_seg = data['building_mask'].to(device=device)  
        y_cls = data['damage_mask'].to(device=device)  
        
        model.train()
        optimizer.zero_grad()
        scores = model(x_pre, x_post)
        
        # modify damage prediction based on UNet arm
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        for c in range(0,scores[2].shape[1]):
            scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
        
        loss = weights_loss[0]*criterion_seg_1(scores[0], y_seg) + weights_loss[1]*criterion_seg_2(scores[1], y_seg) + weights_loss[2]*criterion_damage(scores[2], y_cls)
        loss_seg_pre_tr = criterion_seg_1(scores[0], y_seg)
        loss_seg_post_tr = criterion_seg_2(scores[1], y_seg)
        loss_dmg_tr = criterion_damage(scores[2], y_cls)
        
        losses_tr.update(loss.item(), x_pre.size(0))
        loss_seg_pre.update(loss_seg_pre_tr.item(), x_pre.size(0))
        loss_seg_post.update(loss_seg_post_tr.item(), x_pre.size(0))
        loss_dmg.update(loss_dmg_tr.item(), x_pre.size(0))

        loss.backward()  # compute gradients
        optimizer.step()
        
        # compute predictions & confusion metrics
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
        preds_cls = torch.argmax(softmax(scores[2]), dim=1)

        confusion_mtrx_df_tr_dmg = compute_confusion_mtrx(confusion_mtrx_list_tr_dmg, epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
        confusion_mtrx_df_tr_bld = compute_confusion_mtrx(confusion_mtrx_list_tr_bld, epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])


    # logging image viz        
    prepare_for_vis(sample_train_ids, logger_train, model, 'train', epoch, device, softmax)
    prepare_for_vis(sample_val_ids, logger_val, model, 'val', epoch, device, softmax)
    step_tr += 1

    logger_train.add_scalars('loss_tr', {'_total':losses_tr.avg, '_seg_pre': loss_seg_pre.avg, '_seg_post': loss_seg_post.avg, '_dmg': loss_dmg.avg}, epoch)

    return model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld

def validation(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epochs, epoch, logger_val):
    
    """
    Evaluate the model on dataset of the loader
    """
    confusion_mtrx_list_val_dmg = []
    confusion_mtrx_list_val_bld = []
    losses_val = AverageMeter()
    loss_seg_pre = AverageMeter()
    loss_seg_post = AverageMeter()
    loss_dmg = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
            x_post = data['post_image'].to(device=device)  
            y_seg = data['building_mask'].to(device=device)  
            y_cls = data['damage_mask'].to(device=device)  

            model.eval()  # put model to evaluation mode
            scores = model(x_pre, x_post)

            # modify damage prediction based on UNet arm
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            for c in range(0,scores[2].shape[1]):
                scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            
            loss = weights_loss[0]*criterion_seg_1(scores[0], y_seg) + weights_loss[1]*criterion_seg_2(scores[1], y_seg) + weights_loss[2]*criterion_damage(scores[2], y_cls)
            loss_seg_pre_val = criterion_seg_1(scores[0], y_seg)
            loss_seg_post_val = criterion_seg_2(scores[1], y_seg)
            loss_dmg_val = criterion_damage(scores[2], y_cls)
        
            losses_val.update(loss.item(), x_pre.size(0))
            loss_seg_pre.update(loss_seg_pre_val.item(), x_pre.size(0))
            loss_seg_post.update(loss_seg_post_val.item(), x_pre.size(0))
            loss_dmg.update(loss_dmg_val.item(), x_pre.size(0))

            # compute predictions & confusion metrics
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
            
            confusion_mtrx_df_val_dmg = compute_confusion_mtrx(confusion_mtrx_list_val_dmg, epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_val_bld = compute_confusion_mtrx(confusion_mtrx_list_val_bld, epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])

    logger_val.add_scalars('loss_val', {'_total': losses_val.avg, '_seg_pre': loss_seg_pre.avg, '_seg_post': loss_seg_post.avg, '_dmg': loss_dmg.avg}, epoch)

    return confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val.avg

if __name__ == "__main__":
    main()