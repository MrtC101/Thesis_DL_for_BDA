
def test(loader, model, epoch):
    
    """
    Evaluate the model on test dataset of the loader
    """
    softmax = torch.nn.Softmax(dim=1)
    model.eval()  # put model to evaluation mode

    confusion_mtrx_list_test_dmg = [] 
    confusion_mtrx_list_test_bld = [] 

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            c = data['pre_image'].size()[0]
            h = data['pre_image'].size()[1]
            w = data['pre_image'].size()[2]

            x_pre = data['pre_image'].reshape(1, c, h, w).to(device=device)
            x_post = data['post_image'].reshape(1, c, h, w).to(device=device)

            y_seg = data['building_mask'].to(device=device)  
            y_cls = data['damage_mask'].to(device=device)  

            scores = model(x_pre, x_post)
                    
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
            
            for c in range(0,scores[2].shape[1]):
                scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
             
            # compute comprehensive comfusion metrics
            confusion_mtrx_df_test_dmg = compute_confusion_mtrx(confusion_mtrx_list_test_dmg, epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_test_bld = compute_confusion_mtrx(confusion_mtrx_list_test_bld, epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])

    return confusion_mtrx_df_test_dmg, confusion_mtrx_df_test_bld

def train_model(train_config,path_config):

    # setup output directories
    checkpoint_dir,logger_dir,evals_dir,config_dir = output_directories(path_config['out_dir'])
    dump_json(os.path.join(config_dir,'train_config.txt') , train_config)
    dump_json(os.path.join(config_dir,'path_config.txt') , path_config)
    
    # initialize logger instances
    global logger_test
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
    xBD_test = load_dataset()
    train_loader = DataLoader(xBD_test, batch_size=train_config['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    
    ## Labels
    labels_set_dmg = train_config['labels_dmg']
    labels_set_bld = train_config['labels_bld']
    
    l.info('Log image samples')
    l.info('Get sample chips from test set...')
    sample_test_ids = get_sample_images(which_set='train')    
   
    # Training config setup

    ## define model
    model = SiamUnet().to(device=device)
    l.info(model.model_summary())

    ## resume from a checkpoint if provided
    starting_checkpoint_path = path_config['starting_checkpoint_path']
    resume_model(model,starting_checkpoint_path)

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
        compute_metrics_train()

        # val phase
        confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val = validation(val_loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epochs, epoch, logger_val)
        scheduler.step(losses_val) # decay Learning Rate
        l.info(f'Compute actual metrics for model evaluation based on validation set ...')
        compute_metrics_eval()
        
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