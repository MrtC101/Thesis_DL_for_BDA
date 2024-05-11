def get_sample_images(which_set='train'):

    """
    Get a deterministic set of images in the specified set (train or val) by using the dataset and
    not the dataloader. Only works if the dataset is not IterableDataset.

    Args:
        which_set: one of 'train' or 'val'

    Returns:
        samples: a dict with keys 'chip' and 'chip_label', pointing to torch Tensors of
        dims (num_chips_to_visualize, channels, height, width) and (num_chips_to_visualize, height, width)
        respectively
    """
    assert which_set == 'train' or which_set == 'val'

    dataset = xBD_train if which_set == 'train' else xBD_val

    num_to_skip = 1  # first few chips might be mostly blank
    assert len(dataset) > num_to_skip + config['num_chips_to_viz']

    keep_every = math.floor((len(dataset) - num_to_skip) / config['num_chips_to_viz'])
    samples_idx_list = []

    for sample_idx in range(num_to_skip, len(dataset), keep_every):
        samples_idx_list.append(sample_idx)

    return samples_idx_list
def load_dataset():
    splits = load_json_files(config['disaster_splits_json'])
    data_mean_stddev = load_json_files(config['disaster_mean_stddev'])

    train_ls = [] 
    val_ls = []
    for item, val in splits.items():
        train_ls += val['train'] 
        val_ls += val['val']
    xBD_train = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'train', data_mean_stddev, transform=True, normalize=True)
    xBD_val = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'val', data_mean_stddev, transform=False, normalize=True)

    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

    return xBD_train, xBD_val