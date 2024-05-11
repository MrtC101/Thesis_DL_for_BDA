def prepare_for_vis(sample_train_ids, logger, model, which_set, iteration, device, softmax):
    
    for item in sample_train_ids:
        data = xBD_train[item] if which_set == 'train' else xBD_val[item]
        
        c = data['pre_image'].size()[0]
        h = data['pre_image'].size()[1]
        w = data['pre_image'].size()[2]

        pre = data['pre_image'].reshape(1, c, h, w)
        post = data['post_image'].reshape(1, c, h, w)
        
        
        scores = model(pre.to(device=device), post.to(device=device))
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
        
        # modify damage prediction based on UNet arm        
        for c in range(0,scores[2].shape[1]):
            scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)

        # add to tensorboard
        tag = 'pr_bld_mask_pre_train_id_' + str(item) if which_set == 'train' else 'pr_bld_mask_pre_val_id_' + str(item)
        logger.add_image(tag, preds_seg_pre, iteration, dataformats='CHW')
        
        tag = 'pr_bld_mask_post_train_id_' + str(item) if which_set == 'train' else 'pr_bld_mask_post_val_id_' + str(item)
        logger.add_image(tag, preds_seg_post, iteration, dataformats='CHW')
        
        tag = 'pr_dmg_mask_train_id_' + str(item) if which_set == 'train' else 'pr_dmg_mask_val_id_' + str(item)
        im, buf = viz.show_label_raster(torch.argmax(softmax(scores[2]), dim=1).cpu().numpy(), size=(5, 5))
        preds_cls = transforms.ToTensor()(transforms.ToPILImage()(np.array(im)).convert("RGB"))
        logger.add_image(tag, preds_cls, iteration, dataformats='CHW')
                    
        if iteration == 1:
            pre = data['pre_image']
            tag = 'gt_img_pre_train_id_' + str(item) if which_set == 'train' else 'gt_img_pre_val_id_' + str(item)
            logger.add_image(tag, data['pre_image_orig'], iteration, dataformats='CHW')
            
            post = data['post_image']
            tag = 'gt_img_post_train_id_' + str(item) if which_set == 'train' else 'gt_img_post_val_id_' + str(item)
            logger.add_image(tag, data['post_image_orig'], iteration, dataformats='CHW')
        
            gt_seg = data['building_mask'].reshape(1, h, w)
            tag = 'gt_bld_mask_train_id_' + str(item) if which_set == 'train' else 'gt_bld_mask_val_id_' + str(item)
            logger.add_image(tag, gt_seg, iteration, dataformats='CHW')
        
            im, buf = viz.show_label_raster(np.array(data['damage_mask']), size=(5, 5))
            gt_cls = transforms.ToTensor()(transforms.ToPILImage()(np.array(im)).convert("RGB"))
            tag = 'gt_dmg_mask_train_id_' + str(item) if which_set == 'train' else 'gt_dmg_mask_val_id_' + str(item)
            logger.add_image(tag, gt_cls, iteration, dataformats='CHW')    
    return