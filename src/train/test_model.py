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
