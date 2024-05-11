def compute_eval_metrics(epoch, labels_set, confusion_mtrx_df, eval_results):
    eval_results = []
    for cls in labels_set: 
        class_idx = (confusion_mtrx_df['class']==cls)
        precision = confusion_mtrx_df.loc[class_idx,'true_pos'].sum()/(confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'false_pos'].sum())
        recall = confusion_mtrx_df.loc[class_idx,'true_pos'].sum()/(confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'false_neg'].sum())
        f1 = 2 * (precision * recall)/(precision + recall)
        accuracy = (confusion_mtrx_df.loc[class_idx,'true_pos'].sum() + confusion_mtrx_df.loc[class_idx,'true_neg'].sum())/(confusion_mtrx_df.loc[class_idx,'total_pixels'].sum())
        eval_results.append({'epoch':epoch, 'class':cls, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy})
    return pd.DataFrame(eval_results,columns=['epoch','class','precision','recall','f1','accuracy'])

def compute_confusion_mtrx(confusion_mtrx_df, epoch, batch_idx, labels_set, y_preds, y_true, y_true_bld_mask):
    all = []
    for cls in labels_set[1:]:
        A = compute_confusion_mtrx_class(confusion_mtrx_df, epoch, batch_idx, labels_set, y_preds, y_true, y_true_bld_mask, cls)
        all.append(A)
    res = []
    for x in all:
        res.extend(x)
    confusion_mtrx_df = pd.DataFrame(res,columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])
    return confusion_mtrx_df

def compute_confusion_mtrx_class(confusion_mtrx_list : list, epoch, batch_idx, labels_set, y_preds, y_true, y_true_bld_mask, cls):
    
    y_true_binary = y_true.detach().clone()
    y_preds_binary = y_preds.detach().clone()
    
    if len(labels_set) > 2:
        # convert to 0/1

        y_true_binary[y_true_binary != cls] = -1
        y_preds_binary[y_preds_binary != cls] = -1
        
        y_true_binary[y_true_binary == cls] = 1
        y_preds_binary[y_preds_binary == cls] = 1
        
        
        y_true_binary[y_true_binary == -1] = 0
        y_preds_binary[y_preds_binary == -1] = 0

        
        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1) & (y_true_bld_mask == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1) & (y_true_bld_mask == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0) & (y_true_bld_mask == 1)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0) & (y_true_bld_mask == 1)).float().sum().item()
        
        # compute total pixels
        total_pixels = y_true_bld_mask.float().sum().item()

    else:

        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        
        # compute total pixels
        total_pixels = 1
        for item in y_true_binary.size():
            total_pixels *= item
    
    confusion_mtrx_list.append({'epoch':epoch, 'class':cls, 'batch_idx':batch_idx, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total_pixels':total_pixels})
    
    return confusion_mtrx_list