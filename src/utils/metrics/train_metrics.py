import pandas as pd
import torch

class MetricComputer:

    def __init__(self,phase_context,static_context):
        self.logger = phase_context['logger']
        self.phase = phase_context['phase']
        self.loader = phase_context['loader']
        self.device = static_context['device']
        self.crit_seg_1 = static_context['crit_seg_1'] 
        self.crit_seg_2 = static_context['crit_seg_2']
        self.crit_dmg = static_context['crit_dmg']

    ### Compute confusión Matrixes
    def compute_conf_mtrx(self, y_pred_mask, y_dmg_mask, y_bld_mask, labels_set, conf_mtrx_df, epoch, batch_idx):
        conf_mtrx_list = []
        for cls in labels_set:

            if len(labels_set) <= 2:
                conf_mtrx = self.conf_mtrx_for_bld_mask(y_pred_mask, y_bld_mask,cls)
            else:
                conf_mtrx = self.conf_mtrx_for_cls_mask(y_pred_mask, y_dmg_mask, y_bld_mask, cls)

            conf_mtrx["epoch"] = epoch
            conf_mtrx["batch_idx"] = batch_idx
            conf_mtrx_list.extend([conf_mtrx])
        curr_conf_mtrx_df = pd.DataFrame(conf_mtrx_list,columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])
        if(len(conf_mtrx_df)>0):
            conf_mtrx_df = pd.concat([conf_mtrx_df,curr_conf_mtrx_df],ignore_index=True)
        else:
            conf_mtrx_df = curr_conf_mtrx_df
        return conf_mtrx_df

    def conf_mtrx_for_bld_mask(self,y_preds, y_true, cls):

        y_true_binary = y_true.detach().clone()
        y_preds_binary = y_preds.detach().clone()
        
        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0)).float().sum().item()
        
        # compute total pixels
        total_pixels = 1
        for item in y_true_binary.size():
                total_pixels *= item
        return {'class':cls, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total_pixels':total_pixels}
    
    def conf_mtrx_for_cls_mask(self,y_preds, y_dmg_mask, y_bld_mask, cls):
        
        # Convert any other class to 0 
        y_true_binary = y_dmg_mask.detach().clone()
        y_true_binary[y_true_binary != cls] = -1
        y_true_binary[y_true_binary == cls] = 1
        y_true_binary[y_true_binary == -1] = 0

        y_preds_binary = y_preds.detach().clone()
        y_preds_binary[y_preds_binary != cls] = -1
        y_preds_binary[y_preds_binary == cls] = 1                
        y_preds_binary[y_preds_binary == -1] = 0

        # compute confusion metric
        true_pos_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 1) & (y_bld_mask == 1)).float().sum().item()
        false_neg_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 1) & (y_bld_mask == 1)).float().sum().item()
        true_neg_cls = ((y_true_binary == y_preds_binary) & (y_true_binary == 0) & (y_bld_mask == 1)).float().sum().item()
        false_pos_cls = ((y_true_binary != y_preds_binary) & (y_true_binary == 0) & (y_bld_mask == 1)).float().sum().item()
        
        # compute total pixels
        total_pixels = y_bld_mask.float().sum().item()
        return {'class':cls, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total_pixels':total_pixels}


    ### Compute Metrics
    def compute_metrics_for(self, output, epoch, labels_set, conf_mtrx_df):
        class_metrics, f1_harmonic_mean = self.compute_eval_metrics(epoch, labels_set, conf_mtrx_df)
        self.log_metrics(self.phase,output, self.logger, class_metrics)
        if(output == "dmg"):
            self.logger.add_scalar(f'{self.phase}_dmg_harmonic_mean_f1', f1_harmonic_mean, epoch)
        return class_metrics, f1_harmonic_mean

    def compute_eval_metrics(self, epoch, labels_set, conf_mtrx_df : pd.DataFrame):
        eval_results = []
        f1_harmonic_mean = 0
        for cls in labels_set: 
            class_idx = (conf_mtrx_df['class']==cls)
            tp = conf_mtrx_df.loc[class_idx,'true_pos'].sum()
            fp = conf_mtrx_df.loc[class_idx,'false_pos'].sum()
            fn = conf_mtrx_df.loc[class_idx,'false_neg'].sum()
            tn = conf_mtrx_df.loc[class_idx,'true_neg'].sum()
            tot = conf_mtrx_df.loc[class_idx,'total_pixels'].sum()
            
            precision = tp / (tp + fp)
            if(tp > 0 and fn > 0):
               recall = tp / (tp + fn)
            else:
                recall = 0 
            if(precision > 0 and recall > 0):
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            accuracy = (tp + tn) / (tot)

            eval_results.append({'epoch':epoch, 'class':cls, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy})
            f1_harmonic_mean += 1.0 / (f1 + 1e-10)
        f1_harmonic_mean = len(labels_set) / f1_harmonic_mean
        df =  pd.DataFrame(eval_results,columns=['epoch','class','precision','recall','f1','accuracy'])
        return df, f1_harmonic_mean

    def log_metrics(self,phase, output, logger, metrics_df : pd.DataFrame):
        for index, row in metrics_df.iterrows():
            for metric in ["f1"]:
                msg = f"{phase}_{output}_class_{row['class']}_{metric}"
                logger.add_scalar(msg,row[metric],row['epoch'])