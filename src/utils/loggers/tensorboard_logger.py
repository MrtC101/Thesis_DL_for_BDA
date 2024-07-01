import torch
from utils.dataloaders.train_dataloader import TrainDataLoader
from utils.visualization.raster_label_visualizer import RasterLabelVisualizer
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger(SummaryWriter):
    """
        Implementation of a SummaryWriter that stores the corresponding
        samples of images that are shown during training on tensor board.
    """

    def __init__(self, tb_logger_dir : str, num_patches_to_vis : int,
                  label_map_json : str):
        super().__init__(log_dir=tb_logger_dir)
        self.num_patches_to_vis = num_patches_to_vis
        self.viz = RasterLabelVisualizer(label_map=label_map_json)
        self.softmax = torch.nn.Softmax(dim=1)

    def tb_log_images(self, mode_value : str, loader : TrainDataLoader, model, epoch, device):
        """This method creates logging image files for tensorboard visualization"""
        
        def log_patch(pred_bld_mask, pred_dmg_mask, prefix, idx, epoch):
            current_patch = {}
            current_patch['bld_mask'] = self.viz.bld_mask_raster(pred_bld_mask) 
            current_patch['dmg_mask'] = self.viz.dmg_mask_raster(pred_dmg_mask)
            for key, img in current_patch.items():
                tag = f"{prefix}/images/{key}_{idx}"
                self.add_image(tag, img, epoch, dataformats='CHW')

        patches = loader.det_img_sample(number=self.num_patches_to_vis,
                                                    normalized=True)
        if (epoch == 1):
            # Log ground truth images once for "epoch"=0
            org_patches = loader.det_img_sample(number=self.num_patches_to_vis,
                                                    normalized=False)
            for idx, (dis_id,tile_id,patch_id,org_patche) in enumerate(org_patches):
                tag = f"{mode_value}/images/pre_img_{idx}"
                self.add_image(tag, org_patche["pre_img"], 0, dataformats='CHW')
                tag = f"{mode_value}/images/post_img_{idx}"
                self.add_image(tag, org_patche["post_img"], 0, dataformats='CHW')           

                
        for idx, (dis_id,tile_id,patch_id,patch) in enumerate(patches):

            if (epoch == 1):
                #Normalized images
                tag = f"{mode_value}/images/pre_img_{idx}"
                self.add_image(tag, patch["pre_img"], 1, dataformats='CHW')
                tag = f"{mode_value}/images/post_img_{idx}"
                self.add_image(tag, patch["post_img"], 1, dataformats='CHW')
            
            #Make a prediction
            c, h, w = patch['pre_img'].size()
            pre = patch['pre_img'].reshape(1, c, h, w)
            post = patch['post_img'].reshape(1, c, h, w)
            scores = model(pre.to(device=device), post.to(device=device))
            pred_bld_mask = torch.argmax(self.softmax(scores[0]), dim=1)
            pred_dmg_mask = torch.argmax(self.softmax(scores[2]), dim=1)   
            log_patch(pred_bld_mask, pred_dmg_mask, mode_value, idx, epoch)

