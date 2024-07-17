# Copyright (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import os
import re
import shutil
from datetime import datetime
from time import localtime, strftime
import torch
import torch.nn as nn
import torch.utils
import torch.utils.tensorboard
import torch.utils.tensorboard.summary

from models.siam_unet_model import SiamUnet


class TrainModel(SiamUnet):
    """This class implements the architecture of the Siamese CNN used in this project."""

    def make_binary(self, gt_dmg_mask: torch.Tensor, label_set: list) -> torch.Tensor:
        bin_mask_list = [gt_dmg_mask == label for label in label_set]
        return torch.stack(bin_mask_list)

    def compute_binary(self, dmg_logit_mask: torch.Tensor, threshold=0.5) -> torch.Tensor:
        bin_mask = self.softmax(dmg_logit_mask) >= threshold
        return bin_mask

    def reinitialize_Siamese(self):
        """initialize all layers from the model"""
        torch.nn.init.xavier_uniform_(self.upconv4_c.weight)
        torch.nn.init.xavier_uniform_(self.upconv3_c.weight)
        torch.nn.init.xavier_uniform_(self.upconv2_c.weight)
        torch.nn.init.xavier_uniform_(self.upconv1_c.weight)
        torch.nn.init.xavier_uniform_(self.conv_c.weight)

        self.upconv4_c.bias.data.fill_(0.01)
        self.upconv3_c.bias.data.fill_(0.01)
        self.upconv2_c.bias.data.fill_(0.01)
        self.upconv1_c.bias.data.fill_(0.01)
        self.conv_c.bias.data.fill_(0.01)

        def init_weights(m):
            """Inits a weights of the current layer with xavier uniform"""
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.conv4_c.apply(init_weights)
        self.conv3_c.apply(init_weights)
        self.conv2_c.apply(init_weights)
        self.conv1_c.apply(init_weights)

    def find_last(self, checkpoint_dir: str) -> str:
        """
        Finds the latest checkpoint file based on the date in the filename.

        Args:
            checkpoint_dir (str): Directory containing checkpoint files.

        Returns:
            str: The path to the latest checkpoint file.
        """
        files = os.listdir(checkpoint_dir)
        date_pattern = re.compile(r'checkpoint_epoch\d+_(\d{4}-\d{2}-\d{2}' +
                                  r'-\d{2}-\d{2}-\d{2}).pth.tar')

        latest_file_name = None
        latest_date = None

        for file_name in files:
            if file_name != 'model_best.pth.tar':
                match = date_pattern.search(file_name)
                if match:
                    file_date = datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file_name = file_name

        if latest_file_name is None:
            raise FileNotFoundError(
                "No valid checkpoint files found in the directory.")

        checkpoint_path = os.path.join(checkpoint_dir, latest_file_name)
        return checkpoint_path

    def resume_from_checkpoint(self, checkpoint_dir: str, device,
                               init_learning_rate, tb_logger, new_optimizer: bool) -> \
            tuple[torch.optim.Optimizer, int, float]:
        """
        Resumes the model from a checkpoint.

        Args:
            checkpoint_dir : The path to the checkpoint file.
            tb_log_dir : The directory for TensorBoard logs.
            config : Configuration dictionary containing necessary parameters.

        Returns:
            tuple: A tuple containing the optimizer, starting epoch, and best accuracy.
        """

        checkpoint_path = self.find_last(checkpoint_dir)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])

        if not new_optimizer:
            # don't load the optimizer settings so that a newly
            # specified lr can take effect
            #self.print_network()
            self.freeze_model_param()
            #self.print_network()

            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                tb_logger.add_histogram(
                    tag, value.data.cpu().numpy(), global_step=0)

            self.reinitialize_Siamese()

            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                tb_logger.add_histogram(
                    tag, value.data.cpu().numpy(), global_step=1)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=init_learning_rate)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=init_learning_rate)

        starting_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_f1', 0.0)
        return optimizer, starting_epoch, best_acc

    def resume_from_scratch(self, init_learning_rate) -> tuple:
        """Resumes model from scratch"""
        optimizer = torch.optim.Adam(self.parameters(), lr=init_learning_rate)
        starting_epoch = 1
        best_acc = 0.0
        return optimizer, starting_epoch, best_acc

    def save_checkpoint(self, state: dict, is_best: bool,
                        checkpoint_dir: str = '../checkpoints') -> None:
        """
        checkpoint_dir is used to save the best checkpoint if this checkpoint is best one so far.
        """
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f"checkpoint_epoch{state['epoch']}_"
                                       f"{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.pth.tar")
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, os.path.join(
                checkpoint_dir, 'model_best.pth.tar'))

    def print_network(self) -> None:
        print('model summary')
        for name, p in self.named_parameters():
            print(name)
            print(p.requires_grad)

    def model_summary(self) -> str:
        """
            Returns an string that contains a summary of the model total weights
        """
        class Text(str):
            def __add__(self, other) -> 'Text':
                return Text(super().__add__("\n").__add__(other))

        lay_n = 0
        total_params = 0
        model_params = [layer for layer in self.parameters()
                        if layer.requires_grad]
        layers = [child for child in self.children()]
        lay_t = Text("")
        for layer in layers:
            lay_n_params = model_params[lay_n].numel()
            lay_n += 1
            if hasattr(layer, "bias"):
                if (layer.bias is not None):
                    lay_n_params += model_params[lay_n].numel()
                    lay_n += 1
            total_params += lay_n_params
            lay_t += (str(layer)+"\t"*3+str(lay_n_params))

        t = Text("self_summary")
        t += ""
        t += "Layer_name"+"\t"*7+"Number of Parameters"
        t += "="*100
        t += "\t"*10
        t += lay_t
        t += "="*100
        t += f"Total Params:{total_params}"
        return str(t)
