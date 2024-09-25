# Copyright (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
import os
import re
import shutil
from datetime import datetime
from time import localtime, strftime
import torch
import torch.utils
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
from utils.common.pathManager import FilePath
from models.siam_unet_model import SiamUnet


class TrainModel(SiamUnet):
    """`SiamUnet` class that implements all methods related to load and save weights
    during model training."""

    def save_checkpoint(self, state: dict, is_best: bool,
                        checkpoint_dir: str = '../checkpoints') -> None:
        """
            Saves weights from current epoch as checkpoint. If `is_best` is True a file named
            'model_best' is stored in the same folder.

            Args:
                state: dictionary with current epoch important variables to save.
                is_best: True if this is the epoch with highest harmonic f1 score so far.
                chekcpoint_dir: path to the folder where checkpoint is stored.
        """
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f"checkpoint_epoch{state['epoch']}_"
                                       f"{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.pth.tar")
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, os.path.join(
                checkpoint_dir, 'model_best.pth.tar'))

    def resume_from_checkpoint(self, checkpoint_dir: str, device,
                               init_learning_rate, tb_logger, freeze_seg: bool) -> \
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

        if freeze_seg:
            self.freeze_segmentation_branch()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=init_learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=init_learning_rate)

        starting_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_f1', 0.0)
        return optimizer, starting_epoch, best_acc

    def load_freezed_weights(self, weights_path: FilePath, device: torch.device) -> tuple:
        """
        Loads weights from the weights_path.
        Args:
            weights_path: Path to .tar file of weights saved with `torch.save()`.
            device: Device where to map weights.

        Returns:
            torch.optim.Adam: Optimizer for training.
            int: next starting epoch.
            float: best harmonic f1 score from given weights epoch.
        """
        checkpoint = torch.load(weights_path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self.freeze_model_params()
        optimizer = torch.optim.Adam(self.parameters())
        starting_epoch = checkpoint['epoch'] + 1
        hf1 = checkpoint.get('best_f1', 0.0)
        return optimizer, starting_epoch, hf1

    def freeze_segmentation_branch(self):
        """Disables weight and bias updates for specific segmentation branch layers in the model."""

        # List of indices for weight only and weight + bias layers
        weight_only_indices = [0, 3]
        weight_bias_indices = [1, 4]

        # Helper function to freeze weight (and optionally bias)
        def freeze_module(module, indices):
            for i in indices:
                module[i].weight.requires_grad = False
                if hasattr(module[i], 'bias') and module[i].bias is not None:
                    module[i].bias.requires_grad = False

        # Freeze weights and biases for encoder, bottleneck, and decoder
        encoders = [self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        decoders = [self.decoder1, self.decoder2, self.decoder3, self.decoder4]

        for encoder, decoder in zip(encoders, decoders):
            freeze_module(encoder, weight_only_indices)
            freeze_module(encoder, weight_bias_indices)
            freeze_module(decoder, weight_only_indices)
            freeze_module(decoder, weight_bias_indices)

        # Freeze weights and biases for bottleneck
        freeze_module(self.bottleneck, weight_only_indices)
        freeze_module(self.bottleneck, weight_bias_indices)

        # Freeze upconvolutional layers
        upconvs = [self.upconv1, self.upconv2, self.upconv3, self.upconv4]
        for upconv in upconvs:
            upconv.weight.requires_grad = False
            if hasattr(upconv, 'bias') and upconv.bias is not None:
                upconv.bias.requires_grad = False

        # Freeze final convolution layer
        self.conv_s.weight.requires_grad = False
        if hasattr(self.conv_s, 'bias') and self.conv_s.bias is not None:
            self.conv_s.bias.requires_grad = False

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

    def resume_from_scratch(self, init_learning_rate) -> tuple:
        """Resumes model from scratch"""
        optimizer = torch.optim.Adam(self.parameters(), lr=init_learning_rate)
        starting_epoch = 1
        best_acc = 0.0
        return optimizer, starting_epoch, best_acc
