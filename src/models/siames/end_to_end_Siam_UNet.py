# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
#
# Modification Notes:
# - Documentation added with docstrings for code clarity.
# - Re-implementation of methods to enhance readability and efficiency.
# - Re-implementation of features for improved functionality.
# - Changes in the logic of implementation for better performance.
# - Bug fixes in the code.
#
# See the LICENSE file in the root directory of this project for the full text of the MIT License.

from collections import OrderedDict
import os
import shutil
from time import localtime, strftime
import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard
import torch.utils.tensorboard.summary


class SiamUnet(nn.Module):
    """This class implements the architecture of the Siamese CNN used in this project."""

    def __init__(self, in_channels: int = 3, out_channels_s: int = 2, out_channels_c: int = 5,
                 init_features: int = 16) -> None:
        super(SiamUnet, self).__init__()

        features = init_features

        # UNet layers
        self.encoder1 = SiamUnet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SiamUnet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = SiamUnet._block(
            features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = SiamUnet._block(
            features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SiamUnet._block(
            features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = SiamUnet._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = SiamUnet._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = SiamUnet._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = SiamUnet._block(features * 2, features, name="dec1")

        self.conv_s = nn.Conv2d(in_channels=features,
                                out_channels=out_channels_s, kernel_size=1)

        # Siamese classifier layers
        self.upconv4_c = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.conv4_c = SiamUnet._block(
            features * 16, features * 16, name="conv4")

        self.upconv3_c = nn.ConvTranspose2d(
            features * 16, features * 4, kernel_size=2, stride=2)
        self.conv3_c = SiamUnet._block(
            features * 8, features * 8, name="conv3")

        self.upconv2_c = nn.ConvTranspose2d(
            features * 8, features * 2, kernel_size=2, stride=2)
        self.conv2_c = SiamUnet._block(
            features * 4, features * 4, name="conv2")

        self.upconv1_c = nn.ConvTranspose2d(
            features * 4, features, kernel_size=2, stride=2)
        self.conv1_c = SiamUnet._block(
            features * 2, features * 2, name="conv1")

        self.conv_c = nn.Conv2d(in_channels=features * 2,
                                out_channels=out_channels_c, kernel_size=1)

        self.softmax = torch.nn.Softmax(dim=1)

    
    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """
        Creates a block of layers.

        Args:
            in_channels: The number of input channels.
            features: The number of features for the convolutional layers.
            name: The base name for the layers.

        Returns:
            nn.Sequential: A sequential container of the layers.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


    #Used for debugging
    def forward(self, x1, x2):
        a = nn.Conv2d(3, 2, kernel_size=1)(x1)
        b = nn.Conv2d(3, 5, kernel_size=1)(x2)

        # modify damage prediction based on UNet arm
        preds_seg_pre = torch.argmax(self.softmax(a), dim=1)
        for c in range(0, b.shape[1]):
            b[:, c, :, :] = torch.mul(b[:, c, :, :], preds_seg_pre)

        return a, a, b
    

    """
     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        \"""
        Forward pass through the Siamese U-Net.

        Args:
            x1: The first input tensor.
            x2: The second input tensor.

        Returns:
            tuple: A tuple of logits tensors masks for segmentation and classification.
        \"""

        # UNet on x1
        enc1_1 = self.encoder1(x1)
        enc2_1 = self.encoder2(self.pool1(enc1_1))
        enc3_1 = self.encoder3(self.pool2(enc2_1))
        enc4_1 = self.encoder4(self.pool3(enc3_1))

        bottleneck_1 = self.bottleneck(self.pool4(enc4_1))

        dec4_1 = self.upconv4(bottleneck_1)
        dec4_1 = torch.cat((dec4_1, enc4_1), dim=1)
        dec4_1 = self.decoder4(dec4_1)
        dec3_1 = self.upconv3(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3_1), dim=1)
        dec3_1 = self.decoder3(dec3_1)
        dec2_1 = self.upconv2(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2_1), dim=1)
        dec2_1 = self.decoder2(dec2_1)
        dec1_1 = self.upconv1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1_1), dim=1)
        dec1_1 = self.decoder1(dec1_1)

        # UNet on x2
        enc1_2 = self.encoder1(x2)
        enc2_2 = self.encoder2(self.pool1(enc1_2))
        enc3_2 = self.encoder3(self.pool2(enc2_2))
        enc4_2 = self.encoder4(self.pool3(enc3_2))

        bottleneck_2 = self.bottleneck(self.pool4(enc4_2))

        dec4_2 = self.upconv4(bottleneck_2)
        dec4_2 = torch.cat((dec4_2, enc4_2), dim=1)
        dec4_2 = self.decoder4(dec4_2)
        dec3_2 = self.upconv3(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3_2), dim=1)
        dec3_2 = self.decoder3(dec3_2)
        dec2_2 = self.upconv2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
        dec2_2 = self.decoder2(dec2_2)
        dec1_2 = self.upconv1(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
        dec1_2 = self.decoder1(dec1_2)

        # Siamese
        dec1_c = bottleneck_2 - bottleneck_1

        dec1_c = self.upconv4_c(dec1_c)  # features * 16 -> features * 8
        diff_2 = enc4_2 - enc4_1  # features * 16 -> features * 8
        dec2_c = torch.cat((diff_2, dec1_c), dim=1)  # 512
        dec2_c = self.conv4_c(dec2_c)

        dec2_c = self.upconv3_c(dec2_c)  # 512->256
        diff_3 = enc3_2 - enc3_1
        dec3_c = torch.cat((diff_3, dec2_c), dim=1)  # ->512
        dec3_c = self.conv3_c(dec3_c)

        dec3_c = self.upconv2_c(dec3_c)  # 512->256
        diff_4 = enc2_2 - enc2_1
        dec4_c = torch.cat((diff_4, dec3_c), dim=1)
        dec4_c = self.conv2_c(dec4_c)

        dec4_c = self.upconv1_c(dec4_c)
        diff_5 = enc1_2 - enc1_1
        dec5_c = torch.cat((diff_5, dec4_c), dim=1)
        dec5_c = self.conv1_c(dec5_c)

        out_seg_1 = self.conv_s(dec1_1)
        out_seg_2 = self.conv_s(dec1_2)
        out_class = self.conv_c(dec5_c)

        # modify damage prediction based on UNet arm
        preds_seg_pre = torch.argmax(self.softmax(out_seg_1), dim=1)
        for c in range(0, out_class.shape[1]):
            out_class[:, c, :, :] = torch.mul(
                out_class[:, c, :, :], preds_seg_pre)

        return out_seg_1, out_seg_2, out_class
    """

    """
    Returns logits
    """
    
    def compute_probabilities(self, logit_masks) -> tuple:
        """
            Returns:
                tuple: A tuple of probabilities tensor masks for segmentation and classification.
        """
        return tuple(self.softmax(logit_mask) for logit_mask in logit_masks)

    def compute_predictions(self, logit_masks) -> tuple:
        """
            Returns:
                tuple: A tuple of predictions tensor masks for segmentation and classification.
        """
        return tuple(torch.argmax(self.softmax(logit_mask), dim=1) for logit_mask in logit_masks)
    
    def compute_binary(self, logit_masks, threshold = 0.5) -> tuple:
        probabilities = self.compute_probabilities(logit_masks)
        return tuple(mask >= threshold for mask in probabilities)

    def freeze_model_param(self):
        """Disables weights modification on each layer from the model"""
        for i in [0, 3]:
            self.encoder1[i].weight.requires_grad = False
            self.encoder2[i].weight.requires_grad = False
            self.encoder3[i].weight.requires_grad = False
            self.encoder4[i].weight.requires_grad = False

            self.bottleneck[i].weight.requires_grad = False

            self.decoder4[i].weight.requires_grad = False
            self.decoder3[i].weight.requires_grad = False
            self.decoder2[i].weight.requires_grad = False
            self.decoder1[i].weight.requires_grad = False

        for i in [1, 4]:
            self.encoder1[i].weight.requires_grad = False
            self.encoder1[i].bias.requires_grad = False

            self.encoder2[i].weight.requires_grad = False
            self.encoder2[i].bias.requires_grad = False

            self.encoder3[i].weight.requires_grad = False
            self.encoder3[i].bias.requires_grad = False

            self.encoder4[i].weight.requires_grad = False
            self.encoder4[i].bias.requires_grad = False

            self.bottleneck[i].weight.requires_grad = False
            self.bottleneck[i].bias.requires_grad = False

            self.decoder4[i].weight.requires_grad = False
            self.decoder4[i].bias.requires_grad = False

            self.decoder3[i].weight.requires_grad = False
            self.decoder3[i].bias.requires_grad = False

            self.decoder2[i].weight.requires_grad = False
            self.decoder2[i].bias.requires_grad = False

            self.decoder1[i].weight.requires_grad = False
            self.decoder1[i].bias.requires_grad = False

        self.upconv4.weight.requires_grad = False
        self.upconv4.bias.requires_grad = False

        self.upconv3.weight.requires_grad = False
        self.upconv3.bias.requires_grad = False

        self.upconv2.weight.requires_grad = False
        self.upconv2.bias.requires_grad = False

        self.upconv1.weight.requires_grad = False
        self.upconv1.bias.requires_grad = False

        self.conv_s.weight.requires_grad = False
        self.conv_s.bias.requires_grad = False

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

        self.conv4_c.apply(SiamUnet.init_weights)
        self.conv3_c.apply(SiamUnet.init_weights)
        self.conv2_c.apply(SiamUnet.init_weights)
        self.conv1_c.apply(SiamUnet.init_weights)

    def init_weights(m):
        """Inits a weights of the current layer with xavier uniform"""
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def resume_from_checkpoint(self, checkpoint_path: str, device,
                                init_learning_rate, tb_logger, new_optimizer : bool) -> \
                                tuple[torch.optim.Optimizer, int, float]:
        """
        Resumes the model from a checkpoint.

        Args:
            checkpoint_path : The path to the checkpoint file.
            tb_log_dir : The directory for TensorBoard logs.
            config : Configuration dictionary containing necessary parameters.

        Returns:
            tuple: A tuple containing the optimizer, starting epoch, and best accuracy.
        """

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])

        if not new_optimizer:
            # don't load the optimizer settings so that a newly
            # specified lr can take effect
            self.print_network()
            self.freeze_model_param()
            self.print_network()

            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                tb_logger.add_histogram(tag, value.data.cpu().numpy(),global_step=0)

            self.reinitialize_Siamese()

            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                tb_logger.add_histogram(tag, value.data.cpu().numpy(),global_step=1)
                
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),
                                         lr=init_learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(),lr=init_learning_rate)

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
