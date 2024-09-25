# Copyright (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils


class SiamUnet(nn.Module):
    """
    Fully Convolutional U-Net-like Siamese architecture for Building Damage Assessment of
    pre-disaster and post-disaster images from the xBD dataset.
    """

    def __init__(self, in_channels: int = 3, out_channels_s: int = 2,
                 out_channels_c: int = 5, features: int = 16) -> None:
        """
            Args:
                in_channels : number of channels from input images.
                out_channels_s: number of channels for segmentation output. (one per class)
                out_channels_c: number of chnnels for classification output. (one per class)
                features: number of feature maps or kernnels for the first layer.
        """
        super(SiamUnet, self).__init__()

        # Segmentation branch layers
        self.encoder1 = SiamUnet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SiamUnet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = SiamUnet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = SiamUnet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SiamUnet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = SiamUnet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = SiamUnet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = SiamUnet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = SiamUnet._block(features * 2, features, name="dec1")

        self.conv_s = nn.Conv2d(in_channels=features, out_channels=out_channels_s, kernel_size=1)

        # Clasifier branch layers
        self.upconv4_c = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv4_c = SiamUnet._block(features * 16, features * 8, name="conv4")
        self.upconv3_c = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv3_c = SiamUnet._block(features * 8, features * 4, name="conv3")
        self.upconv2_c = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv2_c = SiamUnet._block(features * 4, features * 2, name="conv2")
        self.upconv1_c = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv1_c = SiamUnet._block(features * 2, features, name="conv1")

        self.conv_c = nn.Conv2d(in_channels=features, out_channels=out_channels_c, kernel_size=1)

        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """
        Creates a defined simple block of layers.

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

    """
    #Used for debugging only
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
        """
        Defines the sequence on which images pass through the Siamese U-Net on foward step.

        Args:
            x1: Can be pre or post input patch tensor image.
            x2: Can be pre or post input patch tensor image.

        Returns:
            tuple: A tuple of logits tensors masks for segmentation and classification. 
            (pre_seg,post_seg,dmg_mask)
        """

        # Segmentation branch 1

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

        out_seg_1 = self.conv_s(dec1_1)

        # Segmentation branch 2
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

        out_seg_2 = self.conv_s(dec1_2)

        # Damage classification branch
        diff_1 = bottleneck_2 - bottleneck_1  # 256x16
        dec1_c = self.upconv4_c(diff_1)  # 256x16 -> 128x32

        diff_2 = enc4_2 - enc4_1  # 128x32
        dec2_c = torch.cat((diff_2, dec1_c), dim=1)  # 256x32
        dec2_c = self.conv4_c(dec2_c)  # 256x32 -> 128x32
        dec2_c = self.upconv3_c(dec2_c)  # 128x32 -> 64x64

        diff_3 = enc3_2 - enc3_1  # 64x64
        dec3_c = torch.cat((diff_3, dec2_c), dim=1)  # 128x64
        dec3_c = self.conv3_c(dec3_c)  # 128x64 -> 64x64
        dec3_c = self.upconv2_c(dec3_c)  # 64x64 -> 32x128

        diff_4 = enc2_2 - enc2_1  # 32x128
        dec4_c = torch.cat((diff_4, dec3_c), dim=1)  # 32x128
        dec4_c = self.conv2_c(dec4_c)  # 32x128 -> 64x128
        dec4_c = self.upconv1_c(dec4_c)  # 64x128 -> 16x256

        diff_5 = enc1_2 - enc1_1  # 16x256
        dec5_c = torch.cat((diff_5, dec4_c), dim=1)  # 32x256
        dec5_c = self.conv1_c(dec5_c)  # 32x256 -> 16x256
        out_class = self.conv_c(dec5_c)  # 16x256 -> 5x256

        # modify damage prediction based on UNet arm
        preds_seg_pre = torch.argmax(self.softmax(out_seg_1), dim=1)
        for c in range(0, out_class.shape[1]):
            out_class[:, c, :, :] = torch.mul(out_class[:, c, :, :], preds_seg_pre)

        return out_seg_1, out_seg_2, out_class

    def compute_predictions(self, logit_masks: tuple) -> tuple:
        """
            Applys Softmax for each output logit mask.
            Returns:
                tuple: A tuple of predictions tensor masks for segmentation and classification. 
                (All three output masks are `torch.Tensors` of shapes (256,256))
        """
        return tuple(torch.argmax(self.softmax(logit_mask), dim=1) for logit_mask in logit_masks)

    def freeze_model_params(self):
        """
            This method iterates over al layers from the model and freeze their weights
        """
        for param in self.parameters():
            param.requires_grad = False
