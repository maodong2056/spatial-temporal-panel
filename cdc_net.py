import torch
import torch.nn as nn


def _make_backbone_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        elif v == 'M0':
            layers += [nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _make_cdc_pairs(in_channels, out_channel, kernel_size):
    layers = [nn.Conv3d(in_channels, out_channel, kernel_size=kernel_size, padding=(1, 0, 0)),
              nn.ReLU(inplace=True),
              nn.Dropout3d(inplace=True)]
    return nn.Sequential(*layers), nn.Sequential(*layers)


class CDCNet(nn.Module):
    """Convolution-De-Convolution Network
    See details: https://arxiv.org/pdf/1703.01515.pdf

    Configuration:
        'M': Maxpool with stride (2, 2, 2)
        'M0': Maxpool with stride (1, 2, 2), i.e., keep the depth but downsize height and width
    """
    cfg = [64, 'M0', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M0']

    def __init__(self, num_classes):
        super(CDCNet, self).__init__()
        self.num_classes = num_classes
        self.features = _make_backbone_layers(CDCNet.cfg)
        self.cdc_6_1, self.cdc_6_2 = _make_cdc_pairs(512, 4096, kernel_size=(3, 4, 4))
        self.cdc_7_1, self.cdc_7_2 = _make_cdc_pairs(4096, 4096, kernel_size=(3, 1, 1))
        self.cdc_8_1, self.cdc_8_2 = _make_cdc_pairs(4096, num_classes, kernel_size=(3, 1, 1))

    def forward(self, x):
        """inputs shape: [batch_size(1), channel(1), depth(video_length), height(128), width(128)]
        """
        features = self.features(x)  # [1, 512, L/8, 4, 4]

        cdc_6_1, cdc_6_2 = self.cdc_6_1(features), self.cdc_6_2(features)
        cdc_6 = torch.cat([cdc_6_1, cdc_6_2], dim=2)  # [1, 4096, L/4, 1, 1]

        cdc_7_1, cdc_7_2 = self.cdc_7_1(cdc_6), self.cdc_7_2(cdc_6)
        cdc_7 = torch.cat([cdc_7_1, cdc_7_2], dim=2)  # [1, 4096, L/2, 1, 1]

        cdc_8_1, cdc_8_2 = self.cdc_8_1(cdc_7), self.cdc_8_2(cdc_7)
        cdc_8 = torch.cat([cdc_8_1, cdc_8_2], dim=2)  # [1, #classes, L , 1, 1]

        return cdc_8
