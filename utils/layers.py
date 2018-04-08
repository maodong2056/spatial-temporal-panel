import torch
import torch.nn as nn
import torch.nn.functional as F

FLOAT = torch.cuda.FloatTensor
LONG = torch.cuda.LongTensor


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def bilinear_interpolate_length(x, target_length):
    """Ensure the output of neural networks to be same length with the input,
    regarding the multiple downsample of odd size.
    Args:
        x: [batch_size(1), num_classes(13), depth(length), height(1), width(1)]
        target_length: target length
    Return:
        y: [num_classes, target_length]
    """
    x = x[0]  # [#cls, d, 1, 1] <==> [W, H, C, 1]
    x = x.permute(3, 2, 0, 1)  # ==> [1, C, W, H]

    W, H = x.size()[2:]

    samples_x = torch.LongTensor([W])[:, None, None, None]
    samples_y = torch.LongTensor([target_length])[:, None, None, None]
    samples = torch.cat([samples_x, samples_y], 3)

    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1
    return F.grid_sample(x, samples)
