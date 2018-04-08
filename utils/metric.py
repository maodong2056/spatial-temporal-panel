import torch.nn.functional as F


def calculate_video_iou(pred, true):
    pass


def nll_loss(pred, true):
    """
    Args:
        pred: <Variable> [batch_size, num_classes, length, 1, 1]
        true: <Variable> [batch_size, length]
    """
    pred = pred.squeeze()
    bs, num_cls, length = pred.size()
    true = true.view(-1).long()
    pred = pred.permute(0, 2, 1).contiguous().view(-1, num_cls)
    loss = F.nll_loss(F.log_softmax(pred, dim=1), true)
    return loss


def accuracy(pred, true):
    predictions = pred.max(1)[1].type_as(true)
    correct = (predictions.eq(true)).float()
    return correct.mean() * 100
