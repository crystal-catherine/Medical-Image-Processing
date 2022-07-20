import torch
import torch.nn as nn
import torch.nn.functional as F


def calDice(gt, pred, eps=1):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    tp, fp, fn = 0, 0, 0
    row, col = gt.shape
    # print(gt.shape, pred.shape)
    for i in range(row):
        for j in range(col):
            if gt[i][j] == 255 and pred[i][j] == 255:
                tp += 1
            if gt[i][j] == 0 and pred[i][j] == 255:
                fp += 1
            if gt[i][j] == 255 and pred[i][j] == 0:
                fn += 1

    # 转为float，以防long类型之间相除结果为0
    loss = float((2 * tp + eps)) / float((2 * tp + fp + fn + eps))

    return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, soft=False):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        if soft:
            return torch.log((torch.exp(score)+torch.exp(-score))/2.0)
        else:
            return score





def calPrecision(gt, pred, eps=1):
    row, col = gt.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if gt[i][j] == 255 and pred[i][j] == 255:
                P_s += 1
            if pred[i][j] == 255:
                P_t += 1

    Precision = float((P_s + eps)) / float((P_t + eps))

    return Precision


def calRecall(gt, pred, eps=1):
    row, col = gt.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if gt[i][j] == 255 and pred[i][j] == 255:
                R_s += 1
            if gt[i][j] == 255:
                R_t += 1

    Recall = float((R_s + eps)) / float((R_t + eps))

    return Recall

