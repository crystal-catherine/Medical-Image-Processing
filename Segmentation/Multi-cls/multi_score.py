import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, soft=False):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        # print(probs.shape)
        # print(targets.shape)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        if soft:
            return torch.log((torch.exp(score) + torch.exp(-score)) / 2.0)
        else:
            return score


def calDice(gt, pred, value, eps=1):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    tp, fp, fn = 0, 0, 0
    row, col = gt.shape
    # print(gt.shape, pred.shape)
    for i in range(row):
        for j in range(col):
            if gt[i][j] == value and pred[i][j] == value:
                tp += 1
            if gt[i][j] == 0 and pred[i][j] == value:
                fp += 1
            if gt[i][j] == value and pred[i][j] == 0:
                fn += 1

    # 转为float，以防long类型之间相除结果为0
    loss = float((2 * tp + eps)) / float((2 * tp + fp + fn + eps))

    return loss


def calPrecision(gt, pred, value, eps=1):
    row, col = gt.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if gt[i][j] == value and pred[i][j] == value:
                P_s += 1
            if pred[i][j] == value:
                P_t += 1

    Precision = float((P_s + eps)) / float((P_t + eps))

    return Precision


def calRecall(gt, pred, value, eps=1):
    row, col = gt.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if gt[i][j] == value and pred[i][j] == value:
                R_s += 1
            if gt[i][j] == value:
                R_t += 1

    Recall = float((R_s + eps)) / float((R_t + eps))

    return Recall


from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        # assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        # assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[-1], y)
        for i in range(1, len(x)):
           if weights[i] != 0:
               l += weights[i] * self.loss(x[i], y)
        return l / 5.
