import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice 损失函数，常用于分割任务。
    支持二分类、多分类和多标签分割。
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) 经过 sigmoid/softmax 前的预测 logits
            targets: (N, H, W) 或 (N, C, H, W)，标签
        """
        if inputs.shape != targets.shape:
            # 多分类情况: 对 inputs 做 softmax，再 one-hot 化 targets
            if inputs.size(1) > 1:  # 多分类
                inputs = F.softmax(inputs, dim=1)
                targets = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
            else:  # 二分类
                inputs = torch.sigmoid(inputs)
                targets = targets.unsqueeze(1).float()
        else:
            inputs = torch.sigmoid(inputs)

        # 展平
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss，解决类别不平衡问题。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha: 类别平衡系数，默认 0.25
            gamma: 难易样本调节系数，默认 2.0
            reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) 预测 logits
            targets: (N, H, W) 或 (N, C, H, W)
        """
        if inputs.size(1) > 1:  # 多分类
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
        else:  # 二分类
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            pt = torch.exp(-bce_loss)
            ce_loss = bce_loss

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
