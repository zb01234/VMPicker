# Code for dice loss

import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001):
        # flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        """
        Focal Loss with adjustable alpha and gamma.
        Args:
            alpha (float): balancing factor, default is 1.
            gamma (float): focusing parameter, default is 2.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute the binary cross-entropy loss
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt is the probability of the true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.65, beta=0.35, smooth=1e-6):
        """
        Tversky Loss function for binary segmentation tasks.
        Args:
            alpha (float): weight of false positives, default is 0.5.
            beta (float): weight of false negatives, default is 0.5.
            smooth (float): smoothing constant to avoid division by zero, default is 1e-6.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - Tversky_index
