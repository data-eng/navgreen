import torch
import torch.nn as nn



def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


class MaskedMSELoss(nn.Module):
    def __init__(self, mask_value=0.):
        super(MaskedMSELoss, self).__init__()
        self.mask_value = mask_value

    def forward(self, input, target, mask=None):
        # Compute element-wise squared difference
        squared_diff = (input - target) ** 2

        if mask is not None:
            # Apply mask to ignore certain elements
            mask = mask.float()
            squared_diff = squared_diff * mask

        # Compute loss
        loss = squared_diff

        # Optionally, compute the mean loss only over non-masked elements
        if mask is not None:
            loss = loss.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero

        return loss


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, mask_value=0.):
        super(MaskedSmoothL1Loss, self).__init__()
        self.mask_value = mask_value

    def forward(self, input, target, mask):
        # Compute element-wise absolute difference
        abs_diff = torch.abs(input - target)
        # Apply mask to ignore certain elements
        mask = mask.float()
        abs_diff = abs_diff * mask
        # Compute loss
        loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero

        return loss
