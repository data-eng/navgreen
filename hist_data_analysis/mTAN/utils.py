import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self, sequence_length):
        super(MaskedMSELoss, self).__init__()
        self.sequence_length = sequence_length

    def forward(self, pred, true, mask):
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length).mean(dim=2)
        # Compute element-wise squared difference
        squared_diff = (pred - true) ** 2
        # Apply mask to ignore certain elements
        mask = mask.float()
        loss = squared_diff * mask
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, sequence_length):
        super(MaskedSmoothL1Loss, self).__init__()
        self.sequence_length = sequence_length

    def forward(self, pred, true, mask):
        '''
        if pred.dim() == 1: pred = pred.unsqueeze(0)
        pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length).mean(dim=2)
        '''
        # Compute element-wise absolute difference
        abs_diff = torch.abs(pred - true)
        # Apply mask to ignore certain elements
        mask = mask.float()
        abs_diff = abs_diff * mask
        # Compute loss
        loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero

        return loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, sequence_length, weights):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.sequence_length = sequence_length
        # Calculate the inverse class frequencies
        if weights is not None:
            w = 1.0 / weights
            self.weights = w / w.sum()
        else: self.weights = None
        print(f'weights is {self.weights}')

    def forward(self, pred, true, mask):
        if pred.dim() == 2: pred = pred.permute(1, 0).unsqueeze(0)
        # pred = pred.view(pred.shape[0], self.sequence_length, pred.shape[1] // self.sequence_length,
        #                  pred.shape[2]).mean(dim=2)

        true = true[mask == 1].long()
        pred = pred[mask == 1]

        loss = F.cross_entropy(pred, true, weight=self.weights)

        return loss
