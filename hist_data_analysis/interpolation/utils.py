import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, true):
        if pred.dim() == 2: pred = pred.permute(1,0).unsqueeze(0)
        true = true.long()
        loss = [F.cross_entropy(pred[b_sz, :, :], true[b_sz, :], reduction='none') for b_sz in range(true.shape[0])]
        loss = torch.stack(loss, dim=0)
        loss = torch.sum(loss)

        return loss / (loss.shape[0] * loss.shape[1])
