import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(CrossEntropyLoss, self).__init__()
        w = 1.0 / weights
        self.weights = w / w.sum()
        print(f'weights is {self.weights}')

    def forward(self, pred, true, mask):
        if pred.dim() == 2: pred = pred.permute(1, 0).unsqueeze(0)
        true = true.long()
        true = true * mask.long()
        loss = [F.cross_entropy(pred[b_sz, :, :], true[b_sz, :], reduction='none', weight=self.weights) for b_sz in range(true.shape[0])]
        loss = torch.stack(loss, dim=0)
        mask = mask.float()
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss