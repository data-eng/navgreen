import math
import torch
import torch.nn as nn

class InterpClassif(nn.Module):

    def __init__(self, dim, init_embed, out_classes=5):
        super(InterpClassif, self).__init__()

        self.enc = nn.RNN(dim, init_embed, batch_first=True)
        self.classifier = nn.Linear(1, out_classes)
        # self.final = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(8, 1), stride=(4, 1), padding=(3, 0))
        #self.final = nn.Linear(embed_time * out_classes, init_embed * out_classes)

    def forward(self, x):
        # x is: [batch_size, sequence_length, input_size]
        _, out = self.enc(x)

        out = out.permute(1, 2, 0)
        out = [self.classifier(out[i, :, :]) for i in range(x.shape[0])]
        out = torch.stack(out, dim=0)  # Stack the outputs along the class dimension

        '''
        if out.dim() == 2:
            out = out.permute(1, 0)
            out = out.unsqueeze(0)

        out = out.unsqueeze(0)
        out = self.final(out.permute(0, 3, 1, 2))
        out = out.squeeze(0).permute(2, 1, 0)
        '''
        '''
        out = out.reshape(out.shape[0], -1)
        if out.dim() == 2: out = out.unsqueeze(1)
        out = self.final(out)
        out = out.reshape(out.shape[0], self.init_embed, -1)
        '''
        return out