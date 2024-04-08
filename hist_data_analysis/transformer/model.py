import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
    
class Transformer(nn.Module):
    def __init__(self, in_size=16, out_size=5, nhead=4, num_layers=4, dim_feedforward=256, dropout=0):
        super(Transformer, self).__init__()

        self.nhead = nhead
        self.layer = nn.TransformerEncoderLayer(d_model=in_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(in_size, out_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights and biases of the decoder linear layer:

        - Sets the bias of the decoder linear layer to zero.
        - Initializes the weights with values drawn from a uniform distribution centered around zero.
        """ 
        self.decoder.bias.data.zero_()
        nn.init.xavier_uniform_(self.decoder.weight.data)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the transformer model:

        - Passes the encoded input through the transformer encoder layer.
        - Passes the output of the encoder through the decoder linear layer.

        :param x: input tensor
        :return: output tensor after passing through the transformer model
        """
        x = x.permute(1, 0, 2)

        if mask is not None:
            x = self.encoder(src=x, src_key_padding_mask=mask)
        else:
            x = self.encoder(src=x)

        x = self.decoder(input=x)
        x = self.softmax(x)

        x = x.permute(1, 0, 2)

        return x