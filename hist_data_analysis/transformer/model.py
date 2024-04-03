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

class PosEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, scaling=10000.0, increment=2):
        """
        Initializes the positional encoding module:

        - Positional indices represent positions within the sequences.
        - Sine and cosine functions of the scaled positional indices define the positional encodings.

        :param d_model: dimensionality of the model
        :param max_seq_len: maximum sequence length
        :param scaling: scaling factor
        :param increment: increment factor
        """
        super(PosEncoding, self).__init__() 
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.increment = increment

    @property
    def div_term(self):
        """
        Computes div_term, used to modulate the frequency of the sine and cosine pe representations.

        :return: tensor containing scaled indices (div_term)
        """
        indices = torch.arange(0, self.d_model, step=2, dtype=torch.float)
        w_i = -math.log(self.scaling) / self.d_model
        return torch.exp(w_i * indices)
    
    @property
    def encoding(self):
        """
        Computes the positional encodings, using sine and cosine functions.

        :return: tensor containing the positional encodings
        """
        pe = torch.zeros(self.max_seq_len, self.d_model)
        logger.debug("pe: %s", pe.size())

        pos = torch.arange(0, self.max_seq_len).unsqueeze(1)
        logger.debug("pos: %s", pos.size())
        logger.debug("div_term: %s", self.div_term.size())

        scaled_pos = self.div_term * pos
        logger.debug("scaled_pos: %s", scaled_pos.size())

        pe[:, 0::2] = torch.sin(scaled_pos)
        pe[:, 1::2] = torch.cos(scaled_pos)

        pe = pe.unsqueeze(0).transpose(0, 1)

        logger.debug("unsq pe: %s", pe.size())

        return pe

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        :param x: input tensor
        :return: encoded tensor
        """
        pe = self.encoding  
        enc_x = x + pe[:x.size(0), :]
        return enc_x

class FeatureFuser(nn.Module):
    def __init__(self, num_feats):
        """
        Initializes the feature fuser network.

        :param num_feats: number of input features
        """
        super(FeatureFuser, self).__init__()
        self.fc = nn.Linear(num_feats, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Combines n input features into 1, using a linear layer followed by a ReLU activation function.

        :param x: n-feature tensor
        :return: 1-feature tensor
        """
        x = self.fc(x)
        x = self.relu(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, in_size=2, out_size=2, d_model=250, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        """
        Initializes a transformer model architecture: [Positional Encoder, Feature Fuser, Transformer Encoder, Linear Decoder]

        :param in_size: X features size
        :param out_size: y features size
        :param d_model: dimensionality of the model
        :param nhead: number of attention heads
        :param num_layers: number of transformer encoder layers
        :param dim_feedforward: dimension of the feedforward network model
        :param dropout: dropout rate
        """
        super(Transformer, self).__init__()

        self.pos_encoder = PosEncoding(d_model)
        self.fuser = FeatureFuser(num_feats=in_size)
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, out_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights and biases of the decoder linear layer:

        - Sets the bias of the decoder linear layer to zero.
        - Initializes the weights with values drawn from a uniform distribution centered around zero.
        """
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        """
        Forward pass of the transformer model:

        - Applies feature fusion to the input tensor, reducing their number from n to 1.
        - Adds positional encodings to the input tensor.
        - Passes the encoded input through the transformer encoder layer.
        - Passes the output of the encoder through the decoder linear layer.

        :param x: input tensor
        :return: output tensor after passing through the transformer model
        """
        x = x.permute(1, 0, 2)
        x = self.fuser(x)
        x = self.pos_encoder(x)
        x = self.encoder(src=x)
        x = self.decoder(input=x)
        x = x.permute(1, 0, 2)
        return x