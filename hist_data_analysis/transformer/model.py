import math
import torch
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, d_model, max_timesteps=1000, scaling=10000.0, increment=2):
        """
        Initializes the positional encoding module:

        - Positional indices represent positions within the sequences.
        - Sine and cosine functions of the scaled positional indices define the positional encodings.

        :param d_model: the size of the embeddings that represent the inputs/outputs of the transformer (dimensionality)
        :param max_timesteps: the maximum number of timesteps in the input sequences
        :param scaling: scaling factor
        :param increment: increment factor
        """
        super(PosEncoding, self).__init__() 
        self.d_model = d_model
        self.scaling = scaling
        self.increment = increment
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_timesteps, 1, self.d_model)

        pos = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        scaled_pos = self.div_term * pos

        pe[:, 0, 0::2] = torch.sin(scaled_pos)
        pe[:, 0, 1::2] = torch.cos(scaled_pos)
    
        self.register_buffer('pe', pe)

    @property
    def div_term(self):
        """
        Computes div_term, used to modulate the frequency of the sine and cosine pe representations.

        :return: tensor containing scaled indices (div_term)
        """
        indices = torch.arange(0, self.d_model, self.increment).float()
        w_i = -math.log(self.scaling) / self.d_model
        return torch.exp(w_i * indices)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        :param x: input tensor
        :return: encoded tensor
        """
        return x + self.pe[:x.size(0), :]        

class Transformer(nn.Module):
    def __init__(self, output_size=1, d_model=192, nhead=10, num_layers=1, dim_feedforward=768, dropout=0.1):
        super(Transformer, self).__init__()

        self.pos_encoder = PosEncoding(d_model)
        
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)
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

        - Adds positional encodings to the input tensor.
        - Passes the encoded input through the transformer encoder layer.
        - Passes the output of the encoder through the decoder linear layer.

        :param x: input tensor
        :return: output tensor after passing through the transformer model
        """
        x = x.permute(1, 0, 2)  # (sequence length, batch size, feature size)
        x = self.pos_encoder(x)
        x = self.encoder(src=x)
        x = x.permute(1, 0, 2)  # (batch size, sequence length, feature size)
        x = self.decoder(input=x)
        return x