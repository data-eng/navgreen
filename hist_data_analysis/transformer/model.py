import math
import torch
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, batch_size, max_timesteps=5000, scaling=10000.0, increment=2):
        """
        Initializes the positional encoding module:

        - Positional indices represent positions within the sequences.
        - Sine and cosine functions of the scaled positional indices define the positional encodings.

        :param batch_size: the batch size of the input sequences
        :param max_timesteps: the maximum number of timesteps in the input sequences
        :param scaling: scaling factor
        :param increment: increment factor
        """
        super(PosEncoding, self).__init__() 
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps  
        self.scaling = scaling
        self.increment = increment
        
        pe = torch.zeros(self.max_timesteps, self.batch_size)

        pos = torch.arange(0, self.max_timesteps, dtype=torch.float).unsqueeze(1)
        scaled_pos = self.scaled_indices * pos

        pe[:, 0::2] = torch.sin(scaled_pos)
        pe[:, 1::2] = torch.cos(scaled_pos)
        pe = pe.unsqueeze(0).transpose(0, 1)
    
        self.register_buffer('pe', self.pe)

    @property
    def scaled_indices(self):
        """
        Computes scaled indices, used to modulate the frequency of the sine and cosine pe representations.

        :return: tensor containing scaled indices
        """
        indices = torch.arange(0, self.batch_size, self.increment).float()
        w_i = -math.log(self.scaling) / self.batch_size
        return torch.exp(w_i * indices)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        :param x: input tensor
        :return: encoded tensor
        """
        return x + self.pe[:x.size(0), :]        
    
class Transformer(nn.Module):
    def __init__(self, num_features=2, num_layers=1, dropout=0.1):
        super(Transformer, self).__init__()

        self.mask = None
        self.pos_encoder = PosEncoding(num_features)
        self.layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=10, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(num_features, 1)
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
        Forward pass of the Transformer model:

        - Adds positional encodings to the input tensor.
        - Passes the encoded input through the transformer encoder layer.
        - Passes the output of the encoder through the decoder linear layer.

        :param x: input tensor
        :return: output tensor after passing through the transformer model
        """
        timesteps = len(x)
        if self.mask is None or self.mask.size(0) != timesteps:
            device = x.device
            mask = self.generate_mask(size=timesteps).to(device)
            self.mask = mask

        x = self.pos_encoder(x)
        x = self.encoder(src=x, mask=self.mask)
        x = self.decoder(input=x)
        return x

    def generate_mask(self, size):
        """
        Generates a square subsequent mask so that each timestep can only attend to previous
        timesteps and not to future ones.
        
        :param size: size of the sequence (number of timesteps)
        :return: square subsequent mask tensor
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask