import math
import torch
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, scaling=10000.0, increment=2):
        """
        Initializes the positional encoding module:

        - Positional indices represent positions within the sequences.
        - Sine and cosine functions of the scaled positional indices define the positional encodings.

        :param scaling: scaling factor
        :param increment: increment factor
        """
        super(PosEncoding, self).__init__() 
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.increment = increment
        self.dropout = nn.Dropout(p=0.1)

    @property
    def div_term(self):
        """
        Computes div_term, used to modulate the frequency of the sine and cosine pe representations.

        :return: tensor containing scaled indices (div_term)
        """
        indices = torch.arange(0, self.d_model, step=2, dtype=torch.float)
        w_i = -math.log(self.scaling) / self.d_model
        return torch.exp(w_i * indices)
    
    def encoding(self, arg):
        _, _, num_feats = arg

        pe = torch.zeros(self.max_seq_len, num_feats, self.d_model)

        pos = torch.arange(0, self.max_seq_len).unsqueeze(1).unsqueeze(-1)
        scaled_pos = self.div_term * pos

        pe[:, :, 0::2] = torch.sin(scaled_pos)
        pe[:, :, 1::2] = torch.cos(scaled_pos)

        return pe.unsqueeze(1)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        :param x: input tensor
        :return: encoded tensor
        """
        pe = self.encoding(arg=x.size())  
        return x.unsqueeze(-1) + pe[:x.size(0), :]

        # return: torch.Size([48, 120, 2, 250]) ~ (sequence length, batch size, feature size, d_model)

class Transformer(nn.Module):
    def __init__(self, output_size=1, d_model=250, nhead=10, num_layers=1, dim_feedforward=768, dropout=0.1):
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
        print("####################### pos_encoder: ", x.size())  # should be (seq_len, batch_size, num_feats, d_model)

        x = self.encoder(src=x)
        print("####################### encoder: ", x.size()) # should be (seq_len, batch_size, num_feats, d_model)

        x = self.decoder(input=x)  # should be (seq_len, batch_size, num_feats, output_size)
        print("####################### decoder: ", x.size())
        return x