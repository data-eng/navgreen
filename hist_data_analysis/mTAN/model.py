import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class create_regressor(nn.Module):
    def __init__(self, latent_dim, nhidden, output_dim=2):
        super(create_regressor, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, output_dim))

    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.regressor(out.squeeze(0))

class MultiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden, embed_time, num_heads):
        super(MultiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, nhidden)])


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        # Compute 'Scaled Dot Product Attention'
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MtanGruRegr(nn.Module):

    def __init__(self, input_dim, query, device, nhidden, embed_time, num_heads ):
        super(MtanGruRegr, self).__init__()
        assert embed_time % num_heads == 0
        self.device = device
        self.embed_time = embed_time
        self.nhidden = nhidden
        self.query = query
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        # self.att = MultiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2))
        self.enc = nn.GRU(nhidden, nhidden)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps, mask):
        # x is: [batch_size, sequence_length, input_size]
        x = torch.cat((x, mask), 2)
        mask = torch.cat((mask, mask), 2)
        time_steps = time_steps.to(self.device)

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))


class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, device, nhidden, embed_time, num_heads, latent_dim=2):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
            # nn.Linear(50, latent_dim))
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        #print(f'tt: {tt}')
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        #print(f'out2: {out2}')
        return torch.cat([out1, out2], -1)


    def forward(self, x, time_steps, mask):
        time_steps = time_steps.to(self.device)
        x = torch.cat((x, mask), 2)
        mask = torch.cat((mask, mask), 2)
        #print(f'time_steps: {time_steps}')
        key = self.learn_time_embedding(time_steps).to(self.device)
        # print(f'key: {key}')
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out



class dec_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, device, nhidden, embed_time, num_heads, latent_dim=2):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = MultiTimeAttention(2 * nhidden, 2 * nhidden, embed_time, num_heads)
        # self.att = MultiTimeAttention(2 * nhidden, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50),
            #nn.Linear(nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.to(self.device)

        query = self.learn_time_embedding(time_steps).to(self.device)
        key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out
