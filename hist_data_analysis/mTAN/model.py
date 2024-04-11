import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTimeAttention(nn.Module):

    def __init__(self, input_dim, embed_time, num_heads):
        super(MultiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, embed_time)])

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

    def forward(self, query, key, value, mask, dropout=None):
        # Compute 'Scaled Dot Product Attention'
        batch, seq_len, dim = value.size()
        mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        x = self.linears[-1](x)
        return x


class MtanRNNRegr(nn.Module):

    def __init__(self, input_dim, query, device, embed_time, num_heads, init_embed=8):
        super(MtanRNNRegr, self).__init__()
        assert embed_time % num_heads == 0
        self.device = device
        self.embed_time = embed_time
        self.query = query
        self.att = MultiTimeAttention(2 * input_dim, embed_time, num_heads)
        # self.att = MultiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.regressor = nn.Sequential(
            nn.Linear(embed_time, 128),
            nn.Linear(128, 128),
            nn.Linear(128, embed_time),
            nn.Linear(embed_time, init_embed))
        self.enc = nn.RNN(embed_time, embed_time, batch_first=True)

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
        _, out = self.enc(out)
        out = out.squeeze()
        return self.regressor(out.squeeze(0))


class MtanRNNClassif(nn.Module):

    def __init__(self, input_dim, query, device, embed_time, num_heads, init_embed=8, out_classes=5):
        super(MtanRNNClassif, self).__init__()
        assert embed_time % num_heads == 0
        self.device = device
        self.embed_time = embed_time
        self.query = query
        self.att = MultiTimeAttention(2 * input_dim, embed_time, num_heads)
        # self.att = MultiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Linear(embed_time, out_classes)
        self.classifiers = nn.ModuleList([self.classifier for _ in range(embed_time)])

        self.enc = nn.RNN(embed_time, embed_time, batch_first=True)

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
        _, out = self.enc(out)
        out = out.squeeze()
        #out = self.classifier(out.squeeze(0))
        #out = [classifier(out[:, i]) for i, classifier in enumerate(self.classifiers)]
        out = [classifier(out) for i, classifier in enumerate(self.classifiers)]
        out = torch.stack(out, dim=1)  # Stack the outputs along the class dimension
        return out
