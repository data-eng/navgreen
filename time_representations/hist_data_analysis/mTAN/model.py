import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

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
                                      nn.Linear(input_dim * num_heads, 5)])

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


class MtanClassif(nn.Module):

    def __init__(self, input_dim, query, device, embed_time, num_heads, time_representation):
        super(MtanClassif, self).__init__()
        assert embed_time % num_heads == 0
        self.device = device
        self.embed_time = embed_time
        self.query = query
        self.att = MultiTimeAttention(2 * input_dim, embed_time, num_heads)

        self.time_repr = torch.sin if time_representation == 'sin' else self.triangular_pulse2
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        self.counter = 0

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = self.time_repr(self.periodic(tt))
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

        return out


    def triangular_pulse1(self, linear_tensor, a=1.0, c=0.0):

        # Calculate the distance from the center
        distance = torch.abs(linear_tensor - c)
        # Calculate the triangular pulse values
        pulse = torch.where(distance <= a, 1 - (distance / a), torch.tensor(0.0))
        return pulse

    def triangular_pulse(self, linear_tensor):

        # Max PYRANOMETER value
        max_pos = 11

        pulse = torch.zeros(linear_tensor.shape).to(self.device)

        first_half_of_the_day = torch.abs(linear_tensor[:, :max_pos, :])
        second_half_of_the_day = torch.abs(linear_tensor[:, max_pos + 1:, :])

        diff_tensor_first = torch.diff(first_half_of_the_day, dim=1)
        diff_tensor_second = torch.diff(second_half_of_the_day, dim=1)

        # Define the start and end positions for the triangular function
        start_pos = torch.argmax(diff_tensor_first, dim=1)
        end_pos = torch.argmax(diff_tensor_second, dim=1) + 12

        start_pos = torch.full(start_pos.shape, 9)
        end_pos = torch.full(end_pos.shape, 17)

        # Create the ascending part of the triangle
        for i in range(0, max_pos+1):
            pulse[:, i, :] = torch.clamp((i - start_pos) / (max_pos - start_pos), min=0.0)

        # Create the descending part of the triangle
        for i in range(max_pos+1, pulse.shape[1]):
            pulse[:, i, :] = torch.clamp((end_pos - i) / (end_pos - max_pos), min=0.0)

        if self.counter % 10000 == 0:
            x_values = range(1, 25)
            '''for j in range(pulse.shape[2]):
                day_tensor = pulse[0, :, j]
                day_array = day_tensor.cpu().numpy()

                # Plot the data
                plt.plot(x_values, day_array, marker='o')
                plt.xlabel('Hour in the day')
                plt.ylabel('Time representation values')
                plt.title(f'Plot of mean time feature {j}')'''

            day_tensor = pulse[0, :, 0]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            day_tensor = pulse[0, :, 5]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            day_tensor = pulse[0, :, 21]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            plt.xlabel('Hour in the day')
            plt.ylabel('Time representation values')
            plt.title(f'Plot of mean time feature {0}')

            plt.show()
        self.counter += 1

        return pulse

    def triangular_pulse4(self, linear_tensor):

        # Max PYRANOMETER value
        max_pos = 12

        pulse = torch.zeros(linear_tensor.shape).to(self.device)

        first_half_of_the_day = torch.abs(linear_tensor[:, :max_pos, :])
        second_half_of_the_day = torch.abs(linear_tensor[:, max_pos + 1:, :])

        # Define the start and end positions for the triangular function
        start_pos = torch.argmax(first_half_of_the_day, dim=1)
        end_pos = torch.argmin(second_half_of_the_day, dim=1) + max_pos + 1

        # Create the ascending part of the triangle
        for i in range(0, max_pos+1):
            pulse[:, i, :] = torch.clamp((i - start_pos) / (max_pos - start_pos), min=0.0)

        # Create the descending part of the triangle
        for i in range(max_pos+1, pulse.shape[1]):
            pulse[:, i, :] = torch.clamp((end_pos - i) / (end_pos - max_pos), min=0.0)

        if self.counter % 1000000 == 0:
            x_values = range(1, 25)

            day_tensor = pulse[0, :, 0]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            day_tensor = pulse[0, :, 5]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            plt.xlabel('Hour in the day')
            plt.ylabel('Time representation values')
            plt.title(f'Plot of mean time feature {0}')

            plt.show()
        self.counter += 1

        return pulse

    def triangular_pulse3(self, linear_tensor):

        # Max PYRANOMETER value
        max_pos = 12

        pulse = torch.zeros(linear_tensor.shape).to(self.device)

        first_half_of_the_day = torch.abs(linear_tensor[:, :max_pos, :])
        second_half_of_the_day = torch.abs(linear_tensor[:, max_pos + 1:, :])

        # Define the start and end positions for the triangular function
        start_pos = torch.argmax(first_half_of_the_day, dim=2)
        end_pos = torch.argmax(second_half_of_the_day, dim=2) + max_pos + 1

        # Create the ascending part of the triangle
        for i in range(0, max_pos+1):
            pulse[:, i, :max_pos] = torch.clamp((i - start_pos) / (max_pos - start_pos), min=0.0)

        # Create the descending part of the triangle
        for i in range(max_pos+1, pulse.shape[1]):
            pulse[:, i, max_pos:] = torch.clamp((end_pos - i) / (end_pos - max_pos), min=0.0)

        if self.counter % 1000000 == 0:
            x_values = range(1, 25)

            day_tensor = pulse[0, :, 0]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            day_tensor = pulse[0, :, 5]
            day_array = day_tensor.cpu().numpy()
            plt.plot(x_values, day_array, marker='o')

            plt.xlabel('Hour in the day')
            plt.ylabel('Time representation values')
            plt.title(f'Plot of mean time feature {0}')

            plt.show()
        self.counter += 1

        return pulse

    def triangular_pulse2(self, linear_tensor):

        max_pos = 11

        # Calculate the distance from the value of the midday of each day
        distance = torch.abs(linear_tensor - linear_tensor[:, max_pos, :].unsqueeze(1))
        a = torch.quantile(distance, 0.25)
        # Calculate the triangular pulse values
        condition = distance <= a
        pulse = torch.where(condition, 1 - (distance / a), torch.tensor(0.0)).to(self.device)

        '''
        if self.counter % 1000 == 0:
            x_values = range(1, 25)
            for j in range(pulse.shape[2]):
                day_tensor = pulse[0, :, j]
                day_array = day_tensor.cpu().numpy()

                # Plot the data
                plt.plot(x_values, day_array, marker='o')
                plt.xlabel('Hour in the day')
                plt.ylabel('Time representation values')
                plt.title(f'Plot of mean time feature {j}')

        day_tensor = pulse[0, :, 0]
        day_array = day_tensor.detach().cpu().numpy()
        plt.plot(x_values, day_array, marker='o')

        day_tensor = pulse[0, :, 5]
        day_array = day_tensor.detach().cpu().numpy()
        plt.plot(x_values, day_array, marker='o')

        day_tensor = pulse[0, :, 8]
        day_array = day_tensor.detach().cpu().numpy()
        plt.plot(x_values, day_array, marker='o')

        day_tensor = pulse[0, :, 10]
        day_array = day_tensor.detach().cpu().numpy()
        plt.plot(x_values, day_array, marker='o')

        day_tensor = pulse[0, :, 20]
        day_array = day_tensor.detach().cpu().numpy()
        plt.plot(x_values, day_array, marker='o')

        plt.xlabel('Hour in the day')
        plt.ylabel('Time representation values')
        plt.title(f'Plot of mean time feature {0}')

        plt.show()
        
        self.counter += 1
        '''

        return pulse