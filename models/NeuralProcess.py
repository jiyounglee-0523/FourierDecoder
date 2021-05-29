import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from utils.loss import kl_divergence, log_normal_pdf

## Attentive Neural Process
# reference : https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
# reference : https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py

class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_per_attn):
        super(MultiheadAttention, self).__init__()
        self.num_hidden_per_attn = num_hidden_per_attn
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1,2))
        attn = attn / math.sqrt(self.num_hidden_per_attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        result = torch.bmm(attn, value)
        return result, attn



class CrossAttention(nn.Module):
    def __init__(self, num_hidden, num_attn):
        super(CrossAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = int(num_hidden / num_attn)
        self.num_attn = num_attn

        self.key = nn.Linear(1, num_hidden)
        self.query = nn.Linear(1, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

    def forward(self, query, key, value):
        batch_size = key.size(0)  # context B
        seq_k = key.size(1)       # context S
        seq_q = query.size(1)     # target S

        # Multihead attention
        key = self.key(key).view(batch_size, seq_k, self.num_attn, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.num_attn, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.num_attn, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attn = self.multihead(query, key, value)

        # Concate all multihead context vector
        #result = result.view(self.num_attn, batch_size, seq_q, self.num_hidden_per_attn)
        #result = result.permute(1, 2, 0, 3).view(batch_size, seq_q, -1)
        return result



class NP_Encoder(nn.Module):
    def __init__(self, hidden_dim, deterministic):
        super(NP_Encoder, self).__init__()
        self.deterministic = deterministic

        self.input_projection = nn.Linear(2, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim, dropout=0.1)
        self.model = nn.TransformerEncoder(encoder_layers, num_layers=2)

        if not deterministic:
            self.linear1 = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_dim, 2*hidden_dim))


    def forward(self, content_x, content_y):
        # Concat content_x (B, S, 1) and content_y (B, S , 1)
        input = torch.cat((content_x, content_y), dim=-1)  # (B, S, 2)
        input = self.input_projection(input)
        input = input.permute(1, 0, 2)   # (S, B, E)

        output = self.model(src=input)  # (S, B, E)

        if self.deterministic:
            return output.permute(1, 0, 2)  # (B, S, E)  확인해보기
        else:
            # Aggregator: take the mean over all points
            output = output.mean(dim=0) # (B, hidden_dim)
            # Apply further linear layer to output latent mu and log sigma
            z = self.linear1(output)
            z0, qz0_mean, qz0_logvar = self.reparameterization(z)
            return z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        latent_dim = z.shape[1] // 2
        qz0_mean, qz0_logvar = z[:, :latent_dim], z[:, latent_dim:]
        qz0_logvar = 0.1 + 0.9 * torch.sigmoid(qz0_logvar)
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar


class NP_Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(NP_Decoder, self).__init__()
        # 128*2 + 2
        self.target_projection = nn.Linear(1, hidden_dim)
        self.embedding = nn.Linear(hidden_dim*3, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim)
        self.model = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, target_x, attended_r, z0):
        target_x = self.target_projection(target_x)
        input = torch.cat((target_x, attended_r, z0), dim=-1) # (B, S, E)
        input = self.embedding(input)
        input = input.permute(1, 0, 2)

        hidden = self.model(src=input)   # (S, B, E)
        hidden = self.output_fc(hidden.permute(1, 0, 2))   # (B, S, 2)

        mu = hidden[..., 0]
        sigma = 0.1 + 0.9*F.softplus(hidden[..., 1])
        return mu, sigma


class AttentiveNP(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentiveNP, self).__init__()
        self.deterministic_encoder = NP_Encoder(hidden_dim, deterministic=True)
        self.latent_encoder = NP_Encoder(hidden_dim, deterministic=False)
        self.attention = CrossAttention(hidden_dim, num_attn=1)
        self.decoder = NP_Decoder(hidden_dim)

    def forward(self, content_x, content_y, target_x, target_y=None):
        """
        every element shape (B, S)
        """
        num_targets = target_x.size(1)
        content_x = content_x.unsqueeze(-1) ; content_y = content_y.unsqueeze(-1)
        target_x = target_x.unsqueeze(-1) ; target_y = target_y.unsqueeze(-1) if target_y is not None else target_y
        prior, prior_mu, prior_var = self.latent_encoder(content_x, content_y)

        # For training
        if target_y is not None:
            posterior, posterior_mu, posterior_var = self.latent_encoder(target_x, target_y)
            z = posterior

        # For Test
        else:
            z = prior # (B, 2E)

        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # (B, S, E)
        r = self.deterministic_encoder(content_x, content_y)

        ## attention
        attn_result = self.attention(target_x, content_x, r)

        # Decoder
        mu, sigma = self.decoder(target_x, attn_result, z)

        # For training
        if target_y is not None:
            # get loss
            mse_loss = log_normal_pdf(target_y.squeeze(-1), mu, sigma).mean()
            kl_loss = kl_divergence(prior_mu, prior_var, posterior_mu, posterior_var)
            loss = -mse_loss + kl_loss

        else:
            mse_loss = None
            kl_loss = None
            loss = None

        return mu, sigma, mse_loss, kl_loss, loss











