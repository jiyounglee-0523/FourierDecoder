import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from utils.loss import kl_divergence, log_normal_pdf, normal_kl
from models.encoder import RNNODEEncoder

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



class NP_Trans_Encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, deterministic):
        super(NP_Trans_Encoder, self).__init__()
        self.deterministic = deterministic

        self.input_projection = nn.Linear(2, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim, dropout=0.1)
        self.model = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
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
            return self.output_fc(output.permute(1, 0, 2))  # (B, S, E)  확인해보기
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


class NP_linear_encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, deterministic):
        super(NP_linear_encoder, self).__init__()
        self.deterministic = deterministic

        self.input_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim))

        self.output_fc = nn.Linear(hidden_dim, output_dim)
        if not self.deterministic:
            self.linear1 = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_dim, 2*hidden_dim))

    def forward(self, content_x, content_y):
        # Concat content_x (B, S, 1) and content_y (B, S, 1)
        input = torch.cat((content_x, content_y), dim=-1)   # (B, S, 2)
        output = self.input_projection(input)   # (B, S, hidden_dim)

        if self.deterministic:
            return self.output_fc(output)  # (B, S, output_dim)
        else:
            # Aggregator: take the mean over all points
            output = output.mean(dim=1)    # add according to the sequence
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



class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, 2*coeffs_size)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(2*coeffs_size, 2*coeffs_size)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(2*coeffs_size, coeffs_size)

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)



def FourierExpansion(n_range, s):
    s_n_range = s * n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis


class FNP_Decoder(nn.Module):
    def __init__(self, args):
        super(FNP_Decoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.coeffs_size = args.in_features*args.out_features*args.n_harmonics*args.n_eig

        self.coeff_generator = CoeffDecoder(args.latent_dimension, coeffs_size=self.coeffs_size)


    def forward(self, target_x, r):
        # target_x  (B, S, 1), r (B, E)
        coeffs = self.coeff_generator(r)  # (B, C)

        # make cos / sin matrix
        cos_x = torch.cos(target_x * 2 * math.pi)  # (B, S, 1)
        sin_x = torch.sin(target_x * 2 * math.pi)  # (B, S, 1)
        for i in range(self.n_harmonics - 1):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * (i+2) * math.pi)), dim=-1)   # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * (i+2) * math.pi)), dim=-1)   # (B, S, H)

        cos_x = torch.mul(cos_x, coeffs[:, :int(self.coeffs_size/2)].unsqueeze(1))
        sin_x = torch.mul(sin_x, coeffs[:, int(self.coeffs_size/2):].unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)  # (B, S)
        return cos_x + sin_x   # (B, S)


# Changed from Transformer Encoder to Linear Encoder
class FNP(nn.Module):
    def __init__(self, args):
        super(FNP, self).__init__()
        self.deterministic_encoder = NP_linear_encoder(args.encoder_hidden_dim, args.encoder_output_dim, deterministic=True)
        self.decoder = FNP_Decoder(args)

    def forward(self, context_x, context_y, target_x, target_y=None):
        """
        every element shape of (B, S)
        """
        num_targets = target_x.size(1)
        context_x = context_x.unsqueeze(-1) ; context_y = context_y.unsqueeze(-1)
        target_x = target_x.unsqueeze(-1) #; target_y = target_y.unsqueeze(-1) if target_y is not None else target_y

        r = self.deterministic_encoder(context_x, context_y)   # (B, S, E)

        mu = self.decoder(target_x, r.mean(1))

        # for Format
        sigma = None
        kl_loss = None
        mse_loss = None

        if target_y is not None:
            loss = nn.MSELoss()(mu, target_y)
        else:
            loss = None

        return mu, sigma, mse_loss, kl_loss, loss


class ConditionalFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalFNP, self).__init__()
        self.dataset_type = args.dataset_type

        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.encoder_output_dim, rnn_hidden_dim=args.encoder_hidden_dim)
        else:
            raise NotImplementedError

        self.decoder = FNP_Decoder(args)
        self.label_num = {'sin': 4,
                          'ECG': 5,
                          'NSynth': 20}

    def forward(self, t, x, label):
        t = torch.squeeze(t[0]).cuda()
        B = x.size(0)

        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(0), 150, replace=False)))[0]
            t = t[sample_idxs]  # (300)
            x = x[:, sample_idxs]

        z, qz0_mean, qz0_logvar = self.encoder(x, label, span=t)
        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1)

        # concat label information
        label_embed = torch.zeros(B, self.label_num[self.dataset_type]).cuda()
        label_embed[range(B), label] = 1

        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t, z)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        return mse_loss, kl_loss










