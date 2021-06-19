import torch
import torch.nn as nn

import math

from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, rnn_hidden_dim):
        super().__init__()
        self.nfe = 0   # num forward evaluation

        self.f = nn.Sequential(nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
                          nn.SiLU(),
                          nn.Linear(rnn_hidden_dim, rnn_hidden_dim))

    def reset_nfe(self):
        self.nfe=0

    def forward(self, t, x):
        self.nfe += 1
        return self.f(x)



class RNNODEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_hidden_dim):
        super().__init__()
        self.output_dim = output_dim
        if input_dim != 1:
            self.embedding = nn.Linear(1, input_dim)
        self.jump = nn.RNNCell(input_dim, rnn_hidden_dim)
        self.f = ODEFunc(rnn_hidden_dim)
        self.out = nn.Linear(rnn_hidden_dim, 2*output_dim)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.label_embed = nn.Embedding(4, rnn_hidden_dim)

    def forward(self, x, label, span):
        total_nfe = 0

        # x shape should be (batch_size, seq_len, dimension)
        x = self.embedding(x)  #(B, S, E)
        h = self._init_hidden(label)
        Y = []
        for idx in range(x.size(1)-1):
            obs = x[:, idx, :]
            h = self.jump(obs, h)
            t_span = torch.Tensor([span[idx], span[idx+1]])
            h = odeint(self.f, h, t_span, method='rk4')[-1]
            Y.append(self.out(h)[None])
            total_nfe += self.f.nfe
            self.f.reset_nfe()


        # for t in range(x.size(1)):
        #     obs = x[:, t, :]
        #     h = self.jump(obs, h)
        #     h = self.flow(h)
        #     Y.append(self.out(h)[None])

        Y = torch.cat(Y)
        print(f'total nfe: {total_nfe}')
        output = Y[-1]
        z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return z0, qz0_mean, qz0_logvar

    def _init_hidden(self, label):
        return self.label_embed(label).cuda()    # (B, H)
        # return torch.zeros((x.shape[0], self.rnn_hidden_dim)).cuda()

    def reparameterization(self, z):
        qz0_mean = z[:, :self.output_dim]
        qz0_logvar = z[:, self.output_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar







class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (S, B, E)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = args.latent_dimension
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)
        #self.pos_encoder = PositionalEncoding(args.encoder_embedding_dim, args.data_length, args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim +1, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)

        self.output_fc = nn.Linear(args.encoder_embedding_dim+1, 2*args.latent_dimension)

    def forward(self, x, label, span):
        # x shape of (B, S, 1), span shape of (S)
        B = x.size(0) ; S = span.size(0)
        x = self.embedding(x) # (B, S, E)
        x = torch.cat((x, span.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)), dim=-1)   # (B, S, E+1)
        x = x.permute(1, 0, 2)  # (S, B, E)
        #x = self.pos_encoder(x)

        output = self.transformer_encoder(src=x)  # (S, B, E)
        output = self.output_fc(output) # (S, B, 2*E)
        output = output.mean(0)  # (B, 2*E)

        z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar


