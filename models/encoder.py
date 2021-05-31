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
    def __init__(self, input_dim, output_dim, rnn_hidden_dim, last_output=True):
        super().__init__()
        if input_dim != 1:
            self.embedding = nn.Linear(1, input_dim)
        self.jump = nn.RNNCell(input_dim, rnn_hidden_dim)
        self.f = ODEFunc(rnn_hidden_dim)
        self.out = nn.Linear(rnn_hidden_dim, output_dim)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.last_output = last_output

    def forward(self, x, span):
        # x shape should be (batch_size, seq_len, dimension)
        x = self.embedding(x)  #(B, S, E)
        h = self._init_latent(x)
        Y = []
        for idx in range(x.size(1)-1):
            obs = x[:, idx, :]
            h = self.jump(obs, h)
            t_span = torch.Tensor([span[idx], span[idx+1]])
            h = odeint(self.f, h, t_span, method='rk4')[-1]
            Y.append(self.out(h)[None])

        # for t in range(x.size(1)):
        #     obs = x[:, t, :]
        #     h = self.jump(obs, h)
        #     h = self.flow(h)
        #     Y.append(self.out(h)[None])

        Y = torch.cat(Y)
        return Y[-1] if self.last_output else Y

    def _init_latent(self, x):
        return torch.zeros((x.shape[0], self.rnn_hidden_dim)).cuda()


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
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.pos_encoder = PositionalEncoding(args.encoder_embedding_dim, args.data_length, args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)

        self.output_fc = nn.Linear(args.encoder_hidden_dim, args.encoder_output_dim)

    def forward(self, x, span):
        # x shape of B, S, 1
        B = x.size(0)
        x = self.embedding(x)   # (B, S, E)
        x = x.permute(1, 0, 2)  # (S, B, E)
        x = self.pos_encoder(x)

        output = self.transformer_encoder(src=x)  # (S, B, E)
        output = output.sum(0)  # (B, E)
        output = self.output_fc(output)
        return output


