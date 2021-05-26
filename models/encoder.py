import torch
import torch.nn as nn

import math

from torchdyn.models import NeuralDE

class RNNODEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_hidden_dim, last_output=True):
        super().__init__()
        self.jump = nn.RNNCell(input_dim, rnn_hidden_dim)
        f = nn.Sequential(nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
                          nn.SiLU(),
                          nn.Linear(rnn_hidden_dim, rnn_hidden_dim))
        self.flow = NeuralDE(f)
        self.out = nn.Linear(rnn_hidden_dim, latent_dim)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.last_output = last_output

    def forward(self, x):
        # x shape should be (batch_size, seq_len, dimension)
        h = self._init_latent(x)
        Y = []
        for t in range(x.size(1)):
            obs = x[:, t, :]
            h = self.jump(obs, h)
            h = self.flow(h)
            Y.append(self.out(h)[None])

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

        self.pos_encoder = PositionalEncoding(args.encoder_embedding_dim, args.data_length, args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)

        self.output_fc = nn.Linear(args.encoder_hidden_dim, args.encoder_output_dim)

    def forward(self, x):
        # x shape of B, S, E
        B = x.size(0)

        x = x.permute(1, 0, 2)  # (S, B, E)
        x = self.pos_encoder(x)

        output = self.transformer_encoder(src=x)
        output = output.sum(0)  # (B, E)
        output = self.output_fc(output)
        return output


