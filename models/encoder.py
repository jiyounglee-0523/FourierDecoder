import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Embedding(max_len, 128, _weight=pe)
        # self.register_buffer('pe', pe)   # shape of (S, 1, E)

    def forward(self, index):
        # x shape: (S, B, E)
        return self.pe(index)
        # return self.pe[:index.size(0), :]
        # return self.dropout(x)


class UnconditionalConvEncoder(nn.Module):
    def __init__(self, args):
        super(UnconditionalConvEncoder, self).__init__()
        self.num_label = args.num_label

        layers = []
        layers.append(nn.Conv1d(in_channels=2+args.num_label, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
        # layers.append(nn.Conv1d(in_channels=2, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
        layers.append(nn.MaxPool1d(kernel_size=2))

        for i in range(args.encoder_blocks):
            layers.append(nn.SiLU())
            layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
            layers.append(nn.MaxPool1d(kernel_size=2))

        layers.append(nn.SiLU())
        layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.latent_dimension, kernel_size=3, stride=1, dilation=1))

        self.model = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_mu = nn.Linear(args.latent_dimension, args.latent_dimension)
        self.latent_sigma = nn.Linear(args.latent_dimension, args.latent_dimension)

    def forward(self, x, label, span):
        # x (B, S, 1)  label (B, num_label)  span (B, S)
        B, S, _ = x.size()

        # span concat
        span = span.unsqueeze(-1)
        label = torch.broadcast_to(label.unsqueeze(1), (B, S, self.num_label))

        input_pairs = torch.cat((x, span, label), dim=-1)  # (B, S, 1+num_label)
        # input_pairs = torch.cat((x, label), dim=-1)
        # input_pairs = torch.cat((x, span), dim=-1)  # (B, S, 2)
        output = self.model(input_pairs.permute(0, 2, 1))  # (B, E, S)
        output = self.glob_pool(output).squeeze(-1)  # (B, E)
        z0, z_dist = self.reparameterization(output)
        # return output, output, 0, 0
        return output, z0, z_dist

    def reparameterization(self, z):
        mean = self.latent_mu(z)
        std = self.latent_sigma(z)
        z_dist = Normal(mean, nn.functional.softplus(std))
        z0 = z_dist.rsample()
        return z0, z_dist


class PositionalConvEncoder(nn.Module):
    def __init__(self, args):
        super(PositionalConvEncoder, self).__init__()
        self.num_label = args.num_label
        self.pos_encoder = PositionalEncoding(128, 500, dropout=args.dropout)

        layers = []
        layers.append(
            nn.Conv1d(in_channels=1+128 + args.num_label, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1,
                      dilation=1))
        # layers.append(nn.Conv1d(in_channels=2, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
        layers.append(nn.MaxPool1d(kernel_size=2))

        for i in range(args.encoder_blocks):
            layers.append(nn.SiLU())
            layers.append(
                nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.encoder_hidden_dim, kernel_size=3,
                          stride=1, dilation=1))
            layers.append(nn.MaxPool1d(kernel_size=2))

        layers.append(nn.SiLU())
        layers.append(
            nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.latent_dimension, kernel_size=3, stride=1,
                      dilation=1))

        self.model = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_mu = nn.Linear(args.latent_dimension, args.latent_dimension)
        self.latent_sigma = nn.Linear(args.latent_dimension, args.latent_dimension)


    def forward(self, x, label, span, index):
        # x (B, S, 1)  label (B, num_label)   span (B, S)   index (B, 250)
        B, S, _ = x.size()

        with torch.no_grad():
            span = self.pos_encoder(index)  # (B, 250, 128)
        label = torch.broadcast_to(label.unsqueeze(1), (B, S, self.num_label))

        input_pairs = torch.cat((x, span, label), dim=-1)
        output = self.model(input_pairs.permute(0, 2, 1))
        output = self.glob_pool(output).squeeze(-1)
        z0, z_dist = self.reparameterization(output)
        return output, z0, z_dist

    def reparameterization(self, z):
        mean = self.latent_mu(z)
        std = self.latent_sigma(z)
        z_dist = Normal(mean, nn.functional.softplus(std))
        z0 = z_dist.rsample()
        return z0, z_dist



"""

class ConditionalNPEncoder(nn.Module):
    def __init__(self, args):
        super(ConditionalNPEncoder, self).__init__()
        self.num_label = args.num_label

        self.encoder = nn.Sequential(nn.Linear(2 + args.num_label, args.encoder_hidden_dim),
                                   nn.SiLU(),
                                   nn.Linear(args.encoder_hidden_dim, 2*args.encoder_hidden_dim),
                                   nn.SiLU(),
                                   nn.Linear(2*args.encoder_hidden_dim, 2*args.encoder_hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(2*args.encoder_hidden_dim, 2*args.encoder_hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(2*args.encoder_hidden_dim, args.encoder_hidden_dim),
                                     nn.SiLU())
                                     # nn.Linear(args.encoder_hidden_dim, args.latent_dimension))

        self.latent_mu = nn.Sequential(nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(args.encoder_hidden_dim, args.latent_dimension))

        self.latent_sigma = nn.Sequential(nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(args.encoder_hidden_dim, args.latent_dimension))

    def forward(self, x, label, span):
        # x (B, S, 1)  label (B, num_label)  span (B, S)
        B, S, _ = x.size()
        span = span.unsqueeze(-1)  # (B, S, 1)
        label = torch.broadcast_to(label.unsqueeze(1), (B, S, self.num_label))

        input_pairs = torch.cat((x, span, label), dim=-1)  # (B, S, 2+num_label)
        r = self.encoder(input_pairs)  # (B, S, H)

        r = r.mean(1)  # (B, H)

        z, z_dist = self.reparameterization(r)
        return r, z, z_dist
        # return r, r, 0, 0

    def reparameterization(self, z):
        mean = self.latent_mu(z)
        std = self.latent_sigma(z)
        z_dist = Normal(mean, nn.functional.softplus(std))
        z0 = z_dist.rsample()
        return z0, z_dist

        # epsilon = torch.randn(qz0_mean.size()).to(z.device)
        # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        # return z0, qz0_mean, qz0_logvar
        
        
class ResConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResConvBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, dilation=1, padding=1),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, stride=1, dilation=1, padding=1),
                                   nn.BatchNorm1d(input_dim))
        self.act1 = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)


    def forward(self, x):
        output = self.model(x)  # (B, S, C)
        output = output + x     # residual connection
        output = self.maxpool(self.act1(output))
        return output

class UnconditionalConvEncoder2(nn.Module):
    def __init__(self, args):
        super(UnconditionalConvEncoder2, self).__init__()
        self.num_label = args.num_label

        layers = []
        layers.append(nn.Conv1d(in_channels=2+args.num_label, out_channels=args.latent_dimension, kernel_size=3, stride=1, dilation=1))

        for i in range(args.encoder_blocks):
            layers.append(ResConvBlock(args.latent_dimension, args.encoder_hidden_dim))

        self.model = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_mu = nn.Linear(args.latent_dimension, args.latent_dimension)
        self.latent_sigma = nn.Linear(args.latent_dimension, args.latent_dimension)

    def forward(self, x, label, span):
        # x (B, S, 1)  label (B, num_label)  span (B, S)
        B, S, _ = x.size()

        span = span.unsqueeze(-1)
        label = torch.broadcast_to(label.unsqueeze(1), (B, S, self.num_label))

        input_pairs = torch.cat((x, span, label), dim=-1)
        output = self.model(input_pairs.permute(0, 2, 1))
        output = self.glob_pool(output).squeeze(-1)
        z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return output, z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        qz0_mean = self.latent_mu(z)
        qz0_logvar = self.latent_sigma(z)
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar

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
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
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
        self.dropout = nn.Dropout(p=args.dropout)
        self.latent_dim = args.latent_dimension
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.label_embedding = nn.Linear(args.num_label, args.encoder_embedding_dim, bias=False)
        # learnable pos_encoder for now
        # self.pos_encoder = PositionalEncoding(args.encoder_embedding_dim, args.data_length, args.dropout)
        self.pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)

        self.output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)
        #self.output_fc = nn.Linear(args.encoder_embedding_dim, args.latent_dimension)   # for before output
        self.label_fc = nn.Sequential(nn.Linear(args.latent_dimension + args.num_label, 2 * args.latent_dimension),
                                      nn.ReLU(),
                                      nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))

    # def forward(self, x, label, span):
    #     # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
    #     B = x.size(0) ; S = span.size(0)
    #     x = self.embedding(x)  # (B, S, E)
    #
    #     # add positional embedding
    #     span = self.pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))
    #     x = x + span
    #     x = self.dropout(x)
    #     x = x.permute(1, 0, 2)   # (S, B, E)
    #
    #     memory = self.transformer_encoder(src = x)   # (S, B, E)
    #     output = self.output_fc(memory)   # (S, B, E)
    #     output = output.mean(0)  # (B, E)
    #
    #     # concat output with label
    #     output = torch.cat((output, label), dim=-1)
    #     output = self.label_fc(output)
    #
    #     z0, qz0_mean, qz0_logvar = self.reparameterization(output)
    #     return memory, z0, qz0_mean, qz0_logvar

    ## forward for label in cls
    def forward(self, x, label, span):
        # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
        # add 0 in span
        span = torch.cat((torch.zeros(1).cuda(), span), dim=0)

        B = x.size(0) ; S = span.size(0)
        label = self.label_embedding(label)
        x = self.embedding(x) # (B, S, E)

        # concat label with x
        x = torch.cat((label.unsqueeze(1), x), dim=1)

        # add positional embedding
        span = self.pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))    # broadcast는 dimension이 맞지 않아도 괜찮다
        x = x + span
        x = self.dropout(x)

        # x = torch.cat((x, span.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)), dim=-1)   # (B, S, E+1)
        x = x.permute(1, 0, 2)  # (S, B, E)
        #x = self.pos_encoder(x)

        memory = self.transformer_encoder(src=x)  # (S, B, E)
        output = self.output_fc(memory) # (S, B, 2*E)
        output = output.mean(0)  # (B, 2*E)

        z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return memory.mean(0), z0, qz0_mean, qz0_logvar

    # before encoding label in the input
    # def forward(self, x, label, span):
    #     B = x.size(0) ; S = span.size(0)
    #     x = self.embedding(x)
    #     x = torch.cat((x, span.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)), dim=-1)  # (B, S, E+1)
    #     x = x.permute(1, 0, 2)  # (S, B, E)
    #
    #     memory = self.transformer_encoder(src=x)  # (S, B, E)
    #     output = self.output_fc(memory) # (B, 2E)
    #     output = output.mean(0)  # (B, 2E)
    #
    #     z0, qz0_mean, qz0_logvar = self.reparameterization(output)
    #     return memory, z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        # z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar


class UnconditionalTransformerEncoder(nn.Module):
    def __init__(self, args):
        super(UnconditionalTransformerEncoder, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.latent_dim = args.latent_dimension
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)

        self.pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)

        self.output_fc = nn.Linear(args.encoder_embedding_dim, args.latent_dimension)

    def forward(self, x, span):
        # x shape of (B, S, 1), span shape of (S)
        B = x.size(0) ; S = span.size(0)
        x = self.embedding(x)   # (B, S, E)

        span = self.pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))   #(B, S, E)
        x = x + span
        x = self.dropout(x)

        x = x.permute(1, 0, 2)  # (S, B, E)

        memory = self.transformer_encoder(src=x)   # (S, B, E)
        output = self.output_fc(memory)   # (S, B, E)
        output = output.mean(0)           # (B, E)

        #z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return output

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar



class ConvEncoder(nn.Module):
    def __init__(self, args):
        super(ConvEncoder, self).__init__()
        self.latent_dim = args.latent_dimension

        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, stride=3, dilation=1))

        for i in range(args.encoder_blocks):
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=3, dilation=1))

        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=256, out_channels=2 * args.latent_dimension, kernel_size=3, stride=3, dilation=1))
        # if sampling change the out channel to double the size


        self.model = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x, label, span):
        # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
        B = x.size(0) ; S = span.size(0)
        x = torch.cat((torch.where(label)[1].unsqueeze(-1).unsqueeze(-1), x), dim=1)
        x = self.model(x.permute(0, 2, 1))   # (B, S, E)
        memory = self.glob_pool(x).squeeze(-1)    # (B, L)

        z0, qz0_mean, qz0_logvar = self.reparameterization(memory)
        #qz0_mean = qz0_logvar = torch.zeros(B, self.latent_dim).cuda()
        return memory, z0, qz0_mean, qz0_logvar


    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar

class UnconditionConvEncoder(nn.Module):
    def __init__(self, args):
        super(UnconditionConvEncoder, self).__init__()
        self.latent_dim = args.latent_dimension

        if args.stride == 3:
            layers = self.stride3(args)
        elif args.stride == 1:
            layers = self.stride1(args)

        self.model = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(2*args.latent_dimension, 2*args.latent_dimension)

    def stride3(self, args):
        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=3, dilation=1))

        for i in range(args.encoder_blocks):
            layers.append(nn.SiLU())
            layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=3, dilation=1))

        layers.append(nn.SiLU())
        layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.latent_dimension, kernel_size=3, stride=3, dilation=1))  # AE의 구조
        return layers


    def stride1(self, args):
        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
        layers.append(nn.MaxPool1d(kernel_size=args.maxpool_kernelsize))

        for i in range(args.encoder_blocks):
            layers.append(nn.SiLU())
            layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
            layers.append(nn.MaxPool1d(kernel_size=args.maxpool_kernelsize))

        layers.append(nn.SiLU())
        layers.append(nn.Conv1d(in_channels=args.encoder_hidden_dim, out_channels=args.latent_dimension, kernel_size=3, stride=1, dilation=1))  # AE의 구조
        return layers

    def forward(self, x, span):
        # x (B, S, 1), span (S)
        x = self.model(x.permute(0, 2, 1)) # (B, E, S)
        memory = self.glob_pool(x).squeeze(-1)  # (B, 2L)

        # z0, qz0_mean, qz0_logvar = self.reparameterization(memory)
        # return memory, z0, qz0_mean, qz0_logvar

        return memory

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar

class UnconditionTransConvEncoder(nn.Module):
    def __init__(self, args):
        super(UnconditionTransConvEncoder, self).__init__()
        self.latent_dim = args.latent_dimension

        self.conv_model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, stride=1, dilation=1),
                                        nn.MaxPool1d(kernel_size=2),
                                        nn.SiLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1),
                                        nn.MaxPool1d(kernel_size=2),
                                        nn.SiLU(),
                                        nn.Conv1d(in_channels=256, out_channels=args.encoder_embedding_dim, kernel_size=3, stride=1, dilation=1),
                                        nn.MaxPool1d(kernel_size=2))

        encoder_layers = nn.TransformerEncoderLayer(args.encoder_embedding_dim, args.encoder_attnheads, args.encoder_hidden_dim, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.encoder_blocks)
        self.pos_encoder = nn.Conv1d(in_channels=args.encoder_embedding_dim, out_channels=args.encoder_embedding_dim, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=args.dropout)
        self.output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)

    def forward(self, x, span):
        # x (B, S, 1)  span (S)
        x = self.conv_model(x.permute(0, 2, 1))  # (B, E, S)

        # positional embedding?
        pos_x = self.pos_encoder(x)   # (B, E, S)
        x = x + pos_x   # (B, E, S)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)    # (S, B, E)

        memory = self.transformer_encoder(src=x)   # (S, B, E)
        output = self.output_fc(memory)    # (S, B, 2E)
        output = output.mean(0)     # (B, 2E)

        z0, qz0_mean, qz0_logvar = self.reparameterization(output)
        return memory.mean(0), z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        return z0, qz0_mean, qz0_logvar

"""


