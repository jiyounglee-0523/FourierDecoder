# reference : https://pytorch.org/tutorials/beginner/translation_transformer.html

import torch
import torch.nn as nn

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time

from datasets.cond_dataset import get_dataloader
from utils.model_utils import count_parameters
from models.encoder import TransformerEncoder
from trainer.ConditionalTrainer import ConditionalBaseTrainer

# class TransformerEncoder(nn.Module):
#     def __init__(self, args):
#         super(TransformerEncoder, self).__init__()
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.num_label = args.num_label
#         self.latent_dim = args.latent_dimension
#
#         self.embedding = nn.Linear(1, args.encoder_embedding_dim)
#         self.label_embedding = nn.Linear(args.num_label, args.encoder_embedding_dim, bias=False)
#         self.pos_embedding = nn.Linear(1, args.encoder_embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
#                                                    dropout=args.dropout)
#         self.model = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_blocks)
#         self.output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)
#
#     def forward(self, x, label, span):
#         # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
#         B = x.size(0) ; S = span.size(0)
#
#         # add 0 in span
#         span = torch.cat((torch.zeros(1).cuda(), span), dim=0)
#
#         B = x.size(0) ; S = span.size(0)
#         label = self.label_embedding(label)  # (B, E)
#         x = self.embedding(x)  # (B, S, E)
#         x = torch.cat((label.unsqueeze(1), x), dim=1)  # (B, S+1, E)
#
#         # add positional embedding
#         span = self.pos_embedding(torch.broadcast_to(span, (B, S)).unsqueeze(-1))   # (B, S, E)
#         x = x + span
#         x = self.dropout(x)
#
#         x = x.permute(1, 0, 2)   # (S, B, E)
#         memory = self.model(src=x)  # (S, B, E)
#
#         z = self.output_fc(memory)  # (S, B, 2E)
#         z = z.mean(0)  # (B, 2E)
#         z0, qz0_mean, qz0_logvar = self.reparameterization(z)  # (B, E)
#
#         return memory, z0, qz0_mean, qz0_logvar
#
#     def reparameterization(self, z):
#         qz0_mean = z[:, :self.latent_dim]
#         qz0_logvar = z[:, self.latent_dim:]
#         epsilon = torch.randn(qz0_mean.size()).to(z.device)
#         z0 = epsilon * qz0_logvar + qz0_mean
#         return z0, qz0_mean, qz0_logvar


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.label_embedding = nn.Linear(args.num_label + args.latent_dimension, args.encoder_embedding_dim, bias=False)

        # model
        self.pos_embedding = nn.Linear(1, args.encoder_embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
                                                   dropout=args.dropout)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=args.encoder_blocks)
        self.output_fc = nn.Linear(args.encoder_embedding_dim, 1)

    def forward(self, memory, z0, span, x, label):
        # write shapes
        # concat 0 in span
        span = torch.cat((torch.zeros(1).cuda(), span), dim=0)

        B = x.size(0) ; S = span.size(0)

        z0 = torch.cat((z0, label), dim=-1)   # (B, L+num_label)
        z0 = self.label_embedding(z0) # (B, E)
        x = self.embedding(x)   # (B, S, E)
        x = torch.cat((z0.unsqueeze(1), x), dim=1)  # (B, S+1, E)

        # add positional embedding
        span = self.pos_embedding(torch.broadcast_to(span, (B, S)).unsqueeze(-1))   # (B, S, E)
        x = x + span
        x = self.dropout(x)



class BaseTransformer(nn.Module):
    def __init__(self, args):
        super(BaseTransformer, self).__init__()

        self.encoder =

    def forward(self, t, x, label, sampling):







class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.num_label = args.num_label
        self.dataset_type = args.dataset_type

        self.latent_dim = args.latent_dimension
        self.enc_embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.enc_label_embedding = nn.Linear(args.num_label, args.encoder_embedding_dim, bias=False)
        self.dec_embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.dec_label_embedding = nn.Linear(args.num_label + args.latent_dimension, args.encoder_embedding_dim, bias=False)

        # model
        self.enc_pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
        self.dec_pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
                                                   dropout=args.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_blocks)
        self.enc_output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)


        decoder_layer = nn.TransformerDecoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
                                                   dropout=args.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.encoder_blocks)
        self.dec_output_fc = nn.Linear(args.encoder_embedding_dim, 1)

    def forward(self, span, x, label, sampling):
        # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
        sampled_span, sampled_x, label = self.data_check(span, x, label, sampling)

        # add 0 in span
        sampled_span = torch.cat((torch.zeros(1).cuda(), sampled_span), dim=0)

        B = x.size(0) ; S = span.size(0)
        label = self.enc_label_embedding(label)   # (B, E)
        enc_x = self.enc_embedding(x)  # (B, S, E)

        # concat label with x
        enc_x = torch.cat((label.unsqueeze(1), enc_x), dim=1)   # (B, S+1, E)

        # add positional embedding
        enc_span = self.enc_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1)) # (B, S, E)
        enc_x = enc_x + enc_span
        enc_x = self.dropout(enc_x)

        enc_x = enc_x.permute(1, 0, 2)   # (S, B, E)
        memory = self.encoder(src=enc_x)   # (S, B, E)
        z = self.enc_output_fc(memory)  # (S, B, 2E)
        z = z.mean(0)  # (B, 2E)
        z0, qz0_mean, qz0_logvar = self.reparameterization(z)   # (B, E)

        z0 = torch.cat((z0, label), dim=-1)  # (B, E+num_label)
        z0 = self.dec_label_embedding(z0)  # (B, E)
        dec_x = self.dec_embedding(x)  # (B, S, E)
        dec_x = torch.cat((z0.unsqueeze(1), dec_x), dim=1)  # (B, S+1, E)

        # add positional embedding
        dec_span = self.dec_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))  # (B, S, E)
        dec_x = dec_x + dec_span
        dec_x = self.dropout(dec_x)

        dec_mask = self.generate_square_subsequent_mask(S)  # (S, S)
        dec_x = dec_x.permute(1, 0, 2)
        output = self.decoder(tgt=dec_x, memory=memory, tgt_mask=dec_mask)  # (S, B, E)
        output = self.dec_output_fc(output).permute(1, 0, 2)  # (B, S, 1)

        return output, qz0_mean, qz0_logvar

    def inference(self, span, x, label, sampling):
        with torch.no_grad():
            span = torch.cat((torch.zeros(1).cuda(), span), dim=0)

            B = x.size(0) ; S = span.size(0)
            label = self.enc_label_embedding(label)  # (B, E)
            enc_x = self.enc_embedding(x)  # (B, S, E)

            # concat label with x
            enc_x = torch.cat((label.unsqueeze(1), enc_x), dim=1)   # (B, S+1, E)

            # add positional embedding
            enc_span = self.enc_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))  # (B, S, E)
            enc_x = enc_x + enc_span
            enc_x = self.dropout(enc_x)

            enc_x = enc_x.permute(1, 0, 2)  # (S, B, E)
            memory = self.encoder(src=enc_x) # (S, B, E)

            z = self.enc_output_fc(memory)  # (S, B, 2E)
            z = z.mean(0)  # (B, 2E)
            z0, qz0_mean, qz0_logvar = self.reparameterization(z)  # (B, E)

            z0 = torch.cat((z0, label), dim=1)  # (B, E+num_label)
            z0 = self.dec_label_embedding(z0)  # (B, E)
            dec_x = z0.unsqueeze(0)  # (1, B, E)

            for i in range(S-1):
                tgt_mask = self.generate_square_subsequent_mask(dec_x.size(0))
                dec_span = self.dec_pos_encoder(torch.broadcast_to(span[:i], (B, i+1)).unsqueeze(-1))  # double check
                dec_x = dec_x + dec_span
                output = self.decoder(dec_x, memory=memory, tgt_mask=tgt_mask)   # (S, B, E)
                output = self.dec_output_fc(output)  # (S, B, 1)
                dec_x = torch.cat((dec_x, output[0].unsqueeze(0)), dim=0)

            return dec_x.permute(1, 0, 2)

    def data_check(self, span, x, label, sampling):
        B = x.size(0)

        # one-hot label
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        if sampling:
            sampled_t, sampled_x = self.sampling(span, x)
            return sampled_t[0], sampled_x, label_embed
        else:
            return span[0], x, label_embed


    def sampling(self, t, x):
        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
            t = t[:, sample_idxs]  # (150)
            x = x[:, sample_idxs]
        elif self.dataset_type == 'NSynth':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 1600, replace=False)))[0]   # sampling..
            t = t[:, sample_idxs]
            x = x[:, sample_idxs]
        return t, x

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def reparameterization(self, z):
        qz0_mean = z[:, :self.latent_dim]
        qz0_logvar = z[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar




class BaseTransTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(BaseTransTrainer, self).__init__(args)

        self.model = Transformer(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.notes)}')

        if not self.debug:
            wandb.init(project='conditionalODE')
            wandb.config.update(args)
            wandb.watch(self.model, log='all')


    def train(self):
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                freq = sample['freq']
                amp = sample['amp']
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=True)


