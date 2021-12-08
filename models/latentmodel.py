import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import *
from models.FourierModel import ConditionalFNP
from models.baseline_models import *

class ConditionalQueryFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalQueryFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension
        self.n_harmonics = args.n_harmonics

        self.encoder = ConvEncoder(args)

        if args.decoder == 'Fourier':
            self.decoder = ConditionalFNP(args)
        elif args.decoder == 'ODE':
            self.decoder = ODEDecoder(args)
        elif args.decoder == 'NP':
            self.decoder = NeuralProcess(args)
        elif args.decoder == 'Transformer':
            self.decoder = TransformerDecoder(args)
        elif args.decoder == 'RNN':
            self.decoder = GRUDecoder(args)

        self.prior = Normal(torch.zeros([self.latent_dim]).cuda(), torch.ones([self.latent_dim]).cuda())

    def forward(self, t, x, label, index):
        # t (B, S)  x (B, S, 1)  label (B)
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        # select irregulary sampled
        dummy = index.unsqueeze(-1)
        input_x = torch.gather(x, 1, dummy)
        input_t = torch.gather(t, 1, index)

        memory, z, z_dist = self.encoder(input_x, label_embed, span=input_t)
        # memory, z, qz0_mean, qz0_logvar = self.encoder(x, 0, span=t)
        kl_loss = torch.distributions.kl.kl_divergence(z_dist, self.prior).mean(-1).mean(0)

        # kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)  # (B, E+num_label)

        decoded_traj = self.decoder(t.unsqueeze(-1), z, x)
        x = x.squeeze(-1)
        # mse_loss = nn.MSELoss(reduction='sum')(decoded_traj, x)
        # mse_loss = mse_loss / B
        mse_loss = nn.MSELoss()(torch.gather(decoded_traj, 1, index), torch.gather(x, 1, index))
        return mse_loss, kl_loss
        # return mse_loss, 0