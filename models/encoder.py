import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ConvEncoder(nn.Module):
    def __init__(self, args):
        super(ConvEncoder, self).__init__()
        self.num_label = args.num_label

        layers = []
        layers.append(nn.Conv1d(in_channels=2+args.num_label, out_channels=args.encoder_hidden_dim, kernel_size=3, stride=1, dilation=1))
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
        output = self.model(input_pairs.permute(0, 2, 1))  # (B, E, S)
        output = self.glob_pool(output).squeeze(-1)  # (B, E)
        z0, z_dist = self.reparameterization(output)
        return output, z0, z_dist

    def reparameterization(self, z):
        mean = self.latent_mu(z)
        std = self.latent_sigma(z)
        z_dist = Normal(mean, nn.functional.softplus(std))
        z0 = z_dist.rsample()
        return z0, z_dist

