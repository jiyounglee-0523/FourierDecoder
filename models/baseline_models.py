import torch
import torch.nn as nn

from torchdyn.models import NeuralDE



class LatentODE(nn.Module):
    def __init__(self, args):
        super(LatentODE, self).__init__()

        f = nn.Sequential(nn.Linear(args.latent_dimension), 4 * args.n_harmonics,
                          nn.SiLU(),
                          nn.Linear(4 * args.n_harmonics, 4* args.n_harmonics),
                          nn.SiLU(),
                          nn.Linear(4 * args.n_harmonics, args.latent_dimension))

        self.decoder = NeuralDE(f)
        self.output_fc = nn.Linear(args.latent_dimension, 1)

    def forward(self, t, z):
        t = t.squeeze(0)
        decoded_traj = self.decoder(z, t).transpose(0, 1)
        decoded_traj = self.output_fc(decoded_traj)
        return decoded_traj



class SONOEs:
    pass

class ANODES:
    pass
