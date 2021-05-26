import torch
import torch.nn as nn

from models.encoder import *
from models.FNODEs import FNODEs



class LatentNeuralDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder()
        elif args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args)
        elif args.encoder == 'BiRNN':
            raise NotImplementedError


        if args.model_type == 'FNODEs':
            self.decoder = FNODEs(args)

    def forward(self, t, x):
        z = self.encoder(x)
        t = torch.squeeze(t[0])
        self.func.z = z

        y0 = x[:, 0]
        decoded_traj = self.decoder(t, z)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss

    def predict(self, t, x):
        with torch.no_grad():
            z = self.encoder(x)
            t = torch.squeeze(t[0])
            self.func.z = z

            y0 = x[x:, 0]
            decoded_traj = self.decoder(t, z)
        return decoded_traj