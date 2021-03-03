import numpy
import torch
import torch.nn as nn
from torchdyn.models import DepthCat, NeuralDE


class Decoder(nn.Module):
    def __init__(self, latent_dimension, output_size, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dimension, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)

class GalLinear(nn.Module):
    def __init__(self, in_features, out_features, latent_dimension, n_harmonics, n_eig):
        super().__init__()
        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        coeffs_size = (in_features + 1) * out_features * n_harmonics * n_eig
        self.weight_generator = Decoder(latent_dimension=latent_dimension+1, output_size=in_features*out_features, hidden_dim=coeffs_size)
        self.bias_generator = Decoder(latent_dimension=latent_dimension+1, output_size=out_features, hidden_dim=coeffs_size)

    def forward(self, x):
        batch_size = x.size(0)
        s = x[-1, self.in_features]
        input = torch.unsqueeze(x[:, :self.in_features], dim=-1)
        latent_variable = x[:, -(self.latent_dimension + 1):]

        weight = self.weight_generator(latent_variable).reshape(batch_size, self.in_features, self.out_features)
        bias = self.bias_generator(latent_variable)

        weighted = torch.squeeze(torch.bmm(input, weight), dim=1)
        return torch.add(weighted, bias)


class AugmentedGalerkin(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, n_harmonics, n_eig):
        super().__init__()
        self.depth_cat = DepthCat(1)
        self.gallinear = GalLinear(in_features=in_features, out_features=out_features, latent_dimension=latent_dim, n_harmonics=n_harmonics, n_eig=n_eig)
        self.z = None

    def forward(self, x):
        x = self.depth_cat(x)
        x = torch.cat((x, self.z), 1)
        out = self.gallinear(x)
        return out

class GalerkinDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.func = AugmentedGalerkin(in_features=args.in_features, out_features=args.out_features, latent_dim=args.latent_dimension,
                                      n_harmonics=args.n_harmonics, n_eig=args.n_eig)
        self.galerkin_ode = NeuralDE(self.func, solver='rk4')

    def forward(self, t, x, z):
        y0 = x[:, 0]
        t = torch.squeeze(t[0])
        self.func.z = z

        decoded_traj = self.galerkin_ode.trajectory(y0, t).transpose(0, 1)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss

    def predict(self, t, x, z):
        y0 = x[:, 0]
        t = torch.squeeze(t[0])
        self.func.z = z

        decoded_traj = self.galerkin_ode.trajectory(y0, t).transpose(0, 1)
        return decoded_traj

