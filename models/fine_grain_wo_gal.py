import numpy as np
import torch
import torch.nn as nn
from torchdyn.models import DepthCat, NeuralDE


# def batch_fourier_expansion(n_range, s):
#     # s is shape of (batch_size x n_eig * n_harmonics)
#     cos_s = s[:, :n_range.size(0)] * n_range
#     cos_s = torch.diag_embed(torch.cos(cos_s))
#     sin_s = s[:, n_range.size(0):] * n_range
#     sin_s = torch.diag_embed(torch.sin(sin_s))
#     return torch.cat((cos_s, sin_s), dim=1)

def FourierExpansion(n_range, s):
    s_n_range = s * n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis


class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension+1, coeffs_size)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(coeffs_size, coeffs_size)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(coeffs_size, 2)

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)

class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, in_features, out_features, latent_dimension, expfunc, n_harmonics, n_eig, lower_bound, upper_bound):
        super().__init__()
        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig
        self.lower_bound, self.upper_bound = lower_bound, upper_bound

        self.coeffs_size = in_features * out_features * n_harmonics * n_eig

        self.coeffs_generator = CoeffDecoder(latent_dimension=latent_dimension, coeffs_size= self.coeffs_size)

    def forward(self, x):
        assert x.size(1) == (self.in_features + self.latent_dimension + 1)
        self.batch_size = x.size(0)
        input = x[:, :self.in_features]
        s = x[-1, self.in_features]
        latent_variable = x[:, -(self.latent_dimension+1):]

        w = self.coeffs_generator(latent_variable)
        weight = w[:, 0].unsqueeze(-1) ; bias = w[:, 1].unsqueeze(-1)
        output = (weight * input) + bias

        return output


class AugmentedGalerkin(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, expfunc, n_harmonics, n_eig, lower_bound, upper_bound):
        super(AugmentedGalerkin, self).__init__()
        self.depth_cat = DepthCat(1)
        if expfunc == 'fourier':
            expfunc = FourierExpansion

        self.gallinear = WeightAdaptiveGallinear(in_features=in_features, out_features=out_features,
                                                 latent_dimension=latent_dim, expfunc=expfunc, n_harmonics=n_harmonics,
                                                 n_eig=n_eig, lower_bound=lower_bound, upper_bound=upper_bound)
        self.z = None

    def forward(self, x):
        x = self.depth_cat(x)
        x = torch.cat((x, self.z), 1)
        out = self.gallinear(x)
        return out


class GalerkinDE_dilationtest(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.func = AugmentedGalerkin(in_features = args.in_features, out_features=args.out_features, latent_dim=args.latent_dimension,
                                      expfunc=args.expfunc, n_harmonics=args.n_harmonics, n_eig=args.n_eig, lower_bound=args.lower_bound, upper_bound=args.upper_bound)
        self.galerkin_ode = NeuralDE(self.func, solver='rk4')

    def forward(self, t, x, z):
        y0 = x[:, 0]
        t = torch.squeeze(t[0])
        self.func.z = z

        decoded_traj = self.galerkin_ode.trajectory(y0, t).transpose(0, 1)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        #dilation_sum = torch.mul(self.func.gallinear.dilation, self.func.gallinear.dilation).sum()
        return mse_loss

    def predict(self, t, x, z):
        y0 = x[:, 0]
        t = torch.squeeze(t[0])
        self.func.z = z

        decoded_traj = self.galerkin_ode.trajectory(y0, t).transpose(0, 1)
        return decoded_traj