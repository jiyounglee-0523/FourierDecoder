import numpy as np
import torch
import torch.nn as nn
from torchdyn.models import DepthCat, NeuralDE


def batch_fourier_expansion(n_range, s):
    # s is shape of (batch_size x n_eig * n_harmonics)
    cos_s = s[:, :n_range.size(0)] * n_range
    cos_s = torch.diag_embed(torch.cos(cos_s))
    sin_s = s[:, n_range.size(0):] * n_range
    sin_s = torch.diag_embed(torch.sin(sin_s))
    return torch.cat((cos_s, sin_s), dim=1)

class CoeffMatrix(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_dimension, coeffs_size, bias=False)
        # K = torch.Tensor([0., 1., 2., 0., 1., 2.])
        # # when given latent variable
        # torch.nn.init.zeros_(self.fc1.weight)
        # with torch.no_grad():
        #     self.fc1.bias = torch.nn.Parameter(K)

        # # when given dilation
        # torch.nn.init.eye_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        out = self.fc1(x)
        return out

class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, coeffs_size)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(coeffs_size, coeffs_size)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(coeffs_size, coeffs_size)
        #self.relu = nn.ReLU()
        print('CoeffDecoder_small output size: {}'.format(coeffs_size))

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)

class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, in_features, out_features, latent_dimension, expfunc, n_harmonics, n_eig, zero_out):
        super().__init__()
        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig
        self.zero_out = zero_out

        self.coeffs_size = ((in_features + 1) * out_features + 1) * n_harmonics * n_eig

        self.coeffs_generator = CoeffMatrix(latent_dimension=latent_dimension, coeffs_size= n_harmonics*n_eig)

    def assign_weights(self, s, coeffs, dilation):
        n_range = torch.Tensor([1.] * self.n_harmonics).to(self.input.device)
        s = s.new_full((self.batch_size, self.n_eig * self.n_harmonics), s.item())
        # s = (s * dilation) + shift
        s = (s * dilation)

        B = self.expfunc(n_range, s) # shape of (batch_size, n_eig * n_harmonics, n_harmonics)
        X = torch.bmm(coeffs, B)
        return X.sum(2)

    def forward(self, x):
        assert x.size(1) == (self.in_features + self.latent_dimension + 1)
        # x should be ordered in input_data, s, latent_variable
        self.batch_size = x.size(0)
        s = x[-1, self.in_features]
        self.input = torch.unsqueeze(x[:, :self.in_features], dim=-1)  # shape of (batch_size, in_features, 1)
        latent_variable = x[:, -self.latent_dimension:]  # shape of (batch_size, latent_dim)
        #true_dilation = torch.Tensor([[0., 1., 2., 0., 1., 2.]]*self.batch_size).to(self.input.device)

        #amps = latent_variable[:, 0].unsqueeze(-1)    # shape of (batch_size, 1)
        self.coeff = latent_variable.repeat(1, 2).view(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)
        self.coeff[:, ::2, :] = 0.
        self.coeff[:, :, 2] = -self.coeff[:, :, 2]
        #self.coeff = torch.tensor([[[0., 0., 0., 0.], [1., 1., 0., 0.]]] * self.batch_size).to(self.input.device)
        #self.coeff = torch.matmul(amps, coeff).permute(1, 0, 2)

        coeffs = self.coeffs_generator(latent_variable)
        #coeffs = self.coeffs_generator(latent_variable).reshape(self.batch_size, self.coeffs_size)
        # self.coeff = coeffs[:, :((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics)].reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)

        #self.dilation = self.coeffs_generator(latent_variable)
        self.dilation = coeffs
            #torch.cat((latent_variable, torch.zeros(self.batch_size, 2).cuda()), dim=-1).cuda()

        #self.dilation = coeffs[:, self.n_eig * self.n_harmonics]
        # self.shift = coeffs[:, :self.n_eig * self.n_harmonics]
        #self.dilation = coeffs[:, ((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics): (((self.in_features + 1) * self.out_features + 1) * self.n_eig * self.n_harmonics)]
        #self.dilation = torch.Tensor([[0., 1., 2., 0., 1., 2.]]*self.batch_size).to(self.input.device)
        #self.dilation.requires_grad = True

        #self.dilation = torch.Tensor([[2., 1.]] * self.batch_size).cuda()
        w = self.assign_weights(s, self.coeff, self.dilation)
        # print('initial dilation values are {}'.format(self.dilation))
        # print('initial coeffs values are {}'.format(self.coeff))
        # time.sleep(60)

        self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
        if self.zero_out:
            self.weight = torch.zeros(self.batch_size, self.in_features, self.out_features).cuda()
        self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)

        self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
        return torch.add(self.weighted, self.bias)


class AugmentedGalerkin(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, expfunc, n_harmonics, n_eig, zero_out):
        super(AugmentedGalerkin, self).__init__()
        self.depth_cat = DepthCat(1)
        if expfunc == 'fourier':
            expfunc = batch_fourier_expansion

        self.gallinear = WeightAdaptiveGallinear(in_features=in_features, out_features=out_features,
                                                 latent_dimension=latent_dim, expfunc=expfunc, n_harmonics=n_harmonics,
                                                 n_eig=n_eig, zero_out=zero_out)
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
                                      expfunc=args.expfunc, n_harmonics=args.n_harmonics, n_eig=args.n_eig, zero_out=args.zero_out)
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