import torch
import torch.nn as nn

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


class LatentNeuralDE(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_hidden_dim, n_harmonics):
        super().__init__()
        self.encoder = RNNODEEncoder(input_dim=input_dim, latent_dim=latent_dim, rnn_hidden_dim=rnn_hidden_dim).cuda()

        # decoder
        f = nn.Sequential(nn.Linear(latent_dim, 4 * n_harmonics),
                          nn.SiLU(),
                          nn.Linear(4 * n_harmonics, 4 * n_harmonics),
                          nn.SiLU(),
                          nn.Linear(4 * n_harmonics, latent_dim))
        self.decoder = NeuralDE(func=f, solver='rk4').cuda()
        self.out = nn.Linear(latent_dim, 1).cuda()

    def forward(self, x, s_span):
        z = self.encoder(x)
        s_span = torch.squeeze(s_span[0])
        decoded_traj = self.decoder.trajectory(z, s_span).transpose(0, 1)
        decoded_traj = self.out(decoded_traj)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss

    def predict(self, x, s_span):
        z = self.encoder(x)
        s_span = torch.squeeze(s_span[0])
        decoded_traj = self.decoder.trajectory(z, s_span).transpose(0, 1)
        decoded_traj = self.out(decoded_traj)
        return decoded_traj