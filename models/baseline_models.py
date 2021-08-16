import torch
import torch.nn as nn

from torchdyn.models import NeuralODE



class LatentODE(nn.Module):
    def __init__(self, args):
        super(LatentODE, self).__init__()

        f = nn.Sequential(nn.Linear(args.latent_dimension), 4 * args.n_harmonics,
                          nn.SiLU(),
                          nn.Linear(4 * args.n_harmonics, 4* args.n_harmonics),
                          nn.SiLU(),
                          nn.Linear(4 * args.n_harmonics, args.latent_dimension))

        self.decoder = NeuralODE(f)
        self.output_fc = nn.Linear(args.latent_dimension, 1)

    def forward(self, t, z):
        t = t.squeeze(0)
        decoded_traj = self.decoder(z, t).transpose(0, 1)
        decoded_traj = self.output_fc(decoded_traj)
        return decoded_traj


class GRUDecoder(nn.Module):
    def __init__(self, args):
        super(GRUDecoder, self).__init__()
        self.decoder_layers = args.decoder_layers
        self.decoder_hidden_dim = args.decoder_hidden_dim

        self.input_embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.init_hidden_embedding = nn.Linear(args.latent_dimension, args.decoder_hidden_dim)

        self.GRU = nn.GRU(input_size=args.encoder_embedding_dim, hidden_size=args.decoder_hidden_dim, num_layers=args.decoder_layers+2, batch_first=True, dropout=args.dropout)
        self.output_fc = nn.Linear(args.decoder_hidden_dim, 1)

    def forward(self, target_x, x, memory):
        # target_x = (B, S, 1),  memory = (B, E), x = (B, S, 1)
        B = target_x.size(0)
        x = self.embedding(x)  # (B, E)
        memory = self.init_hidden_embedding(memory)
        memory = torch.broadcast_to(memory.unsqueeze(0), (self.decoder_layers, B, self.decoder_hidden_dim)) # (num_layers, B, hidden)
        output, _ = self.GRU(x, memory)
        output = self.output_fc(output)   # (B, S, 1)
        return output


class NeuralProcess(nn.Module):
    def __init__(self, args):
        super(NeuralProcess, self).__init__()

        layers = []
        layers.append(nn.Linear(args.latent_dimension + 1, args.decoder_hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(args.decoder_layers):
            layers.append(nn.Linear(args.decoder_hidden_dim, args.decoder_hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(args.decoder_hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, target_x, memory):
        # target_x = (B, S, 1)  memory = (B, E)
        B, S, _ = target_x.size()
        E = memory.size(-1)

        memory = torch.broadcast_to(memory.unsqueeze(1), (B, S, E))
        target_x = torch.cat((memory, target_x), dim=-1)   # (B, S, E+1)

        output = self.model(target_x)  # (B, S, 1)
        return output





class SONOEs:
    pass

class ANODES:
    pass
