import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchdiffeq import odeint



# class LatentODE(nn.Module):
#     def __init__(self, args):
#         super(LatentODE, self).__init__()
#
#         f = nn.Sequential(nn.Linear(args.latent_dimension), 4 * args.n_harmonics,
#                           nn.SiLU(),
#                           nn.Linear(4 * args.n_harmonics, 4* args.n_harmonics),
#                           nn.SiLU(),
#                           nn.Linear(4 * args.n_harmonics, args.latent_dimension))
#
#         self.decoder = NeuralODE(f)
#         self.output_fc = nn.Linear(args.latent_dimension, 1)
#
#     def forward(self, t, z):
#         t = t.squeeze(0)
#         decoded_traj = self.decoder(z, t).transpose(0, 1)
#         decoded_traj = self.output_fc(decoded_traj)
#         return decoded_traj

# RNN Decoder
class GRUDecoder(nn.Module):
    def __init__(self, args):
        super(GRUDecoder, self).__init__()
        self.decoder_layers = args.decoder_layers
        self.decoder_hidden_dim = args.decoder_hidden_dim

        self.input_embedding = nn.Linear(1, 128)
        self.init_hidden_embedding = nn.Linear(args.latent_dimension+args.num_label, args.decoder_hidden_dim)

        self.GRU = nn.GRU(input_size=128, hidden_size=args.decoder_hidden_dim, num_layers=args.decoder_layers, batch_first=True, dropout=args.dropout)
        self.output_fc = nn.Linear(args.decoder_hidden_dim, 1)

    def forward(self, target_x, memory, x):
        # target_x = (B, S, 1),  memory = (B, E), x = (B, S, 1)
        B = target_x.size(0)
        x = torch.cat((torch.ones(B, 1, 1).cuda(), x), dim=1)   # (B, S+1, 1)
        x = self.input_embedding(x)  # (B, E)
        memory = self.init_hidden_embedding(memory)
        memory = torch.broadcast_to(memory.unsqueeze(0), (self.decoder_layers, B, self.decoder_hidden_dim)) # (num_layers, B, hidden)
        memory = memory.contiguous()
        output, _ = self.GRU(x, memory)
        output = self.output_fc(output).squeeze(-1)   # (B, S+1, 1)
        return output[:, :-1]

    def auto_regressive(self, target_x, memory):
        # target_x = (B, S, 1)  z = (B, E)
        B, S, _ = target_x.size()
        xx = self.input_embedding(torch.ones(B, 1, 1).cuda())
        memory = self.init_hidden_embedding(memory)
        memory = torch.broadcast_to(memory.unsqueeze(0), (self.decoder_layers, B, self.decoder_hidden_dim))
        memory = memory.contiguous()

        outputs = []
        for i in range(500):
            output, _ = self.GRU(xx, memory)
            output = self.output_fc(output)[:, -1, :]
            outputs.append(output)
            xx = torch.cat((xx, self.input_embedding(output).unsqueeze(1)), dim=1)
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2)
        return outputs


# NP
class NeuralProcess(nn.Module):
    def __init__(self, args):
        super(NeuralProcess, self).__init__()

        layers = []
        layers.append(nn.Linear(args.latent_dimension + args.num_label+ 1, 2 * args.latent_dimension))
        # layers.append(nn.Linear(args.latent_dimension + 1, 2*args.latent_dimension))
        layers.append(nn.SiLU())

        for _ in range(args.decoder_layers):
            layers.append(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(2*args.latent_dimension, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, target_x, z, x):
        # target_x = (B, S, 1)  z = (B, E)   index (B, N)
        B, S, _ = target_x.size()
        E = z.size(-1)

        memory = torch.broadcast_to(z.unsqueeze(1), (B, S, E))
        target_x = torch.cat((memory, target_x), dim=-1)   # (B, S, E+1)

        output = self.model(target_x).squeeze(-1)  # (B, S, 1)
        return output



class ODEFunc(nn.Module):
    def __init__(self, latent_dimension, decoder_layers):
        super(ODEFunc, self).__init__()
        layers = []
        for _ in range(decoder_layers):
            layers.append(nn.Linear(2 * latent_dimension, 2 * latent_dimension))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.net(x)


class ODEDecoder(nn.Module):
    def __init__(self, args):
        super(ODEDecoder, self).__init__()
        self.fc1 = nn.Linear(args.latent_dimension + args.num_label, 2 * args.latent_dimension)
        self.odenet = ODEFunc(args.latent_dimension, args.decoder_layers)
        self.fc2 = nn.Linear(2*args.latent_dimension, 1)

    def forward(self, target_x, z, x):
        # target_x = (B, S, 1)  z = (B, E)
        z = self.fc1(z)
        pred_y = odeint(self.odenet, z, target_x[0].squeeze(-1), method='rk4')
        pred_y = self.fc2(pred_y).permute(1, 0, 2).squeeze(-1)
        return pred_y

class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = nn.Linear(1, 128, bias=False)
        self.label_embedding = nn.Linear(args.num_label + args.latent_dimension, 128, bias=False)

        # model
        self.pos_embedding = nn.Linear(1, 128)
        decoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=args.decoder_hidden_dim,
                                                   dropout = args.dropout)
        self.model = nn.TransformerEncoder(decoder_layer, num_layers=args.decoder_layers)
        self.output_fc = nn.Linear(128, 1, bias=False)

    def forward(self, target_x, r, x):
        # x (B, S, 1)  target_x (B, S, 1)  r (B, E)
        B = r.size(0)

        r = self.label_embedding(r).unsqueeze(1)  # (B, 1, E)
        x = self.embedding(x)  # (B, S, E)
        x = torch.cat((r, x), dim=1)  # (B, S+1, E)

        target_x = torch.cat((torch.zeros(B, 1, 1).cuda(), target_x), dim=1)  # (B, S+1, 1)
        target_x = self.pos_embedding(target_x)  # (B, S+1, E)
        x = x + target_x  # (B, S+1, E)
        x = self.dropout(x)

        x = x.permute(1, 0, 2)  # (S+1, B, E)
        mask = self.generate_square_subsequent_mask(x.size(0))
        output = self.model(src=x, mask=mask).permute(1, 0, 2)  # (B, S+1, E)
        output = self.output_fc(output).squeeze(-1)
        return output[:, :-1]

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def auto_regressive(self, r, target_x):
        # r (B, E)  target_x (B, S, 1)
        with torch.no_grad():
            B = r.size(0)
            S = target_x.size(1)

            r = self.label_embedding(r).unsqueeze(1)   # (B, 1, E)
            target_x = torch.cat((torch.zeros(B, 1, 1).cuda(), target_x), dim=1)  # (B, S+1, 1)

            dec_x = r.permute(1, 0, 2)  # (1, B, E)
            outputs = []
            for i in range(S):
                mask = self.generate_square_subsequent_mask(dec_x.size(0))
                dec_span = self.pos_embedding(target_x[:, :i+1, :]).permute(1, 0, 2) # (i, B, E)
                x = dec_x + dec_span
                output = self.model(src=x, mask=mask)  # (i, B, E)
                output = self.output_fc(output)[-1]  # (B, 1)
                outputs.append(output)
                dec_x = torch.cat((dec_x, self.embedding(output.unsqueeze(0))), dim=0)
        return outputs

class Conv1D(nn.Module):
    def __init__(self, args):
        super(Conv1D, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose1d(in_channels=args.latent_dimension + args.num_label, out_channels=args.decoder_hidden_dim, kernel_size=3, stride=2, dilation=2, output_padding=1))
        layers.append(nn.Upsample(scale_factor=4, mode='linear'))
        layers.append(nn.SiLU())
        layers.append(nn.ConvTranspose1d(in_channels=args.decoder_hidden_dim, out_channels=args.decoder_hidden_dim, kernel_size=3, stride=2, padding=1, dilation=2))
        layers.append(nn.Upsample(scale_factor=4, mode='linear'))
        layers.append(nn.SiLU())
        layers.append(nn.ConvTranspose1d(in_channels=args.decoder_hidden_dim, out_channels=args.latent_dimension, kernel_size=3, stride=2, padding=1, dilation=2))
        # layers.append(nn.Upsample(scale_factor=4, mode='linear'))  # 403 LENGTH
        layers.append(nn.SiLU())
        layers.append(nn.ConvTranspose1d(in_channels=args.latent_dimension, out_channels=1, kernel_size=3, stride=2, padding=1, dilation=2))
        self.model = nn.Sequential(*layers)

    def forward(self, target_x, r, x):
        # x (B, S, 1)  target_x (B, S, 1)  r (B, E)
        B, E = r.size()
        r = torch.broadcast_to(r.unsqueeze(-1), (B, E, 2))  # (B, E, 2)
        output = self.model(r.unsqueeze(-1)).squeeze(1)
        return output[:, :500]


# (B, E, 1)
"""
# SelfAttentionNP
class MultiHeadAttn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.dim_out)
        if mask is not None:
            mask = mask.bool().to(q.device)
            mask = torch.stack([mask]*q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, -3)
            A = torch.softmax(A_logits.masked_fill(mask, -float('inf')), -1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask=mask))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out  # (B, S, E)

class SelfAttn(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim_in = args.latent_dimension + 1
        dim_out = args.decoder_hidden_dim
        num_heads=8

        self.attn1 = MultiHeadAttn(dim_in, dim_in, dim_in, dim_out, num_heads)
        # self.attn2 = MultiHeadAttn(dim_out, dim_out, dim_out, dim_out, num_heads)
        self.output_fc = nn.Linear(dim_out, 1)

    def forward(self, target_x, r, mask=None):
        # r (B, E), target_x = (B, S, 1)
        B, S, _ = target_x.size()
        E = r.shape[-1]
        r = torch.broadcast_to(r.unsqueeze(1), (B, S, E))
        x = torch.cat((r, target_x), dim=-1)  # (B, S, E+1)
        output = self.attn1(x, x, x, mask=mask)
        # output = self.attn2(output, output, output, mask=mask)
        return self.output_fc(output).squeeze(-1)
"""

