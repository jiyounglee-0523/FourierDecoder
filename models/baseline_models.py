import torch
import torch.nn as nn

from torchdiffeq import odeint



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
