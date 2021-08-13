import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from models.AttentionFourier import NonperiodicDecoder

## Attentive Neural Process
# reference : https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
# reference : https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py

class UnconditionQueryGenerator(nn.Module):
    def __init__(self, args):
        super(UnconditionQueryGenerator, self).__init__()
        self.n_harmonics = args.n_harmonics
        self.lower_bound, self.upper_bound = args.lower_bound, args.upper_bound

        layers = []
        layers.append(nn.Linear(args.latent_dimension, 2*args.latent_dimension))
        layers.append(nn.SiLU())

        for i in range(args.decoder_layers):
            layers.append(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(2*args.latent_dimension, 2))
        self.model = nn.Sequential(*layers)

        # harmonic embedding
        self.harmonic_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension)
        # self.harmonic = torch.linspace(args.lower_bound, args.upper_bound, args.n_harmonics,
        #                                requires_grad=False, dtype=torch.long).cuda()

    def forward(self, x):
        # x (B, E)
        B, E = x.size()

        x = torch.broadcast_to(x.unsqueeze(1), (B, self.n_harmonics, E))  # (B, H, E)

        harmonic = torch.linspace(self.lower_bound, self.upper_bound, self.n_harmonics, requires_grad=False, dtype=torch.long).cuda()
        harmonics = self.harmonic_embedding(harmonic - 1)  # (H, E)
        harmonics = torch.broadcast_to(harmonics.unsqueeze(0), (B, self.n_harmonics, E))  # (B, H, E)
        x = x + harmonics

        output = self.model(x)
        return output

class FNP_UnconditionQueryDecoder(nn.Module):
    def __init__(self, args):
        super(FNP_UnconditionQueryDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step

        # harmonic embedding
        self.coeff_generator = UnconditionQueryGenerator(args)
        self.nonperiodic_decoder = NonperiodicDecoder(args)

    def forward(self, target_x, r):
        # target_x (B, S, 1)  r (B, E)
        nonperiodic_signal = self.nonperiodic_decoder(r, target_x).squeeze(-1)  # (B, S)

        coeffs = self.coeff_generator(r)   # (B, H, 2)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]

        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        periodic_signal = (cos_x + sin_x)   # (B, S)
        return nonperiodic_signal + periodic_signal






"""
class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_per_attn):
        super(MultiheadAttention, self).__init__()
        self.num_hidden_per_attn = num_hidden_per_attn
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1,2))
        attn = attn / math.sqrt(self.num_hidden_per_attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        result = torch.bmm(attn, value)
        return result, attn



class CrossAttention(nn.Module):
    def __init__(self, num_hidden, num_attn):
        super(CrossAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = int(num_hidden / num_attn)
        self.num_attn = num_attn

        self.key = nn.Linear(1, num_hidden)
        self.query = nn.Linear(1, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

    def forward(self, query, key, value):
        batch_size = key.size(0)  # context B
        seq_k = key.size(1)       # context S
        seq_q = query.size(1)     # target S

        # Multihead attention
        key = self.key(key).view(batch_size, seq_k, self.num_attn, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.num_attn, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.num_attn, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attn = self.multihead(query, key, value)

        # Concate all multihead context vector
        #result = result.view(self.num_attn, batch_size, seq_q, self.num_hidden_per_attn)
        #result = result.permute(1, 2, 0, 3).view(batch_size, seq_q, -1)
        return result



class NP_Trans_Encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, deterministic):
        super(NP_Trans_Encoder, self).__init__()
        self.deterministic = deterministic

        self.input_projection = nn.Linear(2, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim, dropout=0.1)
        self.model = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        if not deterministic:
            self.linear1 = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_dim, 2*hidden_dim))


    def forward(self, content_x, content_y):
        # Concat content_x (B, S, 1) and content_y (B, S , 1)
        input = torch.cat((content_x, content_y), dim=-1)  # (B, S, 2)
        input = self.input_projection(input)
        input = input.permute(1, 0, 2)   # (S, B, E)

        output = self.model(src=input)  # (S, B, E)

        if self.deterministic:
            return self.output_fc(output.permute(1, 0, 2))  # (B, S, E)  확인해보기
        else:
            # Aggregator: take the mean over all points
            output = output.mean(dim=0) # (B, hidden_dim)
            # Apply further linear layer to output latent mu and log sigma
            z = self.linear1(output)
            z0, qz0_mean, qz0_logvar = self.reparameterization(z)
            return z0, qz0_mean, qz0_logvar

    def reparameterization(self, z):
        latent_dim = z.shape[1] // 2
        qz0_mean, qz0_logvar = z[:, :latent_dim], z[:, latent_dim:]
        qz0_logvar = 0.1 + 0.9 * torch.sigmoid(qz0_logvar)
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar


class NP_linear_encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, deterministic):
        super(NP_linear_encoder, self).__init__()
        self.deterministic = deterministic

        self.input_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim))

        self.output_fc = nn.Linear(hidden_dim, output_dim)
        if not self.deterministic:
            self.linear1 = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_dim, 2*hidden_dim))

    def forward(self, content_x, content_y):
        # Concat content_x (B, S, 1) and content_y (B, S, 1)
        input = torch.cat((content_x, content_y), dim=-1)   # (B, S, 2)
        output = self.input_projection(input)   # (B, S, hidden_dim)

        if self.deterministic:
            return self.output_fc(output)  # (B, S, output_dim)
        else:
            # Aggregator: take the mean over all points
            output = output.mean(dim=1)    # add according to the sequence
            # Apply further linear layer to output latent mu and log sigma
            z = self.linear1(output)
            z0, qz0_mean, qz0_logvar = self.reparameterization(z)
            return z0, qz0_mean, qz0_logvar


    def reparameterization(self, z):
        latent_dim = z.shape[1] // 2
        qz0_mean, qz0_logvar = z[:, :latent_dim], z[:, latent_dim:]
        qz0_logvar = 0.1 + 0.9 * torch.sigmoid(qz0_logvar)
        epsilon = torch.randn(qz0_mean.size()).to(z.device)
        z0 = epsilon * qz0_logvar + qz0_mean
        return z0, qz0_mean, qz0_logvar






class NP_Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(NP_Decoder, self).__init__()
        # 128*2 + 2
        self.target_projection = nn.Linear(1, hidden_dim)
        self.embedding = nn.Linear(hidden_dim*3, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim)
        self.model = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, target_x, attended_r, z0):
        target_x = self.target_projection(target_x)
        input = torch.cat((target_x, attended_r, z0), dim=-1) # (B, S, E)
        input = self.embedding(input)
        input = input.permute(1, 0, 2)

        hidden = self.model(src=input)   # (S, B, E)
        hidden = self.output_fc(hidden.permute(1, 0, 2))   # (B, S, 2)

        mu = hidden[..., 0]
        sigma = 0.1 + 0.9*F.softplus(hidden[..., 1])
        return mu, sigma


class AttentiveNP(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentiveNP, self).__init__()
        self.deterministic_encoder = NP_Encoder(hidden_dim, deterministic=True)
        self.latent_encoder = NP_Encoder(hidden_dim, deterministic=False)
        self.attention = CrossAttention(hidden_dim, num_attn=1)
        self.decoder = NP_Decoder(hidden_dim)

    def forward(self, content_x, content_y, target_x, target_y=None):
        '''
        every element shape (B, S)
        '''
        num_targets = target_x.size(1)
        content_x = content_x.unsqueeze(-1) ; content_y = content_y.unsqueeze(-1)
        target_x = target_x.unsqueeze(-1) ; target_y = target_y.unsqueeze(-1) if target_y is not None else target_y
        prior, prior_mu, prior_var = self.latent_encoder(content_x, content_y)

        # For training
        if target_y is not None:
            posterior, posterior_mu, posterior_var = self.latent_encoder(target_x, target_y)
            z = posterior

        # For Test
        else:
            z = prior # (B, 2E)

        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # (B, S, E)
        r = self.deterministic_encoder(content_x, content_y)

        ## attention
        attn_result = self.attention(target_x, content_x, r)

        # Decoder
        mu, sigma = self.decoder(target_x, attn_result, z)

        # For training
        if target_y is not None:
            # get loss
            mse_loss = log_normal_pdf(target_y.squeeze(-1), mu, sigma).mean()
            kl_loss = kl_divergence(prior_mu, prior_var, posterior_mu, posterior_var)
            loss = -mse_loss + kl_loss

        else:
            mse_loss = None
            kl_loss = None
            loss = None

        return mu, sigma, mse_loss, kl_loss, loss



class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        # layers = []
        # layers.append(nn.Linear(latent_dimension, 2 * coeffs_size))
        # layers.append(nn.SiLU())
        #
        # for i in range(n_layers):
        #     layers.append(nn.Linear(2 * coeffs_size, 2 * coeffs_size))
        #     layers.append(nn.SiLU())
        #
        # layers.append(nn.Linear(2*coeffs_size, coeffs_size))
        # self.model = nn.Sequential(*layers)

        self.fc1 = nn.Linear(latent_dimension, 2*latent_dimension)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(2*latent_dimension, 2*latent_dimension)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(2*latent_dimension, 2*latent_dimension)
        self.act3 = nn.SiLU()
        self.fc4 = nn.Linear(2*latent_dimension, coeffs_size)

    def forward(self, x):
        # input latent vector
        # return self.model(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.fc4(x)



def FourierExpansion(n_range, s):
    s_n_range = s * n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis


class FNP_Decoder(nn.Module):
    def __init__(self, args):
        super(FNP_Decoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.coeffs_size = args.in_features*args.out_features*args.n_harmonics*args.n_eig
        self.skip_step = args.skip_step

        # self.coeff_generator = CoeffDecoder(n_layers=3, latent_dimension=args.latent_dimension + args.num_label, coeffs_size=self.coeffs_size)
        self.coeff_generator = CoeffDecoder(latent_dimension=args.latent_dimension + args.num_label, coeffs_size=self.coeffs_size)

    def forward(self, target_x, r):
        # target_x  (B, S, 1), r (B, E)
        assert r.size(-1) == self.coeff_generator.latent_dimension, 'Dimension does not match'
        coeffs = self.coeff_generator(r)  # (B, C)
        self.coeffs = coeffs

        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)  # (B, S, 1)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)  # (B, S, 1)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)   # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)   # (B, S, H)

        cos_x = torch.mul(cos_x, coeffs[:, :int(self.coeffs_size/2)].unsqueeze(1))
        sin_x = torch.mul(sin_x, coeffs[:, int(self.coeffs_size/2):].unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)  # (B, S)
        return cos_x + sin_x   # (B, S)




# Changed from Transformer Encoder to Linear Encoder
class FNP(nn.Module):
    def __init__(self, args):
        super(FNP, self).__init__()
        self.deterministic_encoder = NP_linear_encoder(args.encoder_hidden_dim, args.latent_dimension, deterministic=True)
        self.decoder = FNP_Decoder(args)

    def forward(self, context_x, context_y, target_x, target_y=None):
        '''
        every element shape of (B, S)
        '''
        num_targets = target_x.size(1)
        context_x = context_x.unsqueeze(-1) ; context_y = context_y.unsqueeze(-1)
        target_x = target_x.unsqueeze(-1) #; target_y = target_y.unsqueeze(-1) if target_y is not None else target_y

        r = self.deterministic_encoder(context_x, context_y)   # (B, S, E)

        mu = self.decoder(target_x, r.mean(1))

        # for Format
        sigma = None
        kl_loss = None
        mse_loss = None

        if target_y is not None:
            loss = nn.MSELoss()(mu, target_y)
        else:
            loss = None

        return mu, sigma, mse_loss, kl_loss, loss


class ConditionalFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension

        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.latent_dimension, rnn_hidden_dim=args.encoder_hidden_dim)
        elif args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = ConvEncoder(args=args)

        self.decoder = FNP_Decoder(args)

    def sampling(self, t, x):
        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
            t = t[:, sample_idxs]  # (150)
            x = x[:, sample_idxs]
        # elif self.dataset_type == 'NSynth':
        #     sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 8000, replace=False)))[0]   # sampling..
        #     t = t[:, sample_idxs]
        #     x = x[:, sample_idxs]
        # not sampling for ECG for now
        return t, x


    def forward(self, t, x, label, sampling):
        # t (B, 300)  x (B, S, 1)  label(B)
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        if sampling:
            sampled_t, sampled_x = self.sampling(t, x)
            memory, z, qz0_mean, qz0_logvar = self.encoder(sampled_x, label_embed, span=sampled_t[0])
        else:
            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])

        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        # if self.dataset_type == 'NSynth':
        #     decoded_traj = self.decoder(sampled_t.unsqueeze(-1), z)
        #     mse_loss = nn.MSELoss()(decoded_traj, sampled_x.squeeze(-1))
        # else:
        decoded_traj = self.decoder(t.unsqueeze(-1), z)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        return mse_loss, kl_loss

    def predict(self, t, x, label, test_t):
        with torch.no_grad():
            B = x.size(0)
            label_embed = torch.zeros(B, self.num_label).cuda()
            label_embed[range(B), label] = 1

            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])
            z = torch.cat((z, label_embed), dim=-1)
            decoded_traj = self.decoder(test_t.unsqueeze(-1), z)

        return decoded_traj

    def inference(self, t, label):
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim).cuda()
            label_embed = torch.zeros(1, self.num_label).cuda()
            label_embed[0, label] = 1
            z = torch.cat((z, label_embed), dim=-1)

            decoded_traj = self.decoder(t.unsqueeze(-1), z)
        return decoded_traj

######################################################################################
class QueryCoeffGenerator(nn.Module):
    def __init__(self, args):
        super(QueryCoeffGenerator, self).__init__()
        self.n_harmonics = args.n_harmonics
        self.num_label = args.num_label

        # self.model1 = nn.Sequential(nn.Linear(args.latent_dimension + args.num_label, 2*args.latent_dimension),
        #                             nn.SiLU())
        self.model1 = nn.Sequential(nn.Linear(args.latent_dimension + args.num_label, 2*args.latent_dimension),
                                    nn.SiLU())
        # self.model2 = nn.Sequential(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension),
        #                             nn.SiLU())
        # self.model3 = nn.Sequential(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension),
        #                             nn.SiLU())
        # self.model4 = nn.Linear(2*args.latent_dimension, 2)
        layers = []
        for i in range(args.decoder_layers):
            layers.append(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(2*args.latent_dimension, 2))    # 2 amps for now
        self.model2 = nn.Sequential(*layers)

        # harmonics embedding
        self.harmonic_embedding = nn.Embedding(args.n_harmonics, 2*args.latent_dimension)
        #self.harmonic_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension)
        self.harmonic = torch.linspace(args.lower_bound, args.upper_bound, args.n_harmonics,
                                       requires_grad=False, dtype=torch.long, device=torch.device('cuda:0'))

    def forward(self, x):
        # x (B, E)   label (B, num_label)
        B, E = x.size()
        E = E - self.num_label
        # first broadcast to the number of harmonics
        x = torch.broadcast_to(x.unsqueeze(1), (B, self.n_harmonics, E + self.num_label))   # (B, H, E)
        # x = torch.broadcast_to(x.unsqueeze(1), (B, self.n_harmonics, E))  # (B, H, E)
        # x + harmonics (comment, not code)

        # harmonics = self.harmonic_embedding(self.harmonic-1)  # (H, E)
        # harmonics = torch.broadcast_to(harmonics.unsqueeze(0), (B, self.n_harmonics, E))
        # x = x + harmonics  # (B, H, E)

        # label = torch.broadcast_to(label.unsqueeze(1), (B, self.n_harmonics, self.num_label))  # (B, H, L)
        # x = torch.cat((x, label), dim=-1)


        # model1
        x = self.model1(x)  # (B, H, 2E)    # E가 2의 배수인지 확인하기!

        # add harmonic embedding
        with torch.no_grad():   # freeze
            print('harmonic embedding is frozen')
            harmonics = self.harmonic_embedding(self.harmonic - 1)  # (H, 2E)
            harmonics = torch.broadcast_to(harmonics.unsqueeze(0), (B, self.n_harmonics, 2*E))   # (B, H, 2E)

        # add harmonic embedding and model1 output
        x = x + harmonics   # (B, H, 2E)
        # x = self.model2(x) + harmonics
        # x = self.model3(x) + harmonics
        # x = self.model4(x)

        # pass through model2
        x = self.model2(x)  # (B, H, 2)

        return x


class FNP_QueryDecoder(nn.Module):
    def __init__(self, args):
        super(FNP_QueryDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step
        # self.coeffs_size = args.in_features*args.out_features*args.n_harmonics*args.n_eig

        # harmonic embedding
        self.coeff_generator = QueryCoeffGenerator(args)

    def forward(self, target_x, r):
        # target_x (B, S, 1)  r (B, E)  label (B, num_label)
        B = r.size(0)
        coeffs = self.coeff_generator(r)   # (B, H, 2)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]


        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        return cos_x + sin_x


class FNP_QueryShiftDecoder(nn.Module):
    def __init__(self, args):
        super(FNP_QueryShiftDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step

        # harmonic embedding
        self.coeff_generator = QueryCoeffGenerator(args)
        # shift generator
        self.shift_generator = nn.Sequential(nn.Linear(args.encoder_embedding_dim, args.latent_dimension),
                                             nn.SiLU(),
                                             nn.Linear(args.latent_dimension, 1),
                                             nn.Tanh())

    def forward(self, target_x, r, memory):
        # target_x (B, S, 1)  r (B, E)  memory (B, E), H = num of harmonics
        # generate coeffs
        coeffs = self.coeff_generator(r)  # (B, H, 2)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]

        # generate shift
        shift = self.shift_generator(memory).unsqueeze(-1)  # (B, 1, 1)
        # add target_x and shift
        self.shift = shift
        target_x = target_x + shift

        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        return cos_x + sin_x


class FNPShiftDecoder(nn.Module):
    def __init__(self, args):
        super(FNPShiftDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step
        self.coeffs_size = args.in_features*args.out_features*args.n_harmonics*args.n_eig

        # harmonic embedding
        self.coeff_generator = CoeffDecoder(latent_dimension=args.latent_dimension + args.num_label, coeffs_size=self.coeffs_size)
        # shift generator
        self.shift_generator = nn.Sequential(nn.Linear(args.encoder_embedding_dim, args.latent_dimension),
                                             nn.SiLU(),
                                             nn.Linear(args.latent_dimension, 1),
                                             nn.Tanh())

    def forward(self, target_x, r, memory):
        # target_x (B, S, 1)  r (B, E) memroy (B, E), H = num of harmonics
        # generate coeffs
        coeffs = self.coeff_generator(r)
        self.coeffs = coeffs

        # generate shift
        shift = self.shift_generator(memory).unsqueeze(-1)  # (B, 1, 1)
        # add target_x and shift
        target_x = target_x + shift

        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)

        cos_x = torch.mul(cos_x, coeffs[:, :int(self.coeffs_size/2)].unsqueeze(1))
        sin_x = torch.mul(sin_x, coeffs[:, int(self.coeffs_size/2):].unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        return cos_x + sin_x


class FNP_QueryContinualDecoder(nn.Module):
    def __init__(self, args):
        super(FNP_QueryContinualDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step
        self.mask_reverse = args.mask_reverse  # False -> low freq부터 배우기, True -> high freq부터 배우기

        # harmonic embedding
        self.coeff_generator = QueryCoeffGenerator(args)

    def coeff_mask(self, coeff_num):
        # coeff_num : number of harmonics to learn
        # return matrix of (H, 2) consist of 0 and 1 where 1 is for learnable coeffs
        zeros = torch.zeros(((self.n_harmonics - coeff_num), 2), device=torch.device('cuda:0'), requires_grad=False)
        ones = torch.ones((coeff_num, 2), device=torch.device('cuda:0'), requires_grad=False)
        if not self.mask_reverse:  # low freq 부터 배우기
            mask = torch.cat((ones, zeros), dim=0)   # (H, 2)
        elif self.mask_reverse:
            mask = torch.cat((zeros, ones), dim=0)  # (H, 2)
        return mask


    def forward(self, target_x, r, coeff_num):
        # target_x (B, S, 1)   r (B, E)
        B = r.size(0)
        coeffs = self.coeff_generator(r)  # (B, H, 2)
        mask = self.coeff_mask(coeff_num)  # (B, 2)
        mask = torch.broadcast_to(mask, (B, self.n_harmonics, 2))   # (B, H, 2)
        coeffs = torch.mul(coeffs, mask)  # (B, H, 2)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]

        # maks cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        return cos_x + sin_x


class UnconditionQueryGenerator(nn.Module):
    def __init__(self, args):
        super(UnconditionQueryGenerator, self).__init__()
        self.n_harmonics = args.n_harmonics
        self.lower_bound, self.upper_bound = args.lower_bound, args.upper_bound

        layers = []
        layers.append(nn.Linear(args.latent_dimension, 2*args.latent_dimension))
        layers.append(nn.SiLU())

        for i in range(args.decoder_layers):
            layers.append(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(2*args.latent_dimension, 2))
        self.model = nn.Sequential(*layers)

        # harmonic embedding
        self.harmonic_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension)
        # self.harmonic = torch.linspace(args.lower_bound, args.upper_bound, args.n_harmonics,
        #                                requires_grad=False, dtype=torch.long).cuda()

    def forward(self, x):
        # x (B, E)
        B, E = x.size()

        x = torch.broadcast_to(x.unsqueeze(1), (B, self.n_harmonics, E))  # (B, H, E)

        harmonic = torch.linspace(self.lower_bound, self.upper_bound, self.n_harmonics, requires_grad=False, dtype=torch.long).cuda()
        harmonics = self.harmonic_embedding(harmonic - 1)  # (H, E)
        harmonics = torch.broadcast_to(harmonics.unsqueeze(0), (B, self.n_harmonics, E))  # (B, H, E)
        x = x + harmonics

        output = self.model(x)
        return output


class FNP_UnconditionQueryDecoder(nn.Module):
    def __init__(self, args):
        super(FNP_UnconditionQueryDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step

        # harmonic embedding
        self.coeff_generator = UnconditionQueryGenerator(args)

    def forward(self, target_x, r):
        # target_x (B, S, 1)  r (B, E)
        coeffs = self.coeff_generator(r)   # (B, H, 2)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]

        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        return cos_x + sin_x
"""