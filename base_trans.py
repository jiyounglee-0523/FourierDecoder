# reference : https://pytorch.org/tutorials/beginner/translation_transformer.html
import torch
import torch.nn as nn

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time
import argparse
import random

from datasets.cond_dataset import get_dataloader
from utils.model_utils import count_parameters
from utils.loss import kl_divergence, log_normal_pdf, normal_kl
from models.encoder import TransformerEncoder
from trainer.ConditionalTrainer import ConditionalBaseTrainer

# class TransformerEncoder(nn.Module):
#     def __init__(self, args):
#         super(TransformerEncoder, self).__init__()
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.num_label = args.num_label
#         self.latent_dim = args.latent_dimension
#
#         self.embedding = nn.Linear(1, args.encoder_embedding_dim)
#         self.label_embedding = nn.Linear(args.num_label, args.encoder_embedding_dim, bias=False)
#         self.pos_embedding = nn.Linear(1, args.encoder_embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
#                                                    dropout=args.dropout)
#         self.model = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_blocks)
#         self.output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)
#
#     def forward(self, x, label, span):
#         # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
#         B = x.size(0) ; S = span.size(0)
#
#         # add 0 in span
#         span = torch.cat((torch.zeros(1).cuda(), span), dim=0)
#
#         B = x.size(0) ; S = span.size(0)
#         label = self.label_embedding(label)  # (B, E)
#         x = self.embedding(x)  # (B, S, E)
#         x = torch.cat((label.unsqueeze(1), x), dim=1)  # (B, S+1, E)
#
#         # add positional embedding
#         span = self.pos_embedding(torch.broadcast_to(span, (B, S)).unsqueeze(-1))   # (B, S, E)
#         x = x + span
#         x = self.dropout(x)
#
#         x = x.permute(1, 0, 2)   # (S, B, E)
#         memory = self.model(src=x)  # (S, B, E)
#
#         z = self.output_fc(memory)  # (S, B, 2E)
#         z = z.mean(0)  # (B, 2E)
#         z0, qz0_mean, qz0_logvar = self.reparameterization(z)  # (B, E)
#
#         return memory, z0, qz0_mean, qz0_logvar
#
#     def reparameterization(self, z):
#         qz0_mean = z[:, :self.latent_dim]
#         qz0_logvar = z[:, self.latent_dim:]
#         epsilon = torch.randn(qz0_mean.size()).to(z.device)
#         z0 = epsilon * qz0_logvar + qz0_mean
#         return z0, qz0_mean, qz0_logvar


class AutoRegressiveTransformerDecoder(nn.Module):
    def __init__(self, args):
        super(AutoRegressiveTransformerDecoder, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = nn.Linear(1, args.encoder_embedding_dim)
        self.label_embedding = nn.Linear(args.num_label + args.latent_dimension, args.encoder_embedding_dim, bias=False)

        # model
        self.pos_embedding = nn.Linear(1, args.encoder_embedding_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
                                                   dropout = args.dropout)
        self.model = nn.TransformerEncoder(decoder_layer, num_layers=args.encoder_blocks)
        self.output_fc = nn.Linear(args.encoder_embedding_dim, 1, bias=False)

    def forward(self, x, r, target_x):
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
        output = self.output_fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def auto_regressive(self, r, target_x):
        # r (B, E)  target_x (B, S, 1)
        #with torch.no_grad():
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



class BaseTransformer(nn.Module):
    def __init__(self, args):
        super(BaseTransformer, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension

        self.encoder = TransformerEncoder(args)
        self.decoder = AutoRegressiveTransformerDecoder(args)


    def sampling(self, t, x):
        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
            t = t[:, sample_idxs]
            x = x[:, sample_idxs]
        elif self.dataset_type == 'NSynth':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 1600, replace=False)))[0]   # change sampled freq
            t = t[:, sample_idxs]
            x = x[:, sample_idxs]
        return t, x


    def forward(self, t, x, label, sampling):
        # t (B, 300)  x (B, S, 1)  label (B)
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

        # conat label information
        z = torch.cat((z, label_embed), dim=-1)

        decoded_traj = self.decoder(x, z, t.unsqueeze(-1))
        mse_loss = nn.MSELoss()(decoded_traj[:, :-1, :], x)
        return mse_loss, kl_loss


    def predict(self, t, x, label, sampling, test_t):
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

        z = torch.cat((z, label_embed), dim=-1)
        decoded_traj = self.decoder.auto_regressive(z, test_t.unsqueeze(-1))
        decoded_traj = torch.stack(decoded_traj, dim=0).permute(1, 0, 2)
        try:
            mse_loss = nn.MSELoss()(decoded_traj, x)
            return mse_loss, kl_loss, decoded_traj
        except:
            return decoded_traj



class BaseTransTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(BaseTransTrainer, self).__init__(args)

        self.model = BaseTransformer(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.notes)}')

        if not self.debug:
            wandb.init(project='conditionalODE')
            wandb.config.update(args)


    def train(self):
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                freq = sample['freq']
                amp = sample['amp']
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                teacher_mse_loss, teacher_kl_loss = self.model(orig_ts, samp_sin, label, sampling=True)
                teacher_loss = teacher_mse_loss + teacher_kl_loss
                teacher_loss.backward()
                self.optimizer.step()

                # inference
                self.optimizer.zero_grad(set_to_none=True)
                infer_mse_loss, infer_kl_loss, infer_decoded_traj = self.model.predict(orig_ts, samp_sin, label, sampling=True, test_t=orig_ts)
                infer_loss = infer_mse_loss + infer_kl_loss
                infer_loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'teacher_train_loss': teacher_loss,
                               'teacher_train_kl_loss': teacher_kl_loss,
                               'teacher_train_mse_loss': teacher_mse_loss,
                               'infer_train_loss': infer_loss,
                               'infer_train_kl_loss': infer_kl_loss,
                               'infer_train_mse_loss': infer_mse_loss
                               })



            endtime = time.time()
            # print(f'[Time] : {endtime-starttime}')
            # self.plot_sin(samp_sin[0], orig_ts[0], freq[0], amp[0], label[0])

            eval_loss, eval_mse, eval_kl = self.evaluation()
            print(f'Train MSE: {teacher_mse_loss:.4f}   Train_infer: {infer_mse_loss:.4f}   Eval MSE: {eval_mse:.4f}')

            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_kl_loss': eval_kl,
                           'eval_mse_loss': eval_mse})

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print(f'Model parameter saved at {n_epoch}')

        self.test()


    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, decoded_traj = self.model.predict(orig_ts, samp_sin, label, sampling=True, test_t=orig_ts)
                loss = mse_loss + kl_loss
                avg_eval_loss += (loss.item() / len(self.eval_dataloader))
                avg_eval_mse += (mse_loss.item() / len(self.eval_dataloader))
                avg_kl += (kl_loss.item() / len(self.eval_dataloader))

        return avg_eval_loss, avg_eval_mse, avg_kl


    def test(self):
        self.model.eval()
        avg_test_loss = 0.
        avg_test_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloder):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, decoded_traj = self.model.predict(orig_ts, samp_sin, label, sampling=True, test_t=orig_ts)
                loss = mse_loss + kl_loss
                avg_test_loss += (loss.item() / len(self.test_dataloder))
                avg_test_mse += (mse_loss.item() / len(self.test_dataloder))
                avg_kl += (kl_loss.item() / len(self.test_dataloder))

        if not self.debug:
            wandb.log({'test_loss': avg_test_loss,
                       'test_mse': avg_test_mse,
                       'test_kl': avg_kl})

    def plot_sin(self, samp_sin, orig_ts, freq, amp, label):
        self.model.eval()

        # reconstruction
        samp_sin = samp_sin.unsqueeze(0)
        orig_ts = orig_ts.unsqueeze(0)
        test_tss = torch.Tensor(np.linspace(0, 5, 400)).to(samp_sin.device)
        with torch.no_grad():
            decoded_traj = self.model.predict(orig_ts, samp_sin, label.unsqueeze(0), sampling=True, test_t=test_tss)

        test_ts = test_tss.cpu().numpy()
        orig_sin = amp[0] * np.sin(freq[0] * test_ts * 2 * np.pi) + amp[1] * np.sin(freq[1] * test_ts * 2 * np.pi) + \
                   amp[2] * np.sin(freq[2] * test_ts * 2 * np.pi) + \
                   amp[3] * np.cos(freq[3] * test_ts * 2 * np.pi) + amp[4] * np.cos(freq[4] * test_ts * 2 * np.pi) + \
                   amp[5] * np.cos(freq[5] * test_ts * 2 * np.pi)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(test_ts, orig_sin.cpu().numpy(), 'g', label='true trajectory')
        ax.scatter(orig_ts.cpu().numpy(), samp_sin[0].squeeze(-1).cpu().numpy(), s=5, label='sampled points')
        ax.plot(test_ts, decoded_traj.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        if not self.debug:
            wandb.log({'reconstruction': wandb.Image(plt)})
        plt.close('all')



def main():
    parser = argparse.ArgumentParser()

    # Encoder
    parser.add_argument('--encoder_embedding_dim', type=int, default=128)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('--encoder_attnheads', type=int, default=2)
    parser.add_argument('--encoder_blocks', type=int, default=3)

    # Decoder
    parser.add_argument('--latent_dimension', type=int, default=128)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--path', type=str, default='/home/edlab/jylee/generativeODE/output/baseline/transformer/', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth'], default='sin')
    parser.add_argument('--notes', type=str, default='example')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device_num', type=str, default='0')
    args = parser.parse_args()

    if args.dataset_type == 'sin':
        args.num_label = 4
    elif args.dataset_type == 'NSynth':
        args.num_label = 7
    else:
        raise NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = BaseTransTrainer(args)
    trainer.train()




if __name__ == '__main__':
    main()






# class Transformer(nn.Module):
#     def __init__(self, args):
#         super(Transformer, self).__init__()
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.num_label = args.num_label
#         self.dataset_type = args.dataset_type
#
#         self.latent_dim = args.latent_dimension
#         self.enc_embedding = nn.Linear(1, args.encoder_embedding_dim)
#         self.enc_label_embedding = nn.Linear(args.num_label, args.encoder_embedding_dim, bias=False)
#         self.dec_embedding = nn.Linear(1, args.encoder_embedding_dim)
#         self.dec_label_embedding = nn.Linear(args.num_label + args.latent_dimension, args.encoder_embedding_dim, bias=False)
#
#         # model
#         self.enc_pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
#         self.dec_pos_encoder = nn.Linear(1, args.encoder_embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
#                                                    dropout=args.dropout)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_blocks)
#         self.enc_output_fc = nn.Linear(args.encoder_embedding_dim, 2*args.latent_dimension)
#
#
#         decoder_layer = nn.TransformerDecoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
#                                                    dropout=args.dropout)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.encoder_blocks)
#         self.dec_output_fc = nn.Linear(args.encoder_embedding_dim, 1)
#
#     def forward(self, span, x, label, sampling):
#         # x shape of (B, S, 1), label shape of (B, num_label), span shape of (S)
#         sampled_span, sampled_x, label = self.data_check(span, x, label, sampling)
#
#         # add 0 in span
#         sampled_span = torch.cat((torch.zeros(1).cuda(), sampled_span), dim=0)
#
#         B = x.size(0) ; S = span.size(0)
#         label = self.enc_label_embedding(label)   # (B, E)
#         enc_x = self.enc_embedding(x)  # (B, S, E)
#
#         # concat label with x
#         enc_x = torch.cat((label.unsqueeze(1), enc_x), dim=1)   # (B, S+1, E)
#
#         # add positional embedding
#         enc_span = self.enc_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1)) # (B, S, E)
#         enc_x = enc_x + enc_span
#         enc_x = self.dropout(enc_x)
#
#         enc_x = enc_x.permute(1, 0, 2)   # (S, B, E)
#         memory = self.encoder(src=enc_x)   # (S, B, E)
#         z = self.enc_output_fc(memory)  # (S, B, 2E)
#         z = z.mean(0)  # (B, 2E)
#         z0, qz0_mean, qz0_logvar = self.reparameterization(z)   # (B, E)
#
#         z0 = torch.cat((z0, label), dim=-1)  # (B, E+num_label)
#         z0 = self.dec_label_embedding(z0)  # (B, E)
#         dec_x = self.dec_embedding(x)  # (B, S, E)
#         dec_x = torch.cat((z0.unsqueeze(1), dec_x), dim=1)  # (B, S+1, E)
#
#         # add positional embedding
#         dec_span = self.dec_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))  # (B, S, E)
#         dec_x = dec_x + dec_span
#         dec_x = self.dropout(dec_x)
#
#         dec_mask = self.generate_square_subsequent_mask(S)  # (S, S)
#         dec_x = dec_x.permute(1, 0, 2)
#         output = self.decoder(tgt=dec_x, memory=memory, tgt_mask=dec_mask)  # (S, B, E)
#         output = self.dec_output_fc(output).permute(1, 0, 2)  # (B, S, 1)
#
#         return output, qz0_mean, qz0_logvar
#
#     def inference(self, span, x, label, sampling):
#         with torch.no_grad():
#             span = torch.cat((torch.zeros(1).cuda(), span), dim=0)
#
#             B = x.size(0) ; S = span.size(0)
#             label = self.enc_label_embedding(label)  # (B, E)
#             enc_x = self.enc_embedding(x)  # (B, S, E)
#
#             # concat label with x
#             enc_x = torch.cat((label.unsqueeze(1), enc_x), dim=1)   # (B, S+1, E)
#
#             # add positional embedding
#             enc_span = self.enc_pos_encoder(torch.broadcast_to(span, (B, S)).unsqueeze(-1))  # (B, S, E)
#             enc_x = enc_x + enc_span
#             enc_x = self.dropout(enc_x)
#
#             enc_x = enc_x.permute(1, 0, 2)  # (S, B, E)
#             memory = self.encoder(src=enc_x) # (S, B, E)
#
#             z = self.enc_output_fc(memory)  # (S, B, 2E)
#             z = z.mean(0)  # (B, 2E)
#             z0, qz0_mean, qz0_logvar = self.reparameterization(z)  # (B, E)
#
#             z0 = torch.cat((z0, label), dim=1)  # (B, E+num_label)
#             z0 = self.dec_label_embedding(z0)  # (B, E)
#             dec_x = z0.unsqueeze(0)  # (1, B, E)
#
#             for i in range(S-1):
#                 tgt_mask = self.generate_square_subsequent_mask(dec_x.size(0))
#                 dec_span = self.dec_pos_encoder(torch.broadcast_to(span[:i], (B, i+1)).unsqueeze(-1))  # double check
#                 dec_x = dec_x + dec_span
#                 output = self.decoder(dec_x, memory=memory, tgt_mask=tgt_mask)   # (S, B, E)
#                 output = self.dec_output_fc(output)  # (S, B, 1)
#                 dec_x = torch.cat((dec_x, output[0].unsqueeze(0)), dim=0)
#
#             return dec_x.permute(1, 0, 2)
#
#     def data_check(self, span, x, label, sampling):
#         B = x.size(0)
#
#         # one-hot label
#         label_embed = torch.zeros(B, self.num_label).cuda()
#         label_embed[range(B), label] = 1
#
#         if sampling:
#             sampled_t, sampled_x = self.sampling(span, x)
#             return sampled_t[0], sampled_x, label_embed
#         else:
#             return span[0], x, label_embed
#
#
#     def sampling(self, t, x):
#         if self.dataset_type == 'sin':
#             sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
#             t = t[:, sample_idxs]  # (150)
#             x = x[:, sample_idxs]
#         elif self.dataset_type == 'NSynth':
#             sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 1600, replace=False)))[0]   # sampling..
#             t = t[:, sample_idxs]
#             x = x[:, sample_idxs]
#         return t, x
#
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask.cuda()
#
#     def reparameterization(self, z):
#         qz0_mean = z[:, :self.latent_dim]
#         qz0_logvar = z[:, self.latent_dim:]
#         epsilon = torch.randn(qz0_mean.size()).to(z.device)
#         z0 = epsilon * qz0_logvar + qz0_mean
#         return z0, qz0_mean, qz0_logvar

# class pTransformerDecoder(nn.Module):
#     def __init__(self, args):
#         super(pTransformerDecoder, self).__init__()
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.embedding = nn.Linear(1, args.encoder_embedding_dim)
#         self.label_embedding = nn.Linear(args.num_label + args.latent_dimension, args.encoder_embedding_dim, bias=False)
#
#         # model
#         self.pos_embedding = nn.Linear(1, args.encoder_embedding_dim)
#         decoder_layer = nn.TransformerDecoderLayer(d_model=args.encoder_embedding_dim, nhead=args.encoder_attnheads, dim_feedforward=args.encoder_hidden_dim,
#                                                    dropout=args.dropout)
#         self.model = nn.TransformerDecoder(decoder_layer, num_layers=args.encoder_blocks)
#         self.output_fc = nn.Linear(args.encoder_embedding_dim, 1)
#
#     def forward(self, memory, z0, span, x, label):
#         # memory (S, B, E),  z0 (B, E),  x (B, S, 1)  label (B, num_label)
#         # concat 0 in span
#         span = torch.cat((torch.zeros(1).cuda(), span), dim=0)
#
#         B = x.size(0) ; S = span.size(0)
#
#         z0 = torch.cat((z0, label), dim=-1)   # (B, L+num_label)
#         z0 = self.label_embedding(z0) # (B, E)
#         x = self.embedding(x)   # (B, S, E)
#         x = torch.cat((z0.unsqueeze(1), x), dim=1)  # (B, S+1, E)
#
#         # add positional embedding
#         span = self.pos_embedding(torch.broadcast_to(span, (B, S)).unsqueeze(-1))   # (B, S, E)
#         x = x + span
#         x = self.dropout(x)
#
#         mask = self.generate_square_subsequent_mask(span.size(1))
#         x = x.permute(1, 0, 2)  # (S, B, E)
#         output = self.model(tgt=x, memory=memory, tgt_mask=mask)  # (S, B, E)
#         output = self.output_fc(output)  # (S, B, 1)
#         return output.permute(1, 0, 2)
#
#     def auto_regressive(self, memory, z0, span, x, label):
#         # memory (S, B, E), z0 (B, E), x (B, S, 1), label (B, num_label)
#         with torch.no_grad():
#             # concat 0 in span
#             span = torch.cat((torch.zeros(1).cuda(), span), dim=0)
#
#             B = x.size(0) ; S = span.size(0)
#
#             z0 = torch.cat((z0, label), dim=-1)  # (B, L+num_label)
#             z0 = self.label_embedding(z0)  # (B, E)
#             dec_x = z0.unsqueeze(0)   # (1, B, E)
#             outputs = []
#
#             for i in range(S-1):
#                 tgt_mask = self.generate_square_subsequent_mask(dec_x.size(0))
#                 dec_span = self.pos_embedding(torch.broadcast_to(span[:i+1], (B, i+1)).unsqueeze(-1)).permute(1, 0, 2)   # (S, B, E)
#                 dec_x = dec_x + dec_span
#                 output = self.model(tgt=dec_x, memory=memory, tgt_mask=tgt_mask)  # (S, B, E)
#                 output = self.output_fc(output)[-1]  # (B, 1)
#                 outputs.append(output)
#                 dec_x = torch.cat((dec_x, self.embedding(output.unsqueeze(0))), dim=0)
#
#         return outputs
#
#
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask.cuda()
#
#



