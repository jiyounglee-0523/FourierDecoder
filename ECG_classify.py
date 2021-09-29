import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import wandb
import argparse
import random
import pickle
from sklearn.metrics import confusion_matrix

from utils.trainer_utils import EarlyStopping

class ECGDataset(Dataset):
    def __init__(self, args, type):
        super(ECGDataset, self).__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.dataset_path = args.dataset_path
        self.freq = 500
        self.sec = 1
        self.model = args.model
        self.label = args.label

        with open(os.path.join(self.dataset_path, 'AF_ECG', f'newnew_{type}_ECGlist2.pk'), 'rb') as f:
            self.file_list = pickle.load(f)

        # if type == 'train':
        #     aug_file_list = os.listdir(os.path.join(self.dataset_path, 'gen_ECG_dataset', args.model, args.label))
        #     aug_file_list = aug_file_list[:args.num_sample]
        #     self.file_list.extend(aug_file_list)
        #     print('gen dataset added!')

        self.ECG_type = 'V6'   ## change here!

        """
        label 0 : RBBB
        label 1 : LBBB
        label 2 : LVH 
        label 3: AF
        """

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        filename = self.file_list[item]
        try:
            start = int(filename[-1])
            filename = filename[:-1]
            with open(os.path.join(self.dataset_path, 'AF_ECG', filename), 'rb') as f:
                data = pickle.load(f)
            record = np.int32(data['val'][11][500*start:500*(start+1)])

            record_max = record.max() ; record_min = record.min()
            record = (((record - record_min) / (record_max - record_min)) - 0.5)*20    # normalize to -10 to 10

        except:
            raise OSError
            with open(os.path.join(self.dataset_path, 'gen_ECG_dataset', self.model, self.label, filename), 'rb') as f:
                data = pickle.load(f)

            record = data['data']

        # label
        raw_label = data['label']


        data_label = None
        if raw_label[0] == 1:
            data_label = torch.LongTensor([0])
        elif raw_label[1] == 1:
            data_label = torch.LongTensor([1])
        # elif raw_label[2] == 1:
        #     data_label = torch.LongTensor([2])
        elif raw_label[3] == 1:
            data_label = torch.LongTensor([2])

        return {'sin': torch.FloatTensor(record).unsqueeze(-1),
                'orig_ts': torch.linspace(0, self.sec, self.sec*self.freq),
                'label': data_label}


class ConvClassify(nn.Module):
    def __init__(self, args):
        super(ConvClassify, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1))
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(64, args.num_label)  # check the dimension

    def forward(self, x):
        # x shape of (B, S, 1),
        output = self.conv(x.permute(0, 2, 1))
        output = self.glob_pool(output).squeeze(-1)
        output = self.output_fc(output)
        return output

class FCClassify(nn.Module):
    def __init__(self, args):
        super(FCClassify, self).__init__()
        self.fc = nn.Sequential(nn.Linear(500, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, args.num_label))

    def forward(self, x):
        return self.fc(x.squeeze(-1))


class TransformerClassify(nn.Module):
    def __init__(self, args):
        super(TransformerClassify, self).__init__()
        self.dropout = 0.1
        self.embedding = nn.Linear(1, 128, bias=False)
        self.pos_embedding = nn.Linear(1, 128)
        encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, dropout=self.dropout)
        self.model = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.output_fc = nn.Linear(128, args.num_label)

    def forward(self, x):
        # x (B, S)
        x = self.embedding(x)
        t = torch.linspace(0, 499, 500).unsqueeze(-1).cuda()
        t = self.pos_embedding(t).unsqueeze(0)  # (1, 500, 128)
        x = x + t
        output = self.model(x.permute(1, 0, 2))   # (500, B, 128)
        output = output.mean(0)  # (B, 128)
        return self.output_fc(output)


class ECGTrainer():
    def __init__(self, args):
        self.train_dataloader = DataLoader(dataset=ECGDataset(args, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.eval_dataloader = DataLoader(dataset=ECGDataset(args, 'eval'), batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.test_dataloader = DataLoader(dataset=ECGDataset(args, 'test'), batch_size=3615, shuffle=True, num_workers=16)
        self.n_epochs = args.n_epochs
        self.debug = args.debug
        self.early_stopping = EarlyStopping(patience=20, verbose=True)

        self.dataset_type = args.dataset_type
        self.path = f'{args.path}newnewConvclassify_5678'
        self.filepath = self.path +'/'+ f'newnewConvclassify_5678'
        args.filename = self.filepath
        print(f'Model will be saved at {self.path}')
        if not self.debug:
            os.mkdir(self.path)

        self.model = ConvClassify(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if not args.debug:
            wandb.init(project='NIPS_workshop', config=args)

    def train(self):
        best_acc = 0.0
        for n_epoch in range(self.n_epochs):
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()

                output = self.model(samp_sin)  # (B, C)
                loss = nn.CrossEntropyLoss()(output, label.squeeze(-1))
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss})

            eval_loss, acc = self.evaluation()
            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_acc': acc})

            if best_acc < acc:
                best_acc = acc
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_acc}, self.filepath + '_best.pt')
                    print(f'Model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_acc}, self.filepath + f'_{n_epoch}.pt')
                    print(f'Model parameter saved at {n_epoch}')

            self.early_stopping(acc)
            if self.early_stopping.early_stop is True:
                self.test()
                break


    def evaluation(self):
        self.model.eval()
        avg_loss = 0.
        correct_num = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()

                output = self.model(samp_sin)
                loss = nn.CrossEntropyLoss()(output, label.squeeze(-1))
                avg_loss += (loss.item() * samp_sin.size(0))

                # calculate accuracy
                predict = output.argmax(1)
                correct_num += (predict == label.squeeze(-1)).sum().item()

            acc = correct_num / self.eval_dataloader.dataset.__len__()
            avg_loss = avg_loss / self.eval_dataloader.dataset.__len__()

        return avg_loss, acc


    def test(self):
        self.model.eval()
        ckpt = torch.load(self.filepath + '_best.pt')
        self.model.load_state_dict(ckpt['model_state_dict'])
        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()

                predict = self.model(samp_sin)
                predict = predict.argmax(1)

                acc = ((predict == label.squeeze(-1)).sum() / self.test_dataloader.dataset.__len__())

                cm = confusion_matrix(label.detach().cpu().numpy(), predict.detach().cpu().numpy())
                recall = np.diag(cm) / np.sum(cm, axis=1)
                precision = np.diag(cm) / np.sum(cm, axis=0)
                f1 = (2*precision*recall) / (precision + recall)

                wandb.log({
                    'acc': acc,
                    'RBBB_recall': recall[0],
                    'LBB_recall': recall[1],
                    'AF_recall': recall[2],
                    # 'AF_recall': recall[3],
                    'RBBB_precision': precision[0],
                    'LBBB_precision': precision[1],
                    'AF_precision': precision[2],
                    # 'AF_precision': precision[3],
                    'RBBB_F1': f1[0],
                    'LBBB_F1': f1[1],
                    'AF_F1': f1[2],
                    # 'AF_F1': f1[3],
                    'recall': recall.mean(),
                    'precision': precision.mean(),
                    'f1': f1.mean()
                })



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--path', type=str, default='/home/edlab/jylee/generativeODE/output/NIPS_workshop/', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str, default='/home/edlab/jylee/generativeODE/input/')
    # parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth'], default='ECG')
    parser.add_argument('--ECG_type', choices=['V6'], default='V6')
    parser.add_argument('--notes', type=str, default='example')

    parser.add_argument('--model', choices=['Fourier', 'NP', 'ODE', 'RNN', 'Transformer'])
    parser.add_argument('--label', choices=['RBBB', 'LVH', 'AF'])
    parser.add_argument('--num_sample', type=int, choices=[500, 1000, 1500, 2000, 2500])

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device_num', type=str, default='0')
    args = parser.parse_args()

    if args.dataset_type == 'ECG':
        args.num_label = 3

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    SEED = 5678
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = ECGTrainer(args)
    trainer.train()



if __name__ == '__main__':
    main()
