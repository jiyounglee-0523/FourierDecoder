import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os
import numpy as np


def get_dataloader(args, type):
    if args.dataset_type == 'sin':
        data = SinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'ECG':
        data = ECGDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    return dataloader



class SinDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.type = type


        dataset = pickle.load(open(os.path.join(args.dataset_path, f'{args.dataset_name}_sin_{type}_data.pk'), 'rb'))
        self.sin = dataset[f'{type}_sin']
        self.orig_ts = dataset['orig_ts']
        self.label = dataset[f'{type}_label']


    def __len__(self):
        return self.sin.size(0)

    def __getitem__(self, item):
        index = np.sort(np.random.choice(self.orig_ts.size(0), 500, replace=False))
        return {'sin': self.sin[item],
                'label': self.label[item],
                'orig_ts': self.orig_ts,
                'index': index}


class ECGDataset(Dataset):
    def __init__(self, args, type):
        super(ECGDataset, self).__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.dataset_path = args.dataset_path
        self.freq = 500
        self.sec = 1
        self.type = type

        with open(os.path.join(self.dataset_path, f'{args.dataset_name}_{type}_ECGlist2.pk'), 'rb') as f:
            self.file_list = pickle.load(f)

        self.ECG_type = 'V6'

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
        self.filename = filename
        start = int(filename[-1])
        filename = filename[:-1]
        with open(os.path.join(self.dataset_path, filename), 'rb') as f:
            data = pickle.load(f)
        record = np.int32(data['val'][11][500*start:500*(start+1)])

        record_max = record.max() ; record_min = record.min()
        record = (((record - record_min) / (record_max - record_min)) - 0.5)*20    # normalize to -10 to 10

        if self.type == 'test':
            index_filename = filename.split('.')[0] + f'_{start}_index_100.npy'
        else:
            index = self.sampling(record)

        # label
        raw_label = data['label']
        data_label = None

        if raw_label[0] == 1:
            data_label = torch.LongTensor([0])
        elif raw_label[1] == 1:
            data_label = torch.LongTensor([1])
        elif raw_label[3] == 1:
            data_label = torch.LongTensor([2])
        else:
            raise NotImplementedError

        return {'sin': torch.FloatTensor(record).unsqueeze(-1),
                'orig_ts': torch.linspace(0, self.sec, self.sec*self.freq),
                'label': data_label,
                'index': index}

    def sampling(self, record):
        record = torch.FloatTensor(record)
        hist = torch.histc(record, bins=5)

        def cal_prob(x, hist):
            if -10 <= x < -6:
                prob = 0.2 / hist[0].item()
            elif -6 <= x < -2:
                prob = 0.2 / hist[1].item()
            elif -2 <= x < 2:
                prob = 0.2 / hist[2].item()
            elif 2 <= x < 6:
                prob = 0.2 / hist[3].item()
            elif 6 <= x <= 10:
                prob = 0.2 / hist[4].item()
            return prob

        prob = [cal_prob(x.item(), hist) for x in record]
        index = torch.sort(torch.multinomial(torch.FloatTensor(prob), 500, replacement=False))[0]
        return index