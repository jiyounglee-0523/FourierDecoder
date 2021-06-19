import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os

def get_dataloader(args, type):
    if args.dataset_type == 'sin':
        data = SinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    return dataloader



class SinDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'

        # import files
        dataset = pickle.load(open(os.path.join(args.dataset_path, f'sin_{type}_data.pk'), 'rb'))
        self.sin = dataset[f'{type}_sin']
        self.freq = dataset[f'{type}_freq']
        self.amp = dataset[f'{type}_amp']
        self.orig_ts = dataset['orig_ts']
        self.label = dataset[f'{type}_label']

    def __len__(self):
        return self.sin.size(0)

    def __getitem__(self, item):
        return {'sin': self.sin[item],
                'freq': self.freq[item],
                'amp': self.amp[item],
                'label': self.label[item],
                'orig_ts': self.orig_ts}


