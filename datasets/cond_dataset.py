import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os
import json
from scipy.io import wavfile

def get_dataloader(args, type):
    if args.dataset_type == 'sin':
        data = SinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'NSynth':
        data = NSynthDataset(args, type)
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

class NSynthDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.dataset_path = args.dataset_path
        type = 'valid' if type == 'eval' else type
        self.type = type

        # import file_list
        #name_list = os.listdir(os.path.join(args.dataset_path, f'nsynth-{type}', 'audio'))
        metadata = json.load(open(os.path.join(args.dataset_path, f'nsynth-{type}', 'examples.json'), 'rb'))
        names = sorted(metadata.keys())

        #### filter dataset
        ## filter acoustic
        names = [name for name in names if '_acoustic_' in name]

        ## filter selected instruments
        names = [name for name in names if name.split('_')[0] not in ['bass', 'guitar', 'mallet']]

        ## extract fast-fade string
        my_names = []
        for name in names:
            if 'fast_decay' not in metadata[name]['qualities_str']:
                my_names.append(name)

        print(f'{type} dataset has total of {len(my_names)}')
        self.my_names = my_names

        self.instrument = {'brass': 0,
                           'flute': 1,
                           'keyboard': 2,
                           'organ': 3,
                           'reed': 4,
                           'string': 5,
                           'vocal': 6}

    def __len__(self):
        return len(self.my_names)

    def __getitem__(self, item):
        name = self.my_names[item]
        _, wav = wavfile.read(os.path.join(self.dataset_path, f'nsynth-{self.type}', 'audio', name+'.wav'), )
        wav = torch.FloatTensor(wav)[:16000]     # only use 0~1sec
        wav = wav.unsqueeze(-1)
        orig_ts = torch.linspace(0, 1, 16000)
        label = torch.tensor([self.instrument[name.split('_')[0]]])
        freq = amp = torch.zeros(1)

        return {'sin': wav,
                'freq': freq,
                'amp': amp,
                'label': label,
                'orig_ts': orig_ts}



