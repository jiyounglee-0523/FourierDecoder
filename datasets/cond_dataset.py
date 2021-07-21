import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os
import json
import numpy as np
from scipy.io import wavfile
import librosa
from ecgdetectors import Detectors

def get_dataloader(args, type):
    if args.dataset_type == 'sin':
        data = SinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif args.dataset_type == 'NSynth':
        data = NSynthDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif args.dataset_type == 'ECG':
        data = ECGDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    return dataloader



class SinDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'

        # import files
        dataset = pickle.load(open(os.path.join(args.dataset_path, f'conf_sin_{type}_data.pk'), 'rb'))
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
        names = [name for name in names if '_synthetic_' in name]
        names = [name for name in names if '-040-' in name]   # pitch=40

        ## filter selected instruments
        names = [name for name in names if name.split('_')[0] in ['flute', 'keyboard', 'vocal']]

        ## extract fast-fade string
        my_names = []
        for name in names:
            if 'fast_decay' not in metadata[name]['qualities_str']:
                my_names.append(name)

        print(f'{type} dataset has total of {len(my_names)}')
        self.my_names = my_names

        # self.instrument = {'brass': 0,
        #                    'flute': 1,
        #                    'keyboard': 2,
        #                    'organ': 3,
        #                    'reed': 4,
        #                    'string': 5,
        #                    'vocal': 6}
        self.instrument = {'flute': 0,
                           'keyboard': 1,
                           'vocal': 2}

    def __len__(self):
        return len(self.my_names)

    def __getitem__(self, item):
        name = self.my_names[item]
        #_, wav = wavfile.read(os.path.join(self.dataset_path, f'nsynth-{self.type}', 'audio', name+'.wav'), )
        wav, _ = librosa.load(os.path.join(self.dataset_path, f'nsynth-{self.type}', 'audio', name+'.wav'), sr=None)
        wav = torch.FloatTensor(wav)[16000:16000+1600]     # only use 1~1.1sec
        wav = wav.unsqueeze(-1)
        orig_ts = torch.linspace(0, 0.1, 1600)
        label = torch.tensor([self.instrument[name.split('_')[0]]])
        freq = amp = torch.zeros(1)

        return {'sin': wav,
                'freq': name,
                'amp': amp,
                'label': label,
                'orig_ts': orig_ts}



class ECGDataset(Dataset):
    def __init__(self, args, type):
        super(ECGDataset, self).__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.dataset_path = args.dataset_path
        self.freq = 500
        self.sec = 1
        self.detector = Detectors(self.freq)

        # file_list = list(np.load(os.path.join(args.dataset_path, os.pardir,f'{type}_filelist.npy')))
        file_list = os.listdir(args.dataset_path)
        r_peak_len = {}
        for file in file_list:
            data = pickle.load(open(os.path.join(args.dataset_path, file), 'rb'))
            V6 = data['val'][11][2500:]

            try:
                r_peaks = self.detector.christov_detector(V6)
                r_peak_len[file] = len(r_peaks)
            except:
                print('No!')

        names = []
        for name, value in r_peak_len.items():
            if value in [0, 1, 2]:
                names.append(name)

        self.file_list = list(set(file_list) - set(names))
        # if type == 'train':
        #     idx = np.load('/home/jylee/data/generativeODE/input/train_idx.npy')
        # elif type == 'eval':
        #     idx = np.load('/home/jylee/data/generativeODE/input/test_idx.npy')
        #
        # file_list = list(np.array(file_list)[idx]

        self.ECG_type = 'V6'   ## change here!

        # data statistics
        self.max_val = np.float(32751)
        self.min_val = np.float(-8199)

        """
        label 0 : RBBB
        label 1 : LBBB
        label 2 : LVH 
        """

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        data = pickle.load(open(os.path.join(self.dataset_path, self.file_list[item]), 'rb'))
        if self.ECG_type == 'V1':
            # record = np.int32(data['val'][:, :int(self.freq * self.sec)][6])
            record = np.int32(data['val'][6][2500:])
        elif self.ECG_type == 'V6':
            # record = np.int32(data['val'][:, :int(self.freq * self.sec)][11])
            record = np.int32(data['val'][11][2500:])
        raw_label = data['label']
        data_label = None
        # detect r_peak
        r_peaks = self.detector.christov_detector(record)
        try:
            record = np.array([2]) * ((record - self.min_val) / (self.max_val - self.min_val)) + np.array([-1])
            record = record + np.array([0.6])
            record = record[r_peaks[0]: int(r_peaks[0] + (self.freq*self.sec))]
        except:
            print(self.file_list[item])

        # assert (np.min(record) >= -1) and (np.max(record) <= 1), 'check normalization'

        if raw_label[0] == 1 or raw_label[1] == 1 or raw_label[3] == 1:
            data_label = torch.LongTensor([0])
        elif raw_label[2] == 1 or raw_label[4] == 1:
            data_label = torch.LongTensor([1])
        elif raw_label[5] == 1:
            data_label = torch.LongTensor([2])

        freq = amp = torch.zeros(1)

        return {'sin': torch.FloatTensor(record).unsqueeze(-1),
                'label': data_label,
                'freq': self.file_list[item],
                'amp': amp,
                'orig_ts': torch.linspace(0, self.sec, self.freq*self.sec)}





