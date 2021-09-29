import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os
import json
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
#from ecgdetectors import Detectors

def get_dataloader(args, type):
    if args.dataset_type == 'sin':
        data = SinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'sin_onesample':
        data = OneSinDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'NSynth':
        data = NSynthDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'ECG':
        data = ECGDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif args.dataset_type == 'nonlabelECG':
        data = NonlabelECGDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif args.dataset_type == 'GP':
        data = GPDataset(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'atmosphere':
        data = AtmosphericTemperature(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    elif args.dataset_type == 'marketindex':
        data = MarketIndex(args, type)
        dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

    return dataloader



class SinDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.type = type

        # import files
        # if type in ['eval', 'test']:
        dataset = pickle.load(open(os.path.join(args.dataset_path, f'{args.dataset_name}_sin_{type}_data.pk'), 'rb'))
        self.sin = dataset[f'{type}_sin']
        # self.freq = dataset[f'{type}_freq']
        # self.amp = dataset[f'{type}_amp']
        # self.phase = dataset[f'{type}_phase']
        self.orig_ts = dataset['orig_ts']
        self.label = dataset[f'{type}_label']

        # self.index = np.load(os.path.join(args.dataset_path, f'{args.dataset_name}_trainpoints.npy'))

    def __len__(self):
        return self.sin.size(0)

    def __getitem__(self, item):
        index = np.sort(np.random.choice(self.orig_ts.size(0), 500, replace=False))
        return {'sin': self.sin[item],
                'label': self.label[item],
                'orig_ts': self.orig_ts,
                'index': index}


        # else:
        #     sin, amp, freq, phase, orig_ts = self.train_complex2()
        #     return {'sin': sin,
        #             'freq': self.freq[item],
        #             'amp': self.amp[item],
        #             'phase': self.phase[item],
        #             #'label': self.label[item],
        #             'orig_ts': self.orig_ts}

    # def train_complex2(self):
    #     n_comb = 20
    #     start = 0.
    #     end = 3.
    #     n_timestamp = 1000
    #     amp_range = 25
    #     freq_range = 25
    #
    #     orig_ts = np.linspace(start, end, n_timestamp)
    #
    #     amp = [np.around(np.random.uniform(0, amp_range), 1) for i in range(n_comb)]
    #     freq = [np.random.randint(1, freq_range+1) for i in range(n_comb)]
    #     phase = np.around(np.random.uniform(-1, 1), 1)
    #
    #     sinusoidal = amp[0] * np.sin(freq[0] * (orig_ts + phase) * 2 * np.pi)
    #
    #     for j in range(1, int(n_comb / 2)):
    #         sinusoidal += amp[j] * np.sin(freq[j] * (orig_ts + phase) * 2 * np.pi)
    #     for j in range(int(n_comb / 2), n_comb):
    #         sinusoidal += amp[j] * np.cos(freq[j] * (orig_ts + phase) * 2 * np.pi)
    #
    #     sinusoidal += np.random.randn(*sinusoidal.shape) * 0.3
    #     return sinusoidal, np.array(amp), np.array(freq), np.array(phase), torch.Tensor([orig_ts])






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

        self.ECG_type = 'V6'   ## change here!
        # self.index = np.sort(np.load(os.path.join(self.dataset_path, 'sampled_time.npy')))

        # data statistics
        # self.max_val = np.float(32767)
        # self.min_val = np.float(-9138)

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
        # record = (((record - self.min_val) / (self.max_val - self.min_val))) * 100

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
        # elif raw_label[3] == 1:
        #     data_label = torch.LongTensor([3])

        # if raw_label[0] == 1 or raw_label[1] == 1 or raw_label[3] == 1:
        #     data_label = torch.LongTensor([0])
        # elif raw_label[2] == 1 or raw_label[4] == 1:
        #     data_label = torch.LongTensor([1])
        # elif raw_label[5] == 1:
        #     data_label = torch.LongTensor([2])

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
"""
    def sampling(self, record):
        record = torch.FloatTensor(record)
        maxx = torch.max(record)
        minn = torch.min(record)
        hist = torch.histc(record, bins=5)

        def cal_prob(x, hist, maxx, minn):
            length = maxx - minn
            prob = None
            if minn <= x < minn + 0.2 * length:
                prob = 0.2 / hist[0].item()
            elif minn + 0.2 * length <= x < minn + 0.4 * length:
                prob = 0.2 / hist[1].item()
            elif minn + 0.4 * length <= x < minn + 0.6 * length:
                prob = 0.2 / hist[2].item()
            elif minn + 0.6 * length <= x < minn + 0.8 * length:
                prob = 0.2 / hist[3].item()
            elif minn + 0.8 * length <= x <= maxx:
                prob = 0.2 / hist[4].item()
            if prob is None:
                print(self.filename)
            return prob


            # if -10 <= x < -6:
            #     prob = 0.2 / hist[0].item()
            # elif -6 <= x < -2:
            #     prob = 0.2 / hist[1].item()
            # elif -2 <= x < 2:
            #     prob = 0.2 / hist[2].item()
            # elif 2 <= x < 6:
            #     prob = 0.2 / hist[3].item()
            # elif 6 <= x <= 10:
            #     prob = 0.2 / hist[4].item()
            # return prob

        prob = [cal_prob(x.item(), hist, maxx, minn) for x in record]
        index = torch.sort(torch.multinomial(torch.FloatTensor(prob), 100, replacement=False))[0]
        return index

        # return {'sin': torch.FloatTensor(record)[self.index].unsqueeze(-1),
        #         'orig_ts': torch.linspace(0, self.sec, self.freq*self.sec)[self.index],
        #         'label': data_label}
"""

class NonlabelECGDataset(Dataset):
    def __init__(self, args, type):
        super(NonlabelECGDataset, self).__init__()
        assert type in ['train', 'eval', 'test']
        self.dataset_path = args.dataset_path
        self.freq = 500
        self.sec = 1
        self.type = type

        with open(os.path.join(self.dataset_path, f'{args.dataset_name}_{type}_list2.pk'), 'rb') as f:
            self.file_list = pickle.load(f)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        filename = self.file_list[item]
        start = int(filename[-1])
        filename = filename[:-1]

        data = np.load(os.path.join(self.dataset_path, filename), allow_pickle=True)
        record = np.int32(data.item()['val'][11][500*start:500*(start+1)])

        record_max = record.max(); record_min = record.min()
        record = (((record - record_min) / (record_max - record_min)) - 0.5) * 20  # normalize to -10 to 10

        if self.type == 'test':
            index_filename = filename.split('.')[0] + f'_{start}_index.npy'
            index = np.load(os.path.join(self.dataset_path, index_filename))
        else:
            index = self.sampling(record)

        return {'sin': torch.FloatTensor(record)[index].unsqueeze(-1),
                'orig_ts': torch.linspace(0, self.sec, self.sec*self.freq)[index],
                'label': torch.LongTensor([0])}

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
        index = torch.sort(torch.multinomial(torch.FloatTensor(prob), 250, replacement=False))[0]
        return index



"""
class OneSinDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'

        with open(os.path.join(args.dataset_path, f'{args.dataset_name}_sin_{type}_data.pk'), 'rb') as f:
            dataset = pickle.load(f)

        self.sin = dataset[f'{type}_sin'].squeeze(0)   # (1000, 1)
        self.freq = dataset[f'{type}_freq']
        self.amp = dataset[f'{type}_amp']
        self.phase = dataset[f'{type}_phase']
        self.orig_ts = dataset['orig_ts'] # (1000)

    def __len__(self):
        return self.sin.size(0)   # 1000

    def __getitem__(self, item):
        return {'sin': self.sin[item],
                'orig_ts': self.orig_ts[item]}




class GPDataset(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        self.bs = args.batch_size
        self.type = type

        # import files
        # if type in ['eval', 'test']:
        dataset = pickle.load(open(os.path.join(args.dataset_path, f'{args.dataset_name}_{type}_data.pk'), 'rb'))
        self.sin = dataset['y']
        self.orig_ts = dataset['x']

    def __len__(self):
        return self.sin.size(0)

    def __getitem__(self, item):
        return {'sin': self.sin[item],
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

class ECGOnesample(Dataset):
    def __init__(self, args, type):
        super(ECGOnesample, self).__init__()
        self.dataset_path = args.dataset_path
        self.freq = 500
        self.sec = 3

        file_list = pickle.load(open(os.path.join(args.dataset_path, os.pardir, 'normal_ECGlist.pk'), 'rb'))
        file = file_list[2]

        data = pickle.load(open(os.path.join(self.dataset_path, file), 'rb'))
        self.record = np.int32(data['val'][11][2500:2500+int(self.freq*self.sec)])

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return {'sin': torch.FloatTensor(self.record).unsqueeze(-1),
                'orig_ts': torch.linspace(0, self.sec, self.freq*self.sec),
                'label': }



class AtmosphericTemperature(Dataset):
    def __init__(self, args, type):
        super().__init__()
        assert type in ['train', 'eval', 'test'], 'type should be train or eval or test'
        N_sample = 365*6
        N_test = 3287

        x_resample = 7

        N_test = int(np.floor(N_test/x_resample))
        N_sample = int(np.floor(N_sample/x_resample))

        DayTemp = pd.read_csv('/home/edlab/jylee/generativeODE/NeurIPS_2020_Snake/data/day_temp_atmospheric.csv')
        MyTemp = DayTemp.values

        TempTemp = np.zeros(N_test)
        TempDay = np.zeros(N_test)

        for i in range(N_test):
            TempTemp[i] = np.mean(MyTemp[i * x_resample:(i + 1) * x_resample, 1])
            TempDay[i] = np.mean(MyTemp[i * x_resample:(i + 1) * x_resample, 0])

        MeanTemp = np.mean(TempTemp)
        MyTemp1 = TempTemp/MeanTemp
        MyTemp0 = (TempDay/float(365))

        self.X = torch.tensor(MyTemp0[0:N_sample]).float().unsqueeze(1)   # (312, 1)
        self.Y = torch.tensor(MyTemp1[0:N_sample]).float().unsqueeze(1)   # (312, 1)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, item):
        return {'sin': self.Y[item],
                'orig_ts': self.X[item]}


class MarketIndex(Dataset):
    def __init__(self, args, type):
        super().__init__()
        file1 = open('/home/edlab/jylee/generativeODE/NeurIPS_2020_Snake/data/market.txt', 'r')
        lines = file1.readlines()

        count = 0
        acc_list = []
        for line in lines:
            if 'N' not in line:
                try:
                    acc_list.append(float(line))
                except:
                    count += 1
            else:
                count += 1

        acc_list = np.array(acc_list)

        X = torch.Tensor(range(1, len(acc_list)+1)).float()
        C = torch.max(X)
        self.X = (X / torch.max(X))[:6300]

        Y = torch.Tensor(acc_list).unsqueeze(-1)
        self.Y = (Y / torch.max(Y))[:6300, :]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return {'sin': self.Y,
                'orig_ts': self.X}








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
"""