import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import numpy.random as npr

def get_dataloader(args):
    train_data = Sinusoid_from_scratch(dataset_type=args.dataset_type)
    data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    return data_loader

def dataset1(n_sinusoid=1024, n_total=2000, n_sample=400, skip_step=4):
    """
    √3cos(𝑥−0.615) =  √2cos(𝑥)+sin(𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    sinusoidals = []
    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (np.sqrt(3) * np.cos(orig_ts - 0.615))
        sinusoidals.append(sinusoidal)

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    sinusoidals = np.stack(sinusoidals, axis=0)
    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    latent = torch.ones(n_sinusoid, 2)
    amps = torch.cat((amps, latent), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * 2000)

    return samp_sinusoidals, samp_ts, amps

def dataset1_1dim(n_sinusoid=1024, n_total=2000, n_sample=400, skip_step=4):
    """
    √3cos(𝑥−0.615) =  √2cos(𝑥)+sin(𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (np.sqrt(3) * np.cos(orig_ts - 0.615))

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * 2000)

    return samp_sinusoidals, samp_ts, amps



def dataset2(n_sinusoid=2000, n_total=2000, n_sample=400, skip_step=4):
    """
    𝑦(𝑥)=sin(2𝑥)−cos(𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * ((1/0.61) * np.sin((0.61)* orig_ts) - (1/0.07)* np.cos((0.07) * orig_ts))

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)
    latent = torch.ones(n_sinusoid, 2)
    amps = torch.cat((amps, latent), dim=-1)

    return samp_sinusoidals, samp_ts, amps



def dataset3(n_sinusoid=2000, n_total=2000, n_sample=400, skip_step=4):
    """
    𝑦(𝑥)=−4 sin(𝑥)+sin(2𝑥)−cos(𝑥)+0.5 cos(2𝑥)
    samp_sinusoidals.shape = 2000 x 400 x 1
    samp_ts.shape = 2000 x 400
    amps.shape = 2000 x 1
    """
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * (-4 * np.sin(orig_ts) + np.sin(2 * orig_ts) - np.cos(orig_ts) + 0.5 * np.cos(2 * orig_ts))

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    latent = torch.ones(n_sinusoid, 2)
    amps = torch.cat((amps, latent), dim=-1)

    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)

    return samp_sinusoidals, samp_ts, amps

def dataset4(n_sinusoid=1024, n_total=2000, n_sample=300, skip_step=4):
    "different amp for each element"
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    count = int(n_sinusoid / 4)
    amp1 = [1] * count + [2] * count + [3] * count + [4] * count
    count = int(n_sinusoid / 8)
    amp2 = ([1] * count + [2] * count + [3] * count + [4] * count) * 2
    count = int(n_sinusoid / 16)
    amp3 = ([1] * count + [2] * count + [3] * count + [4] * count) * 4
    count = int(n_sinusoid / 32)
    amp4 = ([1] * count + [2] * count + [3] * count + [4] * count) * 8

    amps = np.stack((amp1, amp2, amp3, amp4), axis=1)
    assert len(amps) == n_sinusoid, "amp list incorrectly constructed!"

    for i in range(n_sinusoid):
        amp = amps[i]
        samp_sinusoidal = -amp[0] * np.cos(samp_ts) - amp[1] * 0.5 * np.cos(2 * samp_ts) + amp[2] * np.sin(samp_ts) + \
                          amp[3] * 0.5 * np.sin(2 * samp_ts)
        samp_sinusoidals.append(samp_sinusoidal)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)  # batch_size x seq_len x 1
    amps = torch.Tensor(amps)  # batch_size x 4
    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)  # n_sinusoid x 400
    return samp_sinusoidals, samp_ts, amps


def dataset5(n_sinusoid=2000, n_total = 2000, n_sample=400, skip_step=4):
    start = 0.
    stop = 6. * np.pi
    amp_range = 4
    freq_range = 1.5

    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    amps = []
    freqs = []

    for i in range(n_sinusoid):
        amp = np.around(npr.uniform(1, amp_range), 1)
        freq = np.around(npr.uniform(0.9, freq_range), 1)

        amps.append(amp)
        freqs.append(freq)

        sinusoidal = amp * np.sin(orig_ts * freq)
        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step]
        samp_sinusoidals.append(samp_sinusoidal)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = torch.unsqueeze(torch.Tensor(np.stack(amps, axis=0)), dim=-1)
    freqs = torch.unsqueeze(torch.Tensor(np.stack(freqs, axis=0)), dim=-1)
    latent_v = torch.cat((amps, freqs), dim=-1)
    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)
    return samp_sinusoidals, samp_ts, latent_v


def dataset6(n_sinusoid=2000, n_total=2000, n_sample=400, skip_step=4):
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    amps = []

    for i in range(0, n_sinusoid):
        if i < 500:
            amp = 1
        elif i < 1000:
            amp = 2
        elif i < 1500:
            amp = 3
        else:
            amp = 4

        sinusoidal = amp * np.sin(1.7 * orig_ts)

        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)

        amps.append(amp)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    amps = np.stack(amps, axis=0)
    amps = torch.unsqueeze(torch.Tensor(amps), dim=-1)
    latent = torch.ones(n_sinusoid, 2)
    amps = torch.cat((amps, latent), dim=-1)

    samp_ts = torch.Tensor([samp_ts] * n_sinusoid)

    return samp_sinusoidals, samp_ts, amps


def dataset7(n_sinusoidal=2000, n_total=2000, n_sample=400, skip_step=4):
    """n_harmonics=2, n_eig=2"""
    start = 0.
    stop = 6. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    dilations = []

    for i in range(n_sinusoidal):
        if i < 200:
            dil1 = 1
            dil2 = 1
        elif i < 400:
            dil1 = 1
            dil2 = 2
        elif i < 600:
            dil1 = 1
            dil2 = 3
        elif i < 800:
            dil1 = 1
            dil2 = 4
        elif i < 1000:
            dil1 = 2
            dil2 = 2
        elif i < 1200:
            dil1 = 2
            dil2 = 3
        elif i < 1400:
            dil1 = 2
            dil2 = 4
        elif i < 1600:
            dil1 = 3
            dil2 = 3
        elif i < 1800:
            dil1 = 3
            dil2 = 4
        else:
            dil1 = 4
            dil2 = 4

        dil = np.stack((dil1, dil2))

        sinusoidal = 1/dil1 * np.sin(dil1 * orig_ts) + 1/dil2 * np.sin(dil2 * orig_ts)
        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)
        dilations.append(dil)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    dilations = np.stack(dilations, axis=0)
    dilations = torch.Tensor(dilations)

    samp_ts = torch.Tensor([samp_ts] * n_sinusoidal)

    return samp_sinusoidals, samp_ts, dilations

def dataset8(n_sinusoidal=2048, n_total=3000, n_sample=400, skip_step=6):
    """n_harmonics=2, n_eig=2"""
    start = 0.
    stop = 20. * np.pi
    orig_ts = np.linspace(start, stop, num=n_total)
    samp_ts = orig_ts[0: (n_sample * skip_step): skip_step]

    samp_sinusoidals = []
    dilations = []
    amps = []

    for i in range(n_sinusoidal):
        dil1 = np.around(npr.uniform(0.9, 5), 1)
        dil2 = np.around(npr.uniform(0.9, 5), 1)
        dil = np.stack((dil1, dil2))
        amp1 = np.around(npr.uniform(1., 4.))
        amp2 = np.around(npr.uniform(1., 4.))
        amp = np.stack((amp1, amp2))

        # originally it was np.sin(dil1 * orig_ts) + np.sin(dil2 * orig_ts)
        sinusoidal = amp1 * np.sin(dil1 * orig_ts) + amp2 * np.cos(dil2 * orig_ts)
        samp_sinusoidal = sinusoidal[0: (n_sample * skip_step): skip_step].copy()
        samp_sinusoidals.append(samp_sinusoidal)
        dilations.append(dil)
        amps.append(amp)

    samp_sinusoidals = np.stack(samp_sinusoidals, axis=0)
    samp_sinusoidals = torch.unsqueeze(torch.Tensor(samp_sinusoidals), dim=-1)
    dilations = torch.Tensor(np.stack(dilations, axis=0))
    amps = torch.Tensor(np.stack(amps, axis=0))
    latent_v = torch.cat((dilations, amps), axis=1)

    samp_ts = torch.Tensor([samp_ts] * n_sinusoidal)

    return samp_sinusoidals, samp_ts, latent_v


class Sinusoid_from_scratch(Dataset):
    def __init__(self, dataset_type):
        super().__init__()
        if dataset_type == 'dataset1':
            dataset_type = dataset1
        elif dataset_type == 'dataset5':
            dataset_type = dataset5
        elif dataset_type == 'dataset2':
            dataset_type = dataset2
        elif dataset_type == 'dataset3':
            dataset_type = dataset3
        elif dataset_type == 'dataset1_1dim':
            dataset_type = dataset1_1dim
        elif dataset_type == 'dataset6':
            dataset_type = dataset6
        elif dataset_type == 'dataset7':
            dataset_type = dataset7
        elif dataset_type == 'dataset8':
            dataset_type = dataset8
        self.samp_sin, self.samp_ts, self.latent_v = dataset_type()

    def __len__(self):
        return self.samp_sin.size(0)

    def __getitem__(self, item):
        samp = self.samp_sin[item]
        samp_ts = self.samp_ts[item]
        latent_v = self.latent_v[item]
        return samp, samp_ts, latent_v
