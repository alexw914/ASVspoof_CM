import math, torch, torchaudio, librosa, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import random, sample
from scipy import signal
import speechbrain as sb

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

def load_wav(audio_file, num_frames=398):
    audio, sr = librosa.load(audio_file, sr=16000)
    length = num_frames * 160 + 160
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')
    start_frame = np.int64(random()*(audio.shape[0]-length))
    audio = audio[start_frame:start_frame + length]
    return torch.FloatTensor(audio)

def load_spec_wav(audio_file, num_frames=498):
    audio, sr = librosa.load(audio_file, sr=16000)
    length = num_frames * 160 + 240
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')
    audio = signal.lfilter([1, -0.97], [1], audio)
    start_frame = np.int64(random()*(audio.shape[0]-length))
    audio = audio[start_frame:start_frame + length]
    return torch.FloatTensor(audio)


def load_pt(feat_path, num_frames=400):
    data_x = torch.load(feat_path)
    if data_x.shape[1] > num_frames:    
        start_frame = np.int64(random() * (data_x.shape[1]-num_frames))
        data_x = data_x[:, start_frame: start_frame+num_frames]
    if data_x.shape[1] < num_frames:
        data_x = repeat_padding_Tensor(data_x, num_frames)
    return data_x


class PreEmphasis(nn.Module):
    
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class LADFAug(nn.Module):
    def __init__(self, feature="lfcc", aug_folder="ASVspoof2019/LA/LADF_AUG", aug_type="codec", aug_times=1):
        super().__init__()
        self.aug_path = aug_folder
        self.feature = feature
        self.channel = ['amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]','silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]']
        self.compression = ['aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]']
        self.aug_times = aug_times
        self.aug_type = aug_type
    def forward(self, wav_name):
        with torch.no_grad():
            wavs = []
            lens = len(wav_name)
            for i in range(self.aug_times):
                if self.aug_type == "codec":
                    aug_type = sample(self.channel, 1)
                    file_root = os.path.join(self.aug_path, "codec", aug_type[0])
                    for i in range(lens):
                        file_path = os.path.join(file_root, wav_name[i]+".wav")
                        wavs.append(load_wav(file_path).unsqueeze(0))
                if self.aug_type == "compression":
                    aug_type = sample(self.compression, 1)
                    file_root = os.path.join(self.aug_path, "compression", aug_type[0])
                    for i in range(lens):
                        file_path = os.path.join(file_root, wav_name[i]+".wav")
                        wavs.append(load_wav(file_path).unsqueeze(0))
            wavs = torch.cat(wavs, dim=0)
        return wavs
            


