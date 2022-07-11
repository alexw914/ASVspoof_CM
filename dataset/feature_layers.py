import torch
import torchaudio
import librosa
import numpy as np
import torch.nn as nn
from dataset.data_utils import *

from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
)

"""
The code is Overwrite on speechbrain.processing.features.MFCC
"""


class MFCC(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: True)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: True)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 16000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 23)
        Number of filters to use for creating filterbank.
    n_mfcc : int (default: 20)
        Number of output coefficients
    filter_shape : str (default 'triangular')
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor: bool (default 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor: float (default 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default 5)
        Number of frames of left context to add.
    right_frames : int (default 5)
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 660])
    """

    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=23,
        n_mfcc=20,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
        compute_layer=2,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_dct = DCT(input_size=n_mels, n_out=n_mfcc)
        self.compute_deltas = Deltas(input_size=n_mfcc)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )
        self.compute_layer = compute_layer

    def forward(self, wav):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        with torch.no_grad():
            STFT = self.compute_STFT(wav)
            mag_log, fbanks, mfccs = None, None, None

            mag_log = spectral_magnitude(STFT, log=True)
            if self.compute_layer>=2:
                mag = spectral_magnitude(STFT)
                fbanks = self.compute_fbanks(mag)
                if self.deltas:
                    delta1 = self.compute_deltas(fbanks)
                    delta2 = self.compute_deltas(delta1)
                    fbanks = torch.cat([fbanks, delta1, delta2], dim=2)
                if self.context:
                    fbanks = self.context_window(fbanks)
            if self.compute_layer == 3:
                mfccs = self.compute_dct(fbanks)
                if self.deltas:
                    delta1 = self.compute_deltas(mfccs)
                    delta2 = self.compute_deltas(delta1)
                    mfccs = torch.cat([mfccs, delta1, delta2], dim=2)

                if self.context:
                    mfccs = self.context_window(mfccs)


            return mag_log, fbanks.transpose(1,2), mfccs


class CQT(nn.Module):
    def __init__(self, sampling_rate):
        super(CQT, self).__init__()
        self.sampling_rate = sampling_rate

    def forward(self, x, device):
        batch_size = x.shape[0]
        batch_output = torch.zeros(batch_size, 84, 126)
        batch_count = 0
        for item in x:
            numpy_item = item.numpy()
            item_cqt = librosa.cqt(numpy_item, sr=self.sampling_rate)
            item_cqt = librosa.amplitude_to_db(np.abs(item_cqt), ref=np.max)
            item_torch_cqt = torch.from_numpy(item_cqt).to(device)
            batch_output[batch_count] = item_torch_cqt
            batch_count += 1

        return batch_output.to(device)


class Spectrogram(nn.Module):
    def __init__(self, n_fft):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft

    def forward(self, x, device):
        batch_size = x.shape[0]
        batch_output = torch.zeros(batch_size, 1025, 126)
        batch_count = 0
        for item in x:
            numpy_item = item.numpy()
            item_stft = librosa.stft(numpy_item, n_fft=self.n_fft)
            item_stft = librosa.amplitude_to_db(np.abs(item_stft), ref=np.max)
            item_torch_stft = torch.from_numpy(item_stft).to(device)
            batch_output[batch_count] = item_torch_stft
            batch_count += 1

        return batch_output.to(device)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


class LFCC(nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """

    def __init__(self, fl, fs, fn, sr, filter_num,
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_LFB=False):
        """ Initialize LFCC

        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank
          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta

          for_LFB: bool (default False), reserved for LFB feature
        """
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num

        # build the triangle filter bank
        f = (sr / 2) * torch.linspace(0, 1, fn // 2 + 1)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([fn // 2 + 1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx],
                    filter_bands[idx + 1],
                    filter_bands[idx + 2]])
        self.lfcc_fb = nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')

        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_LFB = flag_for_LFB
        return

    def forward(self, x):
        """

        input:
        ------
         x: tensor(batch, length), where length is waveform length

        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]

        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl,
                            window=torch.hamming_window(self.fl).to(x.device),
                            onesided=True, pad_mode="constant")
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) +
                                 torch.finfo(torch.float32).eps)

        # DCT (if necessary, remove DCT)
        lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature

        # Add energy
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2) +
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc
        
        # lfcc = lfcc.squeeze(0)
        lfcc = lfcc.transpose(-1,-2)

        # done
        return lfcc_output



if __name__=="__main__":
    file_name = "/home/alex/桌面/SASV/asv/VoxCeleb/voxceleb1/vox1_train/id10001/1zcIwhmdeo4/00002.wav"
    file,len = torchaudio.load(file_name)
    file = torch.randn((64,64000))
    trans = LFCC(320, 160, 512, 16000, 20, with_energy=False)
    lfccs = trans(file)
    print(lfccs.shape)
    transcqt = CQT(16000)
    cqt = transcqt(file,"cuda")
    print(cqt.shape)