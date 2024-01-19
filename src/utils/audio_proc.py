import torch
from torch import nn
import numpy as np
import torchaudio.transforms as tf
import torch.nn.functional as F


# Audio processing for KWS
def fix_length(audio, sr):
    # Padding if needed
    # For KWS, makes all samples 1 sec. long
    pad_right = np.random.randint(0, 2)
    if audio.size(1) < sr:
        p = sr - audio.size(1)
        if pad_right:
            audio = F.pad(audio, (0, p), 'constant', 0)
        else:
            audio = F.pad(audio, (p, 0), 'constant', 0)
    elif audio.size(1) > sr:
        start = np.random.randint(0, audio.size(1)-sr)
        audio = audio[:, start:start+sr]
    return audio

def resnorm(x, lam=0.1):
    if lam < 0:
        return x
    else:
        mean = x.mean((1,3), keepdim=True)
        std = ((x-mean)**2).mean((1,3), keepdim=True)
    return x*lam + (x-mean)/std


def bf_fact(x):
    mask = x.clone()
    mask[:] = 0
    size_dct = mask.size(0)
    for i in range(size_dct):
        idx = 2*i
        mask[idx:idx+2, idx:idx+2] = 1
    x = mask*x
    return x

class ButterflyMFCC(tf.MFCC):
    def forward(self, waveform):
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), bf_fact(self.dct_mat)).transpose(-1, -2)
        return mfcc

class LogMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_max,
                 pad_mode, center, mel_scale, window, alpha):
        super(LogMel, self).__init__()
        if window == 'hamming':
            window_fn = torch.hamming_window
        else:
            window_fn = torch.hann_window

        self.melspec = tf.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                       hop_length=hop_length, n_mels=n_mels,
                       f_max=f_max, pad_mode=pad_mode, center=center,
                       mel_scale=mel_scale, window_fn=window_fn)
        self.alpha=alpha

    def forward(self, x):
        x = self.melspec(x)
        x = torch.log10(x + 1e-8)
        x = resnorm(x, self.alpha)
        return x


# --- TEST ---
def test_logmel():
    feature = LogMel(16000, 2048, 480, 256, 8000,
                     'constant', True, 'slaney', 'hamming', 0.1)
    x = torch.randn(4, 1, 160000)
    x  = feature(x)
    assert x.shape == (4, 1, 256, 334) 

def test_fixlength():
    sr = 160
    x1 = torch.randn(1, 150)
    x2 = torch.randn(1, 220)
    x1 = fix_length(x1, sr) 
    x2 = fix_length(x2, sr) 
    assert x1.shape == (1, sr)
    assert x2.shape == (1, sr)
