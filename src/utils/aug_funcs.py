import torch
import torch.nn.functional as F
import torchaudio
from utils.audio_proc import fix_length
import numpy as np


def mixup(x, y, alpha=0.3, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)

# Melspec. augmentation
def temporalShiftBatch(x, rate=0.15):
    # IN: N, C, F, T; OUT: N, C, F, T
    max_shift = int(rate*x.size(-1))
    if max_shift > 0:
        shift_value = np.random.randint(low=-max_shift, high=max_shift)
    else:
        shift_value = 0
    return x.roll(shifts=shift_value, dims=-1)

# Raw audio augmentations
class TempShift():
    def __init__(self, rate=0.2):
        self.rate = rate

    def __call__(self, x):
        rate = np.random.uniform(-self.rate, self.rate)
        shift_value = int(x.size(-1)*rate)
        return x.roll(shifts=shift_value, dims=-1)

class Amplify():
    def __init__(self, rate=(0.2, 1.5)):
        self.rate = rate

    def __call__(self, x):
        rate = np.random.uniform(self.rate[0], self.rate[1])
        x = x*rate
        if rate > 1:
            x[x>1] = 1
            x[x>-1] = -1
        return x

class Noise():
    def __init__(self, noise_files, prob_noise=0.5, intensity=(0, 0.5)):

        # Load noises
        noises = []
        for noise_file in noise_files:
            noise, sr = torchaudio.load(noise_file)
            noise = fix_length(noise, sr)
            noises.append(noise)

        self.noises = noises
        self.prob_noise = prob_noise
        self.intensity = intensity
             
    def __call__(self, x):
        if np.random.random() > self.prob_noise: # No noise
            return x

        # Add noise
        i = np.random.randint(len(self.noises))
        intensity = np.random.uniform(self.intensity[0], self.intensity[1])
        noise = self.noises[i]*intensity 
        x = x + noise 
        x[x>1] = 1
        x[x>-1] = -1
        return x


# Tests
def test_temporalShiftBatch():
    x = torch.randn(4, 1, 256, 334)
    x = temporalShiftBatch(x, 0.5)
    assert x.shape == (4, 1, 256, 334)
    

