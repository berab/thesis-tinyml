import numpy as np
import pandas as pd
import random
import torch
import torchaudio
import torchaudio.transforms as tf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DCASELoader():
    def __init__(self, dataset_dir, batch_size, n_workers, data):
        self.train_dataset = DCASEDataset(dataset_dir, 'setup/fold1_train.csv', data)
        self.valid_dataset = DCASEDataset(dataset_dir, 'setup/fold1_evaluate.csv', data)         
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.duration = data.duration

    def trainloader(self, seed=42):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed()%2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
        num_workers=self.n_workers, worker_init_fn=seed_worker, generator=g, shuffle=True)

    def validloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, 
        num_workers=self.n_workers, shuffle=False)
    
    def testloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, 
        num_workers=self.n_workers, shuffle=False)
    

class DCASEDataset(Dataset):
    def __init__(self, dataset_dir, csv_file, data):
        self.dataset_dir = Path(dataset_dir)
        csv_path = self.dataset_dir / csv_file 

        df = pd.read_csv(csv_path, sep='\t')
        self.filenames = df['filename']
        self.targets = df['scene_label'].astype('category').cat.codes.values

        self.resampler = tf.Resample(orig_freq=data.sr, new_freq=data.resample_rate,
                                     lowpass_filter_width=64, rolloff=0.9476,
                                     resampling_method='kaiser_window', beta=14.769656459479492)
 
    def __len__(self):
        #return 1
        return len(self.targets)

    def __getitem__(self, idx, valid=False):

        target = int(self.targets[idx])
        audio, _ = torchaudio.load(self.dataset_dir / self.filenames[idx], normalize=True)
        audio = self.resampler(audio)

        sample = {'audio': audio, 'target': target}
        return sample
   
