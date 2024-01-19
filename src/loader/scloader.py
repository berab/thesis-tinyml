import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS 
from torchvision.transforms import Compose
from utils.aug_funcs import Noise, TempShift, Amplify
from utils.audio_proc import fix_length
from pathlib import Path


class SCLoader():
    def __init__(self, dataset_dir, batch_size, n_workers, data):
        dataset_dir = Path(dataset_dir)
        noise_dir = dataset_dir / '_background_noise_' / '*.wav'
        noise_files = glob.glob(str(noise_dir))

        transforms = Compose([TempShift(0.2), Amplify((0.2, 1.5)), Noise(noise_files, 0.7, (0, 0.7))])

        self.train_dataset = SubsetSC(dataset_dir, 'training', transforms)
        self.val_dataset = SubsetSC(dataset_dir, 'validation')
        self.test_dataset = SubsetSC(dataset_dir, 'testing')

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.labels = sorted(list(set(datapoint['target'] for datapoint in self.val_dataset)))
        self.duration = data.duration

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def collate_fn(self, batch):
        audios, targets = [], []
        for data in batch:
            audios += [data['audio']]
            targets += [self.label_to_index(data['target'])]

        return {'audio': torch.stack(audios), 'target': torch.stack(targets)}

    def trainloader(self, seed=42):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed()%2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn,
        num_workers=self.n_workers, worker_init_fn=seed_worker, generator=g, shuffle=True)

    def validloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn,
        num_workers=self.n_workers, shuffle=False)

    def testloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, 
        num_workers=self.n_workers, shuffle=False)

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dataset_dir: str=None, subset: str=None, transforms=None):
        super().__init__(dataset_dir.parents[1], download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list(dataset_dir / "validation_list.txt")
        elif subset == "testing":
            self._walker = load_list(dataset_dir / "testing_list.txt")
        elif subset == "training":
            excludes = load_list(dataset_dir / "validation_list.txt") + load_list(dataset_dir 
                        / "testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        self.transforms = transforms

    # [Debug]
    #def __len__(self):
    #    return 1

    def __getitem__(self, idx):

        fileid = self._walker[idx]
        relpath = os.path.relpath(fileid, self._path)
        target = os.path.split(relpath)[0]
        audio, sr = torchaudio.load(fileid)
        audio = fix_length(audio, sr) 
        if self.transforms and np.random.random() > 0.8:
            audio = self.transforms(audio)

        return {'audio': audio, 'target': target}
   
    def label_to_index(self, word):
        return torch.Tensor(self.labels.index(word))

