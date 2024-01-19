import hydra
import random
import numpy as np
import torch
from torch.nn import Module
from typing import Any
from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
from hydra.experimental.callback import Callback
from utils import wrapper, utils, monitor
from methods import single_test, prune
from train import single_run, lottery_run


@hydra.main(config_path="../config/", config_name="asc")
def main(cfg: DictConfig):

    # Initialize RNGs
    cfg.seed = cfg['seed'] or random.randint(0, 1e15)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    print('Using torch deterministic algorithms...')
    if cfg.model.quant == 'nemo': # torch==1.9 doesn`t support warn_only
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Instantiate classes from configuration file
    cfg = instantiate(cfg)

    # Initialize optimizer
    cfg.optim.with_params(cfg.model.parameters())

    # Training backend
    cfg.target = utils.get_training_backend(cfg.target)

    # Applications
    
    if (cfg.apply == 'train' or
        cfg.apply == 'itr-prune' or
        cfg.apply == 'brevitas'):
        single_run(cfg)
    elif cfg.apply == 'lottery':
        lottery_run(cfg)
    elif cfg.apply == 'prune':
        prune(cfg)
    elif cfg.apply == 'test':
        single_test(cfg)
    else:
        raise Exception("xD Not a valid application. Just give me smt...")

if __name__ == "__main__":
    main()


@dataclass
class Nemo:
    in_shape = None

@dataclass
class Mask:
    freq = 0
    time = 0


@dataclass
class Target:
    device:str = 'cpu'
    ids:int = 0

@dataclass
class Dataloader:
    dataset_dir:str = 'datasets/dcase20'
    batch_size:int = 32
    n_workers:int = 8

@dataclass
class Main:
    model: Module
    optim: wrapper.Optim
    monitor: monitor.Monitor
    loader: Dataloader
    wdir: Path
    monitor: monitor.Monitor
    feature: Module

    # KD
    tmodel: Module
    kd_alpha: float

    save:bool

    # Augmentation parameters
    mask: Mask

    # Application name
    apply:str

    # State dict. dirs
    model_dir:str
    tmodel_dir:str

    # Quantization
    nemo: Nemo

    # Pruning and Lottery ticket params
    p: int
    p_method: str
    n_rounds: int

    # Early stopping
    patience: int
    
    # Nvidia quant.
    q_method:str = 'max'

    # Aug.
    shift:int = 0

    # Test before quant. or prun.
    test:bool = False
    
    target:Target = 'cpu'
    seed:int = None
    mixup_alpha:float = 0.3

    # Training parameters
    n_epochs:int = 1

    one:int = 1

    def __post_init__(self):
        self.wdir = Path(self.wdir)


class MyCallback(Callback):
    def __init__(self):
        self.paths = []
    
    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        self.paths.append(Path.cwd())

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print(f"Multirun ended, exporting SVG...")
        utils.export_allSvg(self.paths)
        print('Done.')
        
