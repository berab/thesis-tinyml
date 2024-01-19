import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.pruning import global_pruning, modules2prune, prune_remove
import os
from os.path import isfile
from hydra import utils
from pathlib import Path


def load_model(cfg):
    # CUDA / CPU
    cfg.model.to(cfg.target)
    cfg.feature.to(cfg.target)
    
    cfg.model_dir = Path(cfg.wdir) / cfg.model_dir
    loaded_dict = torch.load(cfg.model_dir)

    if not cfg.tmodel:
        state_dict = cfg.model.state_dict()

        for module in cfg.model.state_dict():
            if module + '_mask' in loaded_dict: # If masked
                state_dict[module] = loaded_dict[module + '_orig']*loaded_dict[module + '_mask']
            else:
                state_dict[module] = loaded_dict[module]

        cfg.model.load_state_dict(state_dict)

    # KD
    else:
        cfg.tmodel.to(cfg.target)

        cfg.model.load_state_dict(loaded_dict)
        cfg.tmodel_dir = Path(cfg.wdir) / cfg.tmodel_dir
        teacher_dict = torch.load(cfg.tmodel_dir)
        cfg.tmodel.load_state_dict(teacher_dict)
        
# GPU/CPU
def get_training_backend(target):
    if target == 'cpu':
        if torch.cuda.is_available():
            print("Warning: You are using CPU training, but CUDA backend would be available")

        device = "cpu"
    elif target.device == 'cuda':
        if not torch.cuda.is_available():
            raise Exception("CUDA backend selected, but not available on current host")

        if target.ids:
            gpu_ids = ",".join(map(lambda x: str(x), target.ids))
        else:
            print("No GPU selected, using cuda:0")
            gpu_ids = '0'

        os.environ['CUDA_VISIBLE_DEVICES']=gpu_ids
        device = "cuda:{}".format(gpu_ids)
    return torch.device(device)


# EXPORTING SVG
def export_svg(df, path, score_names):
    # Scores
    import math
    n_scorePlots = math.ceil(len(score_names)/4)
    for i in range(n_scorePlots):
        scores = df[score_names[i*4:i*4+4]] # Take 4 items
        #axes = scores.plot(linestyle='None', style=['xr','+b', '|c', '_m'])
        axes = scores.plot(style=['r','b', 'c', 'm'])
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Scores')
        axes.set_ylim([0, 1])
        axes.grid(linestyle='--')
        plt.savefig(path / f'scores-{i}.svg', format='svg')

    # Loss
    loss = df[['loss']]
    axes = loss.plot(style='r')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_ylim([0,3])
    axes.grid(linestyle='--')
    plt.savefig(path / f'loss.svg', format='svg')

    ## Accuracy
    accuracy = df[['val_acc', 'train_acc']]

    axes = accuracy.plot(style=['c', 'y'], linewidth=.5)

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Accuracy[%]')

    # Ticks and grids
    limit = math.ceil(len(accuracy) / 50)*50
    minor_ticks = np.arange(0, 301, 5)
    major_ticks = np.arange(0, 301, 20)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks )
    axes.set_yticks(major_ticks )
    axes.grid(which='minor', linestyle='--', alpha=0.2)
    axes.grid(which='major', linestyle='--', alpha=0.5)
    axes.set_xlim([0, limit])
    axes.set_ylim([0, 100])
    plt.savefig(path / f'accuracies.svg', format='svg')

def export_lotterySvg(df, path):

    ## Accuracy
    accuracy = df.drop(['loss'], axis=1)
    axes = accuracy.plot(style=['c', 'y', 'r', 'm', 'g', 'b'], linewidth=.5)

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Val. accuracy[%]')

    # Ticks and grids
    import math
    limit = math.ceil(len(accuracy) / 50)*50
    minor_ticks = np.arange(0, 301, 5)
    major_ticks = np.arange(0, 301, 20)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks )
    axes.set_yticks(major_ticks )
    axes.grid(which='minor', linestyle='--', alpha=0.2)
    axes.grid(which='major', linestyle='--', alpha=0.5)
    axes.set_xlim([0, limit])
    axes.set_ylim([0, 100])
    plt.savefig(path / f'accuracies.svg', format='svg')


def export_pruneSvg(df, path):

    ## Accuracpy
    #accuracy = df[['val_acc']]
    df.head()
    axes = df[0].plot(style='c')

    axes.set_xlabel('Sparsity[%]')
    axes.set_ylabel('Accuracy[%]')

    # Ticks and grids
    import math
    limit = math.ceil(len(df) / 50)*50
    minor_ticks = np.arange(0, 301, 5)
    major_ticks = np.arange(0, 301, 20)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)
    axes.set_yticks(major_ticks )
    axes.set_xticks(major_ticks )
    axes.grid(which='minor', linestyle='--', alpha=0.2)
    axes.grid(which='major', linestyle='--', alpha=0.5)
    axes.set_xlim([0,limit])
    axes.set_ylim([0,100])
    plt.savefig(path / 'acc-p.svg', format='svg')

def export_sparsitySvg(df, path):
    sparsity = df[['Perc. of weights']]
    axes = sparsity.plot(linestyle='None', style='-xr')

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Perc. of weights[%]')

    # Ticks and grids
    axes.grid()
    axes.set_ylim([0,100])
    plt.savefig(path / 'pweights.svg', format='svg')

def export_allSvg(paths):

    #breakpoint()
    val_accs = pd.DataFrame()
    for path in paths:
        csv_file = path / 'scores.csv'
        df = pd.read_csv(csv_file)
        val_accs.insert(0, f'acc{str(path.name)}', df['acc'])
    
    axes = val_accs.plot(linewidth=.5)
    axes.set_xlabel('epoch')
    axes.set_ylabel('Accuracy[%]')
    
    # Ticks and grids
    minor_ticks = np.arange(0, 301, 5)
    major_ticks = np.arange(0, 301, 20)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(minor_ticks, minor=True)
    axes.set_xticks(major_ticks )
    axes.set_yticks(major_ticks )
    axes.grid(which='minor', linestyle='--', alpha=0.2)
    axes.grid(which='major', linestyle='--', alpha=0.5)
    axes.set_xlim([0,300])
    axes.set_ylim([0,100])
    plt.savefig(path.parent / 'multirun-acc.svg', format='svg')

# NEMO
def nemo_dict_handler(bit_dict):
    items = bit_dict.items()
    for i in range(len(bit_dict)):
        key = items[i][0]
        new_key = key.split('__')
        new_key = ('.').join(new_key)
        bit_dict[new_key] = bit_dict.pop(key)
    
# MACS/PARAMS
def validate(macc, params):

    # From DCASE22
    MAX_MACC=30e6       #30M MACC
    MAX_PARAMS=128e3    #128K params

    print('Model statistics:')
    print('MACC:\t \t %.3f' %  (macc/1e6), 'M')
    print('Params:\t \t %.3f' %  (params/1e3), 'K\n')
    if macc>MAX_MACC:
        print('[Warning] In case of ASC, multiply accumulate count', macc, 'is more than the allowed maximum of', int(MAX_MACC))
    if params>MAX_PARAMS:
        print('[Warning] In case of ASC, parameter count', params, 'is more than the allowed maximum of', int(MAX_PARAMS))
