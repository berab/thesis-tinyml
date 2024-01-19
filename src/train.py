import torch
import torch.nn as nn
from pathlib import Path
from methods import train, test, nemo_quant, nvidia_quant
from utils.pruning import modules2prune, layerwise_pruning, global_pruning, mask_state
from utils.utils import load_model

#import torch_tensorrt
# TensorRT warnings: Set torch_tensorrt logging level LogLevel: ERROR, DEBUG, WARNING
#from torch_tensorrt import logging
#logging.Level(logging.LogLevel.INFO)
#logging.set_reportable_log_level(logging.Level(logging.LogLevel.ERROR))


def single_run(cfg):
    trainloader = cfg.loader.trainloader(cfg.seed)
    validloader = cfg.loader.validloader()

    cfg.model.to(cfg.target)
    cfg.feature.to(cfg.target)
    criterion = nn.CrossEntropyLoss()

    # Iterative pruning
    if cfg.apply == 'itr-prune':
        modules, _ = modules2prune(cfg.model)
        # For converging after the last pruning
        n_epochs = cfg.n_epochs + 20
    # KD
    elif cfg.tmodel:
        modules, _ = modules2prune(cfg.model)
        if 'global' in cfg.p_method:
            global_pruning(cfg.p_method, modules, p)
        elif 'layerwise' in cfg.p_method:
            layerwise_pruning(cfg.p_method, modules, p)

        load_model(cfg)
        n_epochs = cfg.n_epochs
    else:
        modules = None
        n_epochs = cfg.n_epochs

    # Quantization
    if cfg.model.quant == 'nemo':
        nemo_quant(cfg)
    elif cfg.model.quant == 'nvidia':
        nvidia_quant(cfg)

    # Training/QAT/Fine-tuning
    for epoch in range(n_epochs):
        train_acc = train(cfg, trainloader, criterion, epoch, modules=modules)
        output_list, target_list, loss_list = test(cfg, validloader, criterion)

        cfg.monitor.calculate_scores(output_list, target_list, loss_list, train_acc)
        cfg.monitor.print_scores()
        if cfg.save:
            cfg.monitor.save_model(cfg.model, epoch)

        # Early stopping
        if cfg.monitor.stop_counter == cfg.patience:
            print('\nEarly stopping...')
            cfg.monitor.log_scores()
            break
    cfg.monitor.log_scores()

def lottery_run(cfg):

    trainloader = cfg.loader.trainloader(cfg.seed)
    validloader = cfg.loader.validloader()

    criterion = nn.CrossEntropyLoss()
    cfg.feature.to(cfg.target)
    cfg.model.to(cfg.target)

    modules, module_names = modules2prune(cfg.model) 

    # Store initial states
    if cfg.optim.warmup_cfg:
        warmup_state = cfg.optim.warmup.state_dict()
    if cfg.optim.sched_cfg:
        sched_state = cfg.optim.scheduler.state_dict()
    optim_state = cfg.optim.optim.state_dict()
    torch.save(cfg.model.state_dict(), 'init-state.pt')
    init_state = torch.load('init-state.pt')

    # I made +1 since last round not pruned. Is this correct?
    for run in range(cfg.n_rounds+1):

        # Print and save sparsity
        sparsity = cfg.monitor.calculate_sparsity(modules)
        print(f'Run: {run+1}')
        
        for epoch in range(cfg.n_epochs):
            train(cfg, trainloader, criterion, epoch)
            output_list, target_list, loss_list = test(cfg, validloader, criterion)

            cfg.monitor.calculate_scoresLottery(output_list, target_list, loss_list, run=run)
            cfg.monitor.print_scores()
            if cfg.save:
                cfg.monitor.save_model(cfg.model, epoch)

            # Early stopping
            if cfg.monitor.stop_counter == cfg.patience:
                print('Early stopping...')
                cfg.monitor.log_scores()
                break

        # No pruning after the last round 
        #breakpoint()
        if run != cfg.n_rounds:
            # Pruning
            p = 1 - (1 - cfg.p/100.)**(1/cfg.n_rounds)
            #p = (cfg.p/100.)/(cfg.one)
            #cfg.one -= cfg.p/100.

            if 'global' in cfg.p_method:
                global_pruning(cfg.p_method, modules, p)
            elif 'layerwise' in cfg.p_method:
                layerwise_pruning(cfg.p_method, modules, p)

            # Loading masked dict 
            mask_state(modules, module_names, init_state)

        # If warmup/scheduler used, restart
        cfg.optim.optim.load_state_dict(optim_state)
        if cfg.optim.warmup_cfg:
            cfg.optim.warmup.load_state_dict(warmup_state)
        if cfg.optim.sched_cfg:
            cfg.optim.scheduler.load_state_dict(sched_state)

    cfg.monitor.log_scores()
 


