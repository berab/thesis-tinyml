import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchaudio.transforms as tf
from utils.utils import load_model, nemo_dict_handler, validate
from utils.aug_funcs import temporalShiftBatch, mixup, mixup_criterion
from utils.pruning import prune_remove, layerwise_pruning, global_pruning, modules2prune
from tqdm import tqdm


# ---- BASE ----
def train(cfg, trainloader, criterion, epoch, modules=None):

    correct = 0
    total = 0

    cfg.model.train()

    if cfg.mask:
        freq_mask = tf.FrequencyMasking(cfg.mask.freq, False)
        time_mask = tf.TimeMasking(cfg.mask.time, False)

    # Iterative pruning
    # And
    # Knowledge distillation
    if cfg.tmodel and modules:
        kd_criterion = nn.MSELoss()
        sparsity = cfg.monitor.calculate_sparsity(modules)
    elif modules:
        cycle = int(cfg.n_epochs/cfg.n_rounds)
        assert cycle != 0
        if ((epoch+1) % cycle  == 0
            and epoch < cfg.n_rounds*cycle):
            # This formula is a bit weird since each time
            # 'the parameters that survive' are pruned
            p = 1 - (1 - cfg.p/100.)**(1/cfg.n_rounds)
            if 'global' in cfg.p_method:
                global_pruning(cfg.p_method, modules, p)
            elif 'layerwise' in cfg.p_method:
                layerwise_pruning(cfg.p_method, modules, p)
            sparsity = cfg.monitor.calculate_sparsity(modules)
        else:
            sparsity = cfg.monitor.calculate_sparsity(modules)
        
    # Continuous scheduler
    #breakpoint()
    cont_step = False
    if isinstance(cfg.optim.scheduler, optim.lr_scheduler.CosineAnnealingLR):
        cont_step = True

    # Training starts
    with tqdm(trainloader) as loader:
        for i, data in enumerate(loader):
            loader.set_description(f"Epoch {epoch + 1}")
            inputs, targets = data['audio'].to(cfg.target), data['target'].to(cfg.target)
            
            inputs = cfg.feature(inputs)
            inputs = temporalShiftBatch(inputs, rate=cfg.shift)

            # Aug.
            if cfg.mask:
                inputs = freq_mask(time_mask(inputs))
                inputs = freq_mask(time_mask(inputs))

            # Mixup, set mixup_alpha=0 to disable
            inputs, targets_a, targets_b, lam = mixup(inputs, targets
                                                          ,cfg.mixup_alpha, cfg.target)
            outputs = cfg.model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Knowledge distillation
            if cfg.tmodel:
                toutputs = cfg.tmodel(inputs)
                kd_loss = kd_criterion(outputs, toutputs) 
                loss = (1-cfg.kd_alpha)*loss + cfg.kd_alpha*kd_loss

            # Mixup preds
            preds = torch.argmax(outputs, dim=-1)
            total += cfg.loader.batch_size
            correct += (lam*((targets_a == preds).float().sum())
                        + (1-lam)*((targets_b == preds).float().sum())).item()

            cfg.optim.step(loss) # Optimizer step + backward pass
            # Log loss & amount of time processed per second [sec/sec] 
            # Note: change 10 --> 1  if dcase22 since each sample is one sec long there
            time_processed = (i+1)*cfg.loader.duration*cfg.loader.batch_size
            time_elapsed = loader.format_dict['elapsed'] # check this format_dict, whats inside?
            loader.set_postfix(acc=correct/total, loss=loss.item(), 
                               speed=loader.format_interval(time_processed/time_elapsed))

            if cont_step:
                cfg.optim.step_sched(epoch=epoch)
        if not cont_step:
            cfg.optim.step_sched(epoch=epoch)
    return 100.*correct/total

def test(cfg, validloader, criterion):

    cfg.model.eval()
    # Initialize to calculate the scores to save after training
    output_list = []
    target_list = []
    loss_list = []
    softmax = nn.Softmax()
    
    with torch.no_grad(), tqdm(validloader) as vloader:
        for data in vloader:
            inputs, targets = data['audio'].to(cfg.target), data['target'].to(cfg.target)

            inputs = cfg.feature(inputs)

            outputs = cfg.model(inputs)
            loss = criterion(outputs, targets)

            # Appending the target and predicted list
            loss_list.append(loss)
            output_list.append(softmax(outputs))
            target_list.append(targets)

    return output_list, target_list, loss_list

def single_test(cfg): 
    cfg.feature.to(cfg.target)
    testloader = cfg.loader.testloader()

    criterion = nn.CrossEntropyLoss()

    # Load the model
    load_model(cfg)

    print('\nTesting...')
    output_list, target_list , loss_list = test(cfg, testloader, criterion)
    _, targets, preds = cfg.monitor.calculate_scores(output_list, target_list, loss_list, 0)
    cfg.monitor.log_conf(targets, preds)
    cfg.monitor.print_scores()
    if cfg.save:
        cfg.monitor.save_model(cfg.model)

# --- QUANTIZATION ----
def nvidia_quant(cfg):
    from pytorch_quantization import quant_modules
    from utils.quant_funcs import calibrate_model

    trainloader = cfg.loader.trainloader(cfg.seed)
    validloader = cfg.loader.validloader()
    criterion = nn.CrossEntropyLoss()

    if cfg.test:
        single_test(cfg)
    else:
        load_model(cfg)

    # Fake quantization
    print('\n Calibration...')
    cfg.model.load_state_dict(cfg.model.state_dict())
    with torch.no_grad():
        calibrate_model(
            cfg=cfg,
            model_name="model",
            data_loader=trainloader,
            num_calib_batch=cfg.loader.batch_size,
            calibrator=cfg.q_method,
            hist_percentile=[99.9, 99.99, 99.999, 99.9999],
            out_dir="./")

    print('\n Validation...')
    output_list, target_list , loss_list = test(cfg, validloader, criterion)
    cfg.monitor.calculate_scores(output_list, target_list, loss_list, 0)
    cfg.monitor.print_scores()
    
    if cfg.n_epochs:
        print('\n Fine-tuning...')
    
def nemo_quant(cfg): 
    import nemo
 
    trainloader = cfg.loader.trainloader(cfg.seed)
    validloader = cfg.loader.testloader()
    criterion = nn.CrossEntropyLoss()

    # Note that single_test also loads the model
    if cfg.test:
        single_test(cfg)
    else:
        # Load the model
        load_model(cfg)

    # FakeQuant
    in_shape = [1, cfg.nemo.n_chan, cfg.nemo.F, cfg.nemo.T] 
    cfg.model = nemo.transform.quantize_pact(cfg.model 
                                             , dummy_input=torch.randn(in_shape).to(cfg.target))
    # This function is needed because of hydra's dot notation. Otherwise we cannot use
    # multirun for bits
    nemo_dict_handler(cfg.nemo.model_bits)
    # NOTE:You can learn the quantizable layers by setting verbose=True
    cfg.model.change_precision(bits=1, min_prec_dict=cfg.nemo.model_bits, verbose=cfg.nemo.verbose)

    print('\n Calibration...')
    with cfg.model.statistics_act():
        test(cfg, trainloader, criterion)
    cfg.model.reset_alpha_act()

    print('\n Test...')
    output_list, target_list , loss_list = test(cfg, validloader, criterion)
    cfg.monitor.calculate_scores(output_list, target_list, loss_list, 0)
    cfg.monitor.print_scores()
    
    if cfg.n_epochs:
        print('\n Fine-tuning...')


    #  ------------ DEPLOYMENT -------------
    #Folding
    #print('\nBN folding...')
    #fg.model.fold_bn()
    #fg.model.reset_alpha_weights()
    #output_list, target_list , loss_list = test(cfg, validloader, criterion)
    #cfg.monitor.calculate_acc(output_list, target_list, loss_list)
    #cfg.monitor.print_scores()
    

    #print('\nQuantized deployable stage...')
    #cfg.model = nemo.transform.bn_to_identity(cfg.model)
    #cfg.model.qd_stage(eps_in=1./2)
    #output_list, target_list , loss_list = test(cfg, validloader, criterion)
    #cfg.monitor.calculate_acc(output_list, target_list, loss_list)
    #cfg.monitor.print_scores()
    

    #print('\nInteger deployable stage...')
    #cfg.model.id_stage()
    #output_list, target_list , loss_list = test(cfg, validloader, criterion)
    #cfg.monitor.calculate_acc(output_list, target_list, loss_list)
    #cfg.monitor.print_scores()

 
# ---- PRUNING ---- 
def prune(cfg):
    criterion = nn.CrossEntropyLoss()

    # Prune all the module
    modules, _ = modules2prune(cfg.model) 

    # Test also loads the model
    if cfg.test:
        single_test(cfg)
        cfg.monitor.append_prunep(0)
    else:
        # Load the model
        load_model(cfg)

    testloader = cfg.loader.testloader()

    # Global pruning
    for perc in range(0, 100, 1):
        p = perc/100.
        if 'global' in cfg.p_method:
            global_pruning(cfg.p_method, modules, p)
        elif 'layerwise' in cfg.p_method:
            layerwise_pruning(cfg.p_method, modules, p)
        prune_remove(modules)
        cfg.monitor.calculate_sparsity(modules)

        print('Testing after pruning...')
        output_list, target_list , loss_list = test(cfg, testloader, criterion)
        cfg.monitor.append_prunep(p)
        cfg.monitor.calculate_acc(output_list, target_list, loss_list, 0)
        cfg.monitor.print_scores()
        if cfg.save:
            cfg.monitor.just_save(cfg.model, p)
        # Reset for another itr.
        load_model(cfg)
    cfg.monitor.log_prune()

