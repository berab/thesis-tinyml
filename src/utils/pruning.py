import torch
import torch.nn.utils.prune as prune


def modules2prune(model):
    modules = []
    module_names = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
            for param, _ in module.named_parameters():
                if 'weight' in param:
                    modules.append((module, param))
                    module_names.append(name)
        elif isinstance(module, torch.nn.Conv2d):
            modules.append((module, 'weight'))
            module_names.append(name)
        elif isinstance(module, torch.nn.Linear):
            modules.append((module, 'weight'))
            module_names.append(name)
    return modules, module_names

def mask_state(modules, module_names, init_state):
    with torch.no_grad():
        for module, module_name in zip(modules, module_names):
            module[0].weight_orig.copy_(init_state[module_name + '.weight'] * module[0].weight_mask)

def prune_remove(modules):
    # For module in modules:
    for module in modules:
        prune.remove(module[0], module[1])
       
def global_pruning(method, modules, p):
    # Global pruning, p: pruning percentage
    #if 'l1add' in method:
    #    method = L1AddUnstructured
    #else:
    method = prune.L1Unstructured
    prune.global_unstructured(modules, pruning_method=method, 
                              amount=p)

def layerwise_pruning(method, modules, p):
    #if 'l1add' in method:
    #    method = l1add_unstructured
    #else:
    method = prune.l1_unstructured

    print(len(modules)-2)
    for i, module in enumerate(modules):
        if i < 5 or i > len(modules)-2:
            method(module[0], name=module[1], amount=p/1.5)
        else:
            method(module[0], name=module[1], amount=p)

#class L1AddUnstructured(prune.L1Unstructured):
#
#    def compute_mask(self, t, default_mask):
#        # Check that the amount of units to prune is not > than the number of
#        # parameters in t
#        tensor_size = t.nelement()
#        # Compute number of units to prune: amount if int,
#        # else amount * tensor_size
#        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
#        # This should raise an error if the number of units to prune is larger
#        # than the number of units in the tensor
#        _validate_pruning_amount(nparams_toprune, tensor_size)
#
#        mask = default_mask.clone(memory_format=torch.contiguous_format)
#
#        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
#            # largest=True --> top k; largest=False --> bottom k
#            # Prune the smallest k
#            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
#            small_add = topk[0].sum()/topk[0].nelement()
#            
#            #t += small_add
#            t += 10
#            mask.view(-1)[topk.indices] = 0
#        return mask
#
#def l1add_unstructured(module, name, amount):
#    L1AddUnstructured.apply(module, name, amount)
#    return module

def _compute_nparams_toprune(amount, tensor_size):
    import numbers
    r"""Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.
    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    Returns:
        int: the number of units to prune in the tensor
    """
    # incorrect type already checked in _validate_pruning_amount_init
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return int(round(amount * tensor_size))  # int needed for Python 2

def _validate_pruning_amount(amount, tensor_size):
    import numbers
    r"""Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).
    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """
    # TODO: consider removing this check and allowing users to specify
    # a number of units to prune that is greater than the number of units
    # left to prune. In this case, the tensor will just be fully pruned.

    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            "amount={} should be smaller than the number of "
            "parameters to prune={}".format(amount, tensor_size)
        )

