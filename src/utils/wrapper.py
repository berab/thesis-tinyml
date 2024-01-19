import importlib


class Optim:
    def __init__(self, optim, scheduler, warmup):
        # Init optim
        mod = importlib.import_module("torch.optim")
        self.sched_cfg = None
        self.warmup_cfg = None

        self.optim = getattr(mod, optim['class'])
        del optim['class']
        self.optim_cfg = optim

        # Init schedulers
        mod = importlib.import_module("torch.optim.lr_scheduler")
        if scheduler:
            self.scheduler = getattr(mod, scheduler['class'])
            del scheduler['class']
            self.sched_cfg = scheduler

        if warmup:
            self.warmup = getattr(mod, warmup['class'])
            del warmup['class']
            self.warmup_cfg = warmup


    def with_params(self, params):
        self.optim = self.optim(params, **self.optim_cfg)

        if self.sched_cfg:
            self.scheduler = self.scheduler(self.optim, **self.sched_cfg)
        else:
            self.scheduler = 0
        if self.warmup_cfg:
            self.warmup = self.warmup(self.optim, **self.warmup_cfg)
        else:
            self.warmup = 0

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def step_sched(self, epoch=None):
        if (self.sched_cfg and self.warmup_cfg and
            epoch+1 > 5): # After 5 epoch of warmup
            self.scheduler.step()
        elif self.warmup_cfg:
            self.warmup.step()
        elif self.sched_cfg:
            self.scheduler.step()

