import torch
import numpy as np
import importlib
import pandas as pd
from pathlib import Path
from utils.utils import export_svg, export_lotterySvg, export_sparsitySvg, export_pruneSvg
from utils.pruning import prune_remove
from torchmetrics import ConfusionMatrix, FBetaScore

# Ignore noisy GPU memory warning from torchmetrics
import warnings
warnings.filterwarnings('ignore')


class Monitor:
    def __init__(self, scores, n_class, device):
        self.device = device

        # Inits for scores
        self.n_class = n_class
        self.scores = scores
        self.score_dict = {}
        self.method_dict = {}

        # Import torchmetrics to take methods from it later on
        self.metrics = importlib.import_module('torchmetrics')

        # Create the models saving directory
        self.model_dir = Path.cwd() / 'models'
        self.model_dir.mkdir()

        self.best_acc = 0
        self.best_loss = 100
        self.best_epoch = 0

        # Lottery ticket
        self.cur_run = None
        self.val_name = 'val_acc'

        # Early stopping
        self.stop_counter = 0
        
    def save_model(self, model, epoch=0):
        # Save if better acc/loss
        
        if self.score_dict[self.val_name][-1] > self.best_acc:
            self.best_acc = self.score_dict[self.val_name][-1]
            self.best_epoch = epoch

            if self.cur_run != None: # If lottery
                torch.save(model.state_dict(), self.model_dir / f'model-{self.cur_run}-{epoch}.pt')
            else:
                torch.save(model.state_dict(), self.model_dir / f'model-{epoch}.pt')

        elif self.score_dict['loss'][-1] < self.best_loss:
            self.best_loss = self.score_dict['loss'][-1]
            self.best_epoch = epoch

            if self.cur_run != None: # If lottery
                torch.save(model.state_dict(), self.model_dir / f'model-{self.cur_run}-{epoch}.pt')
            else:
                torch.save(model.state_dict(), self.model_dir / f'model-{epoch}.pt')

    def just_save(self, model, epoch):
        torch.save(model.state_dict(), self.model_dir / f'model-{epoch}.pt')

    def remove_pruning(self, model, modules):
        # Remove pruning of the best result
        state_dict = torch.load(self.model_dir / f'model-{self.best_epoch}.pt')
        model.load_state_dict(state_dict)
        prune_remove(modules)
        torch.save(model.state_dict(), self.model_dir / f'model-premoved-{self.best_epoch}.pt')

    def log_prune(self):
        print('Done.')
        # Save accuracies and pruning ratios
        df = pd.DataFrame(self.score_dict['val_acc'], index=self.score_dict['p'])
        df.to_csv(Path.cwd() / 'acc-p.csv')
        export_pruneSvg(df, Path.cwd())

    def log_scores(self):
        print('Finished.')

        # Sparsity for itr-prune
        if 'Perc. of weights' in self.score_dict:
            pweights = {'Perc. of weights': self.score_dict['Perc. of weights']}
            df = pd.DataFrame(pweights)
            df.to_csv(Path.cwd() / 'pweights.csv', index=True, index_label='epoch')
            export_sparsitySvg(df, Path.cwd())
            
            self.score_dict.pop('Perc. of weights')

        # lottery or not
        if 'val_acc' not in self.score_dict:
            df = pd.DataFrame(self.score_dict)
            df.to_csv(Path.cwd() / 'scores.csv', index=True, index_label='epoch')
            export_lotterySvg(df, Path.cwd())

        else:
            # Save accuracies
            df = pd.DataFrame(self.score_dict)
            export_svg(df, Path.cwd(), self.scores.methods)
            if self.cur_run != None:
                df.to_csv(Path.cwd() / f'scores-{self.cur_run}.csv', index=True, index_label='epoch')
            else:
                df.to_csv(Path.cwd() / f'scores.csv', index=True, index_label='epoch')
        
            
    def calculate_acc(self, output_list, target_list, loss_list, train_acc=0):

        outputs = torch.cat(output_list)
        targets = torch.cat(target_list)
        preds = torch.argmax(outputs, dim=-1)

        # Calculate accuracy, loss
        loss = torch.stack(loss_list).mean().item()
        correct = ((targets == preds).float().mean().item())
        acc = correct*100

        
        if 'val_acc' in self.score_dict:
            self.score_dict['val_acc'].append(round(acc, 2))
            self.score_dict['train_acc'].append(round(train_acc, 2))

            # Early stop counter increase
            if loss > self.score_dict['loss'][-1]:
                self.stop_counter += 1
            else:
                self.stop_counter = 0

            self.score_dict['loss'].append(loss)
        # One time creation
        else:
            self.score_dict['val_acc'] = [(round(acc, 2))]
            self.score_dict['train_acc'] = [round(train_acc, 2)]
            self.score_dict['loss'] = [loss]
        return outputs, targets, preds

    def calculate_scoresLottery(self, output_list, target_list, loss_list, run):

        # Updating run index and 
        # resetting the best accuracy for the next run
        if run != self.cur_run:
            self.cur_run = run
            self.best_acc = 0
            self.score_dict['loss'] = []
            self.stop_counter = 0

        outputs = torch.cat(output_list)
        targets = torch.cat(target_list)
        preds = torch.argmax(outputs, dim=-1)

        # Calculate accuracy, loss
        loss = torch.stack(loss_list).mean().item()
        correct = ((targets == preds).float().mean().item())
        acc = correct*100

        val_name = f'val_acc-{self.score_dict["Perc. of weights"][-1]}'
        if val_name in self.score_dict:
            self.score_dict[val_name].append(round(acc, 2))
            self.score_dict['loss'].append(loss)
        # One time creation
        else:
            self.score_dict[val_name] = [(round(acc, 2))]
            self.score_dict['loss'] = [loss]

        # Early stopping
        if loss > self.score_dict['loss'][-1]:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        self.val_name = val_name
        return outputs, targets, preds


    
    # For prune
    def append_prunep(self, p):
        if 'p' in self.score_dict:
            self.score_dict['p'].append(round(p, 2))
        else:
            self.score_dict['p'] = [round(p,2)]

    # For itr-prune
    def calculate_sparsity(self, modules):
        total_zeros = 0
        total_elem = 0
        for module, _ in modules:
            total_zeros += float(torch.sum(module.weight == 0))
            total_elem += float(module.weight.nelement())
        sparsity = 100*total_zeros/total_elem
        pweights = round(100 - sparsity, 1)
        print('Perc. of weights: {}%'.format(pweights))

        if 'Perc. of weights' in self.score_dict:
            self.score_dict['Perc. of weights'].append(pweights)
        else:
            self.score_dict['Perc. of weights'] = [pweights]
   
    def calculate_scores(self, output_list, target_list, loss_list, train_acc=0):

        outputs, targets, preds = self.calculate_acc(output_list, target_list, loss_list, train_acc)

        # Other scores (I think this code is something beautiful)
        for method, kwargs in zip(self.scores.methods, self.scores.kwargs):
            if method in self.score_dict:
                current_score = self.method_dict[method](outputs, targets)
                self.score_dict[method].append(round(current_score.item(), 4))
            # One time creation of the score key
            else:
                metric = getattr(self.metrics, method)
                self.method_dict[method] = metric(num_classes=self.n_class, **kwargs).to(self.device)
                current_score = self.method_dict[method](outputs, targets)
                self.score_dict[method] = [round(current_score.item(), 4)]
        return outputs, targets, preds

    def log_conf(self, targets, preds):
        m = ConfusionMatrix(num_classes=self.n_class).to(self.device)
        conf_mtx = m(preds, targets).cpu()
        torch.save(conf_mtx, Path.cwd() / 'conf_mtx.pt')

        print('Confusion matrix:')
        print(conf_mtx.numpy()[:16, :16])

    def print_scores(self):
        acc = self.score_dict[self.val_name][-1]
        loss = self.score_dict['loss'][-1]
        print('\nTest/Valid Loss: {:.4f} \t Accuracy: {} %'.format(loss, acc))
        print('------------------------------------------------------------')

