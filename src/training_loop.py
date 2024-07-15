import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb
from src.styles import TXT_ACC, TXT_RESET
import gc
import os

from sklearn.metrics import cohen_kappa_score

from src.ES_model import EssayClassifierModel, count_trainable_parameters

import torch.nn.functional as F


def collate_batch(inputs):
    """
    It truncates the inputs to the maximum sequence length in the batch.
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max()) # Get batch's max sequence length
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


def train_epoch(dataloader, model, optimizer, loss_criterion, device):
    
    model.train()
    loss = 0.0
    kappa = 0.0

    preds = np.empty(len(dataloader), dtype=int)
    targets = np.empty(len(dataloader), dtype=int)

    for b, batch in enumerate(pbar := tqdm(dataloader)):
        data = collate_batch(batch['inputs']).to(device)
        target = batch['labels'].to(device)
        
        output = model(data)
        cur_loss = loss_criterion(output, target)
        loss += cur_loss.item()

        preds[b] = output.detach().cpu().numpy().argmax(axis=1)
        targets[b] = target.cpu().detach().numpy()

        if b % 30 == 29:
            kappa = cohen_kappa_score(targets[b-29:b+1], preds[b-29:b+1], weights='quadratic')

        cur_loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        pbar.set_description(f'  training batch:    mean loss {loss / (b+1): .5f}     kappa {kappa: .5f}')
        torch.cuda.empty_cache()

    return loss / len(dataloader), cohen_kappa_score(targets, preds, weights='quadratic')


def validate_epoch(dataloader, model, loss_criterion, device):
    model.eval()
    loss = 0.0
    kappa = 0.

    preds = np.empty(len(dataloader), dtype=int)
    targets = np.empty(len(dataloader), dtype=int)

    with torch.no_grad():
        for b, batch in enumerate(pbar := tqdm(dataloader)):
            data = collate_batch(batch['inputs']).to(device)
            target = batch['labels'].to(device)
            
            output = model(data)
            cur_loss = loss_criterion(output, target)
            loss += cur_loss

            preds[b] = output.detach().cpu().numpy().argmax(axis=1)
            targets[b] = target.cpu().detach().numpy()

            if b % 30 == 29:
                kappa = cohen_kappa_score(targets[b-29:b+1], preds[b-29:b+1], weights='quadratic')

            pbar.set_description(f'validation batch:    mean loss {loss / (b+1): .5f}     kappa {kappa: .5f}')
            torch.cuda.empty_cache()

    return loss / len(dataloader), cohen_kappa_score(targets, preds, weights='quadratic')
 

DIR_SAVE = 'trained_models'

def log_training(wandb_run, epoch, model, optimizer, loss_train, loss_valid, metric_train, metric_valid, log_name):
    name = f'{wandb_run.id}_{log_name}'

    if not os.path.exists(DIR_SAVE):
        os.mkdir(DIR_SAVE)

    path = f'{DIR_SAVE}/{name}.pth'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': loss_train, 
                'loss_valid': loss_valid,                         
                'metric_train': metric_train,
                'metric_valid': metric_valid}, 
            path)
                      
    artifact = wandb.Artifact(name=name, type=log_name)
    artifact.add_file(path)            
    wandb_run.log_artifact(artifact)


def training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs, 
                  start_epoch,
                  device,
                  wandb_run,
                  prev_best_metric = -np.inf,
                  post_proc=None):
    
    losses_train = np.zeros(num_epochs)
    losses_valid = np.zeros(num_epochs)
    
    best_metric = prev_best_metric
    best_epoch = start_epoch - 1

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    print('', '-'*100, '\n', '  Number of trainable parameters in model: ', count_trainable_parameters(model), '\n', '-'*100)

    params_unfreeze = False
    for epoch in range(num_epochs):  

        if not params_unfreeze and (epoch + start_epoch) > 1 and isinstance(model, EssayClassifierModel):
            params_unfreeze = True
            model.unfreeze_feature_exctractor_weights()
            print('', '-'*100, '\n', '  Number of trainable parameters in model: ', count_trainable_parameters(model), '\n', '-'*100)
        
        print(f'{TXT_ACC} Epoch {start_epoch+epoch} {TXT_RESET}')
        
        loss_train, metric_train = train_epoch(dataloader_train, model, optimizer, loss_criterion, device)
        loss_valid, metric_valid = validate_epoch(dataloader_valid, model, loss_criterion, device)
        
        scheduler.step()
        # print new lr!

        losses_train[epoch] = loss_train
        losses_valid[epoch] = loss_valid        
            
        print(f'\ttrain:       loss {loss_train: .5f}      metric {metric_train: .5f}')        
        print(f'\tvalidation:  loss {loss_valid: .5f}      metric {metric_valid: .5f}')

        if wandb_run is not None:
            wandb_run.log({'loss_train':   loss_train, 
                           'loss_valid':   loss_valid,
                           'metric_train': metric_train,
                           'metric_valid': metric_valid}) 
        
        if metric_valid > (best_metric + 0.0001):
            best_epoch = epoch
            best_metric = metric_valid
            
            if wandb_run is not None:
                log_training(wandb_run, epoch+start_epoch, model, optimizer, loss_train, loss_valid, metric_train, metric_valid, 'best_model')
            
    if wandb_run is not None:       
        log_training(wandb_run, epoch+start_epoch, model, optimizer, loss_train, loss_valid,  metric_train, metric_valid, 'checkpoint')

    return losses_train, losses_valid
