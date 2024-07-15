import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import gc


from src.ES_model import EssayClassifierModel
from src.ES_dataset import create_dataloaders
from src.training_loop import training_loop


PROJECT_NAME = 'EssayScoring_LLM_features'

dataloaders_dict = {'EssayClassifier': create_dataloaders,}
model_dict = {'EssayClassifier': EssayClassifierModel,}

loss_dict = {'CrossEntropy': nn.CrossEntropyLoss,
             'MSE': nn.MSELoss }

optim_dict = {'Adam': optim.Adam,
              'AdamW': optim.AdamW}





def experiment(config):    

    if config['project_name'] is not None:
        run = wandb.init(project=config['project_name'], config=config, name=config['run_name'])
    else:
        run = None
    
    create_dataloaders = dataloaders_dict.get(config['create_dataloaders_func'])    
    dataloader_train, dataloader_valid = create_dataloaders(**config['data_parameters'])        

    model_class = model_dict.get(config['model'])
    model = model_class(**config['model_parameters']).to(device=config['device'], dtype=torch.float32)
    # print('', '-'*100, '\n', '  Number of trainable parameters in model: ', count_trainable_parameters(model), '\n', '-'*100)
    
    loss_criterion = loss_dict.get(config['loss'])()
    optimizer = optim_dict.get(config['optimizer'])(model.parameters(), lr=config['learning_rate'])

    
    training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs=config['epochs'], 
                  start_epoch=0,
                  device=config['device'],
                  wandb_run=run)

    wandb.finish()

    del run, dataloader_train, dataloader_valid, model, loss_criterion, optimizer
    gc.collect()

    torch.cuda.empty_cache()
    





def continue_experiment(run_id, checkpoint_name='checkpoint', version='latest', num_epochs=None, learning_rate=None):
    run = wandb.init(project=PROJECT_NAME, id=run_id, resume='allow')
    config = run.config
    
    create_dataloaders = dataloaders_dict.get(config['create_dataloaders_func'])    
    dataloader_train, dataloader_valid = create_dataloaders(**config['data_parameters'])        

    model_class = model_dict.get(config['model'])
    model = model_class(**config['model_parameters']).to(device=config['device'], dtype=torch.float32)
    
    loss_criterion = loss_dict.get(config['loss'])()
    optimizer = optim_dict.get(config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    
    last_epoch = 0    

    if wandb.run.resumed:
        
        checkpoint_name = f'{run.id}_{checkpoint_name}'
        artifact = run.use_artifact(checkpoint_name + f':{version}')
        entry = artifact.get_path(checkpoint_name + '.pth')
        
        file = entry.download()
        
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        
        if learning_rate is not None:
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        print(f'\nResuming training after epoch {last_epoch}\n')
        print(f'Pevious best mtric:  train {checkpoint["metric_train"]:.5f}')
        print(f'                     valid {checkpoint["metric_valid"]:.5f}\n')


    
    training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs=config['epochs'] if num_epochs is None else num_epochs, 
                  start_epoch=last_epoch+1,
                  device=config['device'],
                  wandb_run=run,
                  prev_best_metric=checkpoint["loss_valid"], 
                  post_proc=None)

    run.finish()
    wandb.finish()

