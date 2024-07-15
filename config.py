import pandas as pd
from sklearn.model_selection import StratifiedKFold



class ProjectPaths:
    train = 'data/train.csv'
    train_cluster3 = 'data/train_cluster3.csv'
    test = 'data/test.csv'
    sample_submission = 'data/sample_submission.csv'

    

class CFG:    
    project_name = 'EssayScoring'
    model_name = 'microsoft/deberta-v3-xsmall' # 'bert-base-uncased'
    n_splits = 5
    seed = 42
    max_length = 1024
    num_labels = 6


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def read_train():

    df_train = pd.read_csv(ProjectPaths.train)
    df_train['label'] = (df_train['score'] - 1).astype('int32') 

    cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    for i, (_, val_index) in enumerate(cv.split(df_train, df_train['label'])):
        df_train.loc[val_index, 'fold'] = i
        
    return df_train