import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transformers import  AutoTokenizer, DataCollatorWithPadding
from tokenizers import AddedToken

import torch
from torch.utils.data import DataLoader, Dataset


def collate_batch(inputs):
    max_len = 0
    for data in inputs:
        l = data['inputs']['attention_mask'].sum(axis=1)
    print(inputs['inputs'])
    mask_len = int(inputs['attention_mask'].sum(axis=1).max()) # Get batch's max sequence length
    for k in inputs.keys():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

def create_dataloaders(path_df, fold, model_name, batch_size):

    df = pd.read_csv(path_df)
    df['label'] = (df['score'] - 1).astype('int32') 

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (_, val_index) in enumerate(cv.split(df, df['label'])):
        df.loc[val_index, 'fold'] = i

    dataset_train = EssayDataset(df.query(f'fold!={fold}'), model_name)
    dataset_valid = EssayDataset(df.query(f'fold=={fold}'), model_name)

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    dataloader_valid = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)                                  

    return dataloader_train, dataloader_valid



class EssayDataset(Dataset):
    def __init__(self, df, model_name):

        self.texts = df['full_text'].values
        self.labels = df['label'].values
        self.essay_ids = df['essay_id'].values

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        self.tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])
        
        self.max_len = 1024


    def __len__(self):
        return len(self.texts)
    

    def tokenize_input(self, text):
        '''
        returns dictionary with tensors "input_ids", "token_type_ids" and "attention_mask"
        '''
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length', 
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long) # deberta requires long
        return inputs


    def __getitem__(self, index):
        output = {}
        output["inputs"] = self.tokenize_input(self.texts[index])
        output["labels"] = torch.tensor(self.labels[index], dtype=torch.long) # deberta requires long
        output["essay_ids"] = self.essay_ids[index]
        return output
