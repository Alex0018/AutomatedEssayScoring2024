import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import CFG


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # print('input hidden state shape', last_hidden_state.shape)
        # print('input attention mask shape', attention_mask.shape)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # print('expanded attention mask shape', input_mask_expanded.shape)

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # print('sum embedding shape', sum_embeddings.shape)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EssayClassifierModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.feature_extractor = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.feature_extractor.config.hidden_size, CFG.num_labels)
        self.pool = MeanPooling()

        self.freeze_feature_exctractor_weights()
        

    def _get_features(self, inputs):
        # print(inputs)

        outputs = self.feature_extractor(**inputs)
        last_hidden_states = outputs[0]
        embeddings = self.pool(last_hidden_states, inputs['attention_mask'])
        return embeddings
    

    def freeze_feature_exctractor_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_exctractor_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


    def forward(self, inputs):
        features = self._get_features(inputs)
        output = self.fc(features)
        return output