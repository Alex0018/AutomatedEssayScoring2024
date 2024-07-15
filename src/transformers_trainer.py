import wandb

from transformers import  TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from tokenizers import AddedToken
from datasets import Dataset

from sklearn.metrics import cohen_kappa_score
from src.config import seed_everything



class TransformersClassifier:

    METRIC_NAME = 'qwk'
    PROJECT_NAME = 'EssayScoring'

    def __init__(self, model_name, max_tokenizer_len=512, seed=42):
        self.max_tokenizer_len = max_tokenizer_len
        self.model_name = model_name
        self.num_labels = 6

        # ADD NEW TOKENS for ("\n") new paragraph and (" "*2) double space 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        self.tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

        seed_everything(seed)



    def train(self, df_train, df_valid, training_args_dict, name):

        wandb.init(project=self.PROJECT_NAME, name=name)

        tokenized_train = self.create_tokenized_dataset(df_train)
        tokenized_valid = self.create_tokenized_dataset(df_valid)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        model.resize_token_embeddings(len(self.tokenizer))

        training_args = TrainingArguments(
            **training_args_dict,
            output_dir='results',
            fp16=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            evaluation_strategy='epoch',
            metric_for_best_model=self.METRIC_NAME,
            greater_is_better=True,
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,  
            logging_strategy='epoch',
            optim='adamw_torch',)

        trainer = Trainer( 
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_for_classification,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        wandb.finish()

        trainer.save_model(name)



    def predict(self, model_dir, df, true_labels=None):

        tokenized_ds = self.create_tokenized_dataset(df)

        training_args = TrainingArguments(
            report_to='none',
            output_dir='results',
            per_device_eval_batch_size=2,
            metric_for_best_model=self.METRIC_NAME,
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=self.num_labels)

        trainer = Trainer( 
            model=model,
            args=training_args,
            eval_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
        )

        predictions = trainer.predict(tokenized_ds).predictions

        if true_labels is None:
            return {'predictions': predictions}
        
        score = self.compute_metrics_for_classification([predictions, true_labels])
        return {'predictions': predictions,    self.METRIC_NAME: score}





    def compute_metrics_for_classification(self, eval_data):    
        predictions, true_labels = eval_data
        score = cohen_kappa_score(true_labels, predictions.argmax(-1), weights='quadratic')
        return {self.METRIC_NAME: score}
    


    def _tokenize_function(self, example):
        return self.tokenizer(example['full_text'], truncation=True, max_length=self.max_tokenizer_len)

    def create_tokenized_dataset(self, df):
        features = ['essay_id', 'full_text', 'label']
        if not 'label' in df.columns:
            features.remove('label')

        ds = Dataset.from_pandas(df[features])      
        tokenized_ds = ds.map(self._tokenize_function, batched=True)

        return tokenized_ds
