import re
import polars as pl
import pandas as pd
import numpy as np
import string
import spacy

'''
https://www.kaggle.com/code/yongsukprasertsuk/0-818-deberta-v3-large-lgbm-baseline/notebook
'''


class TextFeaturesExtractor:

    PATH_VOCAB = 'data/words.txt'

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        with open(TextFeaturesExtractor.PATH_VOCAB, 'r') as file:
            self.english_vocab = set(word.strip().lower() for word in file)


    def process_data(self, path_data):
        transform_columns = [(
                pl.col("full_text").str.split(by="\n\n").alias("paragraph")
            ),]
        df = pl.read_csv(path_data).with_columns(transform_columns)

        tmp = self.Paragraph_Preprocess(df)
        paragraph_features = self.get_paragraph_features(tmp)
        paragraph_features['score'] = df['score']

        tmp = self.Sentence_Preprocess(df)
        sentence_features = self.get_sentence_features(tmp)

        word_features = self.get_word_features(df)

        res = paragraph_features.merge(sentence_features, on='essay_id') \
                                .merge(word_features, on='essay_id')
        
        return res


    def get_sentence_features(self, df):
        sentence_features = ['sentence_len','sentence_word_cnt']
        aggs = [
            # Count the number of sentences with a length greater than i
            *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [0,15,50,100,150,200,250,300] ], 
            *[pl.col('sentence').filter(pl.col('sentence_len') <= i).count().alias(f"sentence_<{i}_cnt") for i in [15,50] ], 
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_features],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_features],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_features],
            *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in sentence_features],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_features],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_features],
            *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in sentence_features],
            *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in sentence_features], 
            *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in sentence_features], 
            ]
        df_res = df.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df_res = df_res.to_pandas()
        return df_res


    def get_word_features(self, df):
        tmp = df.to_pandas()

        nn = 15
        word_count_features = [f'word_{i+1}_cnt' for i in range(nn)]
        word_len_features = ['word_len_max', 'word_len_mean', 'word_len_std', 'word_len_q1', 'word_len_q2', 'word_len_q3']
        
        df_res = pd.DataFrame(0, index=tmp.index, columns=[*word_count_features, *word_len_features])
        df_res['esay_id'] = tmp['essay_id']

        for index, row in tmp.iterrows():
            words = row['full_text'].split(' ')
            word_lens = np.array([len(w) for w in words])

            for i in range(nn):
                df_res.loc[index, f'word_{i+1}_cnt'] = np.sum(word_lens >= i+1)

            df_res.loc[index, 'word_len_max'] = np.max(word_lens)
            df_res.loc[index, 'word_len_mean'] = np.mean(word_lens)
            df_res.loc[index, 'word_len_std'] = np.std(word_lens)
            df_res.loc[index, 'word_len_q1'] = np.quantile(word_lens, 0.25)
            df_res.loc[index, 'word_len_q2'] = np.quantile(word_lens, 0.5)
            df_res.loc[index, 'word_len_q3'] = np.quantile(word_lens, 0.75)

        return df_res

    

    def get_paragraph_features(self, df):
        paragraph_features = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt', 'paragraph_error_num']
        aggs = [
            # Count the number of paragraph lengths greater than and less than the i-value
            *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700] ], 
            *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in [25,49]], 
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_features],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_features],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_features],
            *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in paragraph_features],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_features],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_features],
            *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in paragraph_features],
            *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in paragraph_features],  
            *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in paragraph_features],  
            ]
        
        df_res = df.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        return df_res.to_pandas()



    def Word_Preprocess(self, tmp):
        tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=" ").alias("word"))
        tmp = tmp.drop('full_text').explode('word')
        tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
        tmp = tmp.filter(pl.col('word_len')!=0)        
        return pl.DataFrame(tmp)



    def Paragraph_Preprocess(self, tmp):
        # Expand the paragraph list into several lines of data
        tmp = tmp.explode('paragraph')
        # Paragraph preprocessing
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(self.dataPreprocessing))
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(self.remove_punctuation).alias('paragraph_no_pinctuation'))
        tmp = tmp.with_columns(pl.col('paragraph_no_pinctuation').map_elements(self.count_spelling_errors).alias("paragraph_error_num"))
        # Calculate the length of each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len"))
        # Calculate the number of sentences and words in each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias("paragraph_sentence_cnt"),
                        pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias("paragraph_word_cnt"),)
        return tmp



    def Sentence_Preprocess(self, tmp):
        tmp = tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=".").alias("sentence"))
        tmp = tmp.explode('sentence')
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))        
        tmp = tmp.filter(pl.col('sentence_len')>=15) # Filter out the portion of data with a sentence length greater than 15
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))        
        return tmp



    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    

    def count_spelling_errors(self, text):
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_.lower() for token in doc]
        spelling_errors = sum(1 for token in lemmatized_tokens if token not in self.english_vocab)
        return spelling_errors


    def dataPreprocessing(self, x):
        x = x.lower()
        x = self.removeHTML(x)        
        x = re.sub("@\w+", '',x) # Delete strings starting with @        
        x = re.sub("'\d+", '',x) # Delete Numbers
        x = re.sub("\d+", '',x)  # Delete Numbers        
        x = re.sub("http\w+", '',x) # Delete URL        
        x = re.sub(r"\s+", " ", x) # Replace consecutive empty spaces with a single space character        
        x = re.sub(r"\.+", ".", x) # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\,+", ",", x)        
        x = x.strip() # Remove empty characters at the beginning and end
        return x
    


    def removeHTML(self, x):
        html=re.compile(r'<.*?>')
        return html.sub(r'', x)