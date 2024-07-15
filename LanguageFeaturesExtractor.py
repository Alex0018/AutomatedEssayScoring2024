import pandas as pd
import re
import language_tool_python
from tqdm import tqdm


class LanguageFeaturesExtractor():

    COLUMNS = ['ruleId', 'message', 'replacements', 'offsetInContext', 'context', 'offset', 'errorLength', 'category', 'ruleIssueType', 'sentence']
    ERROR_CATEGORIES = ['AMERICAN_ENGLISH_STYLE',   'CASING',    'COLLOCATIONS',           'COMPOUNDING',          'CONFUSED_WORDS',
                        'GRAMMAR',                  'MISC',      'MULTITOKEN_SPELLING',    'NONSTANDARD_PHRASES',  'PUNCTUATION',
                        'REDUNDANCY',               'STYLE',     'REPETITIONS_STYLE',      'TYPOGRAPHY',           'TYPOS']


    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')



    def run(self, text):

        tmp = LanguageFeaturesExtractor.preprocess_text(text)

        matches = self.lang_tool.check(tmp)
        df_errors = pd.DataFrame(matches, columns=LanguageFeaturesExtractor.COLUMNS)

        return df_errors
    
    
    def process_data(self, df):
        df_error_count = pd.DataFrame(0, index=df.index, columns=LanguageFeaturesExtractor.ERROR_CATEGORIES)
        df_error_count['id'] = df['essay_id']

        for index, row in tqdm(df.iterrows(), desc='Counting errors', total=df.shape[0]):
            df_errors = self.run(row['full_text'])
            error_count = df_errors.groupby('category')['ruleId'].count().to_dict()
            for key, val in error_count.items():
                if not key in LanguageFeaturesExtractor.ERROR_CATEGORIES:
                    key = 'MISC'
                df_error_count.loc[index, key] += val
        
        return df_error_count


    @staticmethod
    def preprocess_text(text):
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
        text = text.strip()  # Remove leading and trailing whitespace

        # Preserve paragraphs
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]  # Remove empty paragraphs

        return '\n\n'.join(paragraphs)  # Join paragraphs with double newlines


 