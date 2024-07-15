import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from src.styles import *



class DataExplorer:
    
    def __init__(self, path_train, path_test, target_feature, data_preproc_pipeline=None):
        self.path_train = path_train
        self.path_test = path_test
        self.target_feature = target_feature
        self.data_preproc_pipeline = data_preproc_pipeline
        
    def decorate(title):     
        def dec(func):     
            def inner(self):    
                print('-' * 100, '\n',' '*20, title, '\n', '-' * 100, '\n')
                func(self)  
                print('\n\n\n\n')                
            return inner
        return dec
    
    def data_at_a_glance(self):        
        self.read_data()
        self.fill_nans_dummy()
        
        self.display_data()
        self.display_duplicates()
        self.display_target()
        self.display_features_info()
        self.display_categorical_features_info()
        self.display_numerical_features_info()
        
        self.encode_categorical_features()
        
        self.display_correlation_matrix()
        self.display_correlations_to_target()        
        
        
        
        
    def read_data(self):        
        self.df_train = pd.read_csv(self.path_train).drop('id', axis=1)
        self.df_test = pd.read_csv(self.path_test).drop('id', axis=1)
        
        if self.data_preproc_pipeline is not None:
            self.df_train = self.data_preproc_pipeline.fit_transform(self.df_train)
            self.df_test = self.data_preproc_pipeline.transform(self.df_test)
        
        self.target_is_categorical = self.df_train[self.target_feature].nunique() < 5
        
        self.features_categorical = list(set(self.df_train.select_dtypes('object').columns) - set([self.target_feature]))
        self.features_numerical = list(set(self.df_train.select_dtypes(['int', 'float']).columns)
                                  - set(['id', self.target_feature]))

        self.numerical_categorical_features = [feature for feature in self.features_numerical if self.df_train[feature].nunique() < 10]
        self.features_categorical += self.numerical_categorical_features
        self.features_numerical = list(set(self.features_numerical) - set(self.numerical_categorical_features))
        
        
        
    def get_data(self):
        self.read_data()
        self.fill_nans_dummy()
        self.encode_categorical_features()
        
        return self.df_train, self.df_test
        
        
    def fill_nans_dummy(self):
        self.df_train[self.features_categorical] = self.df_train[self.features_categorical].fillna('N/A')
        self.df_test[self.features_categorical] = self.df_test[self.features_categorical].fillna('N/A')
        
        self.df_train[self.features_numerical] = self.df_train[self.features_numerical].fillna(self.df_train[self.features_numerical].min() - 100)
        self.df_test[self.features_numerical] = self.df_test[self.features_numerical].fillna(self.df_train[self.features_numerical].min() - 100)
    
    
    def encode_categorical_features(self):
        if self.target_is_categorical:
            oe = OrdinalEncoder()
            self.df_train[self.target_feature] = oe.fit_transform(self.df_train[self.target_feature].values.reshape(-1,1))

        features_binary = [feature for feature in self.features_categorical if self.df_train[feature].nunique() == 2]
#         print('Binary features:\n', features_binary)
        for feature in features_binary:
            top_cat = self.df_train[feature].mode()[0]
            self.df_train[feature] = (self.df_train[feature] == top_cat).astype(int)
            self.df_test[feature] = (self.df_test[feature] == top_cat).astype(int)


        features_encode = list(set(self.features_categorical) - set(features_binary))
#         print('Categorical features to encode:\n', features_encode)
        ohe = OneHotEncoder()
        ohe.fit(pd.concat([self.df_train[features_encode], self.df_test[features_encode]], axis=0))
        df_ohe_train = pd.DataFrame(ohe.transform(self.df_train[features_encode]).toarray(), columns=ohe.get_feature_names_out())
        df_ohe_test = pd.DataFrame(ohe.transform(self.df_train[features_encode]).toarray(), columns=ohe.get_feature_names_out())

        oe = OrdinalEncoder()
        oe.fit(pd.concat([self.df_train[features_encode], self.df_test[features_encode]], axis=0))
        self.df_train[features_encode] = oe.transform(self.df_train[features_encode])
        self.df_test[features_encode] = oe.transform(self.df_test[features_encode])
    
    
    @decorate(title='DATA')
    def display_data(self):
        print(f'{TXT_ACC} TRAIN {TXT_RESET}    {self.df_train.shape[0]} rows, {self.df_train.shape[1]} columns')
        display(self.df_train)
        print(f'\n\n{TXT_ACC} TEST {TXT_RESET}    {self.df_test.shape[0]} rows, {self.df_test.shape[1]} columns')
        display(self.df_test)
        
        
    @decorate(title='DUPLICATES')
    def display_duplicates(self):
        print(f'{TXT_ACC} TRAIN {TXT_RESET}')
        print('{0} duplicated rows out of {1} ({2}%)'.format(self.df_train.duplicated(keep=False).sum(), 
                                                             self.df_train.shape[0],
                                                             self.df_train.duplicated(keep=False).sum() / self.df_train.shape[0]))
        print(f'\n\n{TXT_ACC} TEST {TXT_RESET}')
        print('{0} duplicated rows out of {1} ({2}%)'.format(self.df_test.duplicated(keep=False).sum(), 
                                                             self.df_test.shape[0],
                                                             self.df_test.duplicated(keep=False).sum() / self.df_test.shape[0]))
        
    @decorate (title='TARGET')    
    def display_target(self):        
        if self.target_is_categorical:
            display(self.df_train[self.target_feature].value_counts() \
                    .to_frame() \
                    .style.bar(color=PALETTE[3], width=90, height=50)
                   )
        else:
            _,ax = plt.subplots(1,1, figsize=(5, 3))
            sns.kdeplot(self.df_train[self.target_feature], ax=ax, fill=True)
            plt.show()

        
    @decorate(title='FEATURES')
    def display_features_info(self): 

        print('Numerical features with few values: ', self.numerical_categorical_features)

        count_nan_func = lambda x: x.isna().sum()

        print(f'\n\n{TXT_ACC} Categorical features {TXT_RESET}')
        if len(self.features_categorical) > 0:
            display(self.df_train[self.features_categorical].agg([pd.Series.nunique, count_nan_func]).T \
                    .rename(columns={'<lambda>': 'count NaN'})
                    .astype({'nunique': 'int'}) \
                    .sort_values('nunique'))
        else:
            print('no categorical features')

        print(f'\n\n{TXT_ACC} Numerical features {TXT_RESET}')
        display(self.df_train[self.features_numerical].agg([pd.Series.nunique, count_nan_func, 'mean', 'min', 'max', 'std']).T \
                .rename(columns={'<lambda>': 'count NaN'})
                .astype({'nunique': 'int', 'count NaN': 'int'}) \
                .sort_values('nunique'))
        
        
    @decorate(title='CATEGORICAL FEATURES')
    def display_categorical_features_info(self): 
        
        if len(self.features_categorical) == 0:
            print('No categorical features')
            pass
        
        for feature in self.features_categorical:
            df1 = pd.concat([self.df_train[feature].value_counts().rename('train'), 
                             self.df_test[feature].value_counts().rename('test')],
                            axis=1) \
                        .fillna(0).astype(int)

            if not self.target_is_categorical:
                display(df1.style.bar(color=PALETTE[0], height=50, width=90))
                
                # add kde of features for each category
            else:

                top_target_category = self.df_train[self.target_feature].mode()[0]

                df2 = self.df_train[[feature, self.target_feature]] \
                            .value_counts() \
                            .rename('target') \
                            .to_frame() \
                            .unstack() 
                
                df2.columns = ['_'.join(list(map(str, col))) for col in df2.columns]
                
                df = pd.concat([df1, df2], axis=1) \
                            .fillna(0) \
                            .astype(int) \
                            .sort_values('target_' + str(top_target_category), ascending=False)

                display(df.style.bar(df1.columns, color=PALETTE[0], height=50, width=90) \
                       .bar(df2.columns, color=PALETTE[2], height=50, width=90)
                       )

            print()
            
            
            
    @decorate(title='NUMERICAL FEATURES')
    def display_numerical_features_info(self):

        _, axes = plt.subplots(ncols=2, nrows=len(self.features_numerical), figsize=(6, 2*len(self.features_numerical)))

        for i, feature in enumerate(self.features_numerical):
            sns.kdeplot(self.df_train[feature], ax=axes[i, 0], fill=True, label='train')
            sns.kdeplot(self.df_test[feature], ax=axes[i, 0], fill=True, label='test')
            axes[i, 0].set_title(f'{feature}', fontsize=8)
            axes[i, 0].legend()   
        
        # relationship with target        
        if self.target_is_categorical:
            for i, feature in enumerate(self.features_numerical):
                sns.boxplot(y=self.df_train[feature], x=self.df_train[self.target_feature], ax=axes[i, 1])
                axes[i, 1].set_title(f'{feature}', fontsize=8)
        
        else:
            for i, feature in enumerate(self.features_numerical):
                sns.scatterplot(x=self.df_train[feature], y=self.df_train[self.target_feature], ax=axes[i, 1])
                axes[i, 1].set_title(f'{feature}', fontsize=8)
            pass
    
        plt.tight_layout()
        plt.show() 
        
        
    @decorate(title='CORRELATION MATRIX')
    def display_correlation_matrix(self):
        X_train = self.df_train.drop([self.target_feature], axis=1)

        cormatrix = X_train.corr()
        mask = np.eye(len(cormatrix)) # mask out identity correlations    

        plt.figure(figsize=(cormatrix.shape[1] * 1.1, cormatrix.shape[1] * 0.4))

        heatmap = sns.heatmap(cormatrix, mask=mask, vmin=-1, vmax=1, annot_kws={'alpha': 0.5},
                              annot=True, cmap=sns.color_palette(PALETTE_HEATMAP), cbar=False)
        heatmap.set_title('Feature correlations');

        for label in heatmap.get_xticklabels():
            label.set_rotation(90)
        for label in heatmap.get_yticklabels():
            label.set_rotation(0)
            
        plt.tight_layout()
        plt.show()
        
        
    @decorate(title='CORRELATIONS TO TARGET')
    def display_correlations_to_target(self):
        X_train = self.df_train.drop([self.target_feature], axis=1)    
        y_train = self.df_train[self.target_feature]
        
        corrs = [(feature, abs(X_train[feature].corr(y_train))) for feature in X_train.columns]
        df_corrs = pd.DataFrame(corrs)
        df_corrs.index = df_corrs[0]
        df_corrs = df_corrs.drop(0, axis=1)
        df_corrs.columns = ['corr']

        display(df_corrs.sort_values('corr', ascending=False))