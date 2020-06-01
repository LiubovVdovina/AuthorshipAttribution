import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin

def average_word_length(text):
    avg_word_len = np.mean([len(w) for w in str(text).split()])
    return avg_word_len*100

def typeTokenRatio(text):
    # print(text)
    # print(len(set(text.split())))
    # print(len(text.split()))
    unique_words = len(set(text.split()))*100/len(text.split())
    print("text")
    return unique_words

def hapaxUniqueRatio(text):
    wordcounts = {}
    unique_words = set(text.split())
    for word in unique_words: wordcounts[word] = text.count(word)
    hapaxLegomena_counter = 0
    for word in wordcounts:
        if wordcounts[word]==1:
            hapaxLegomena_counter = hapaxLegomena_counter +1
    res = hapaxLegomena_counter/len(unique_words)*400
    res  = round(res,5)
    print(res)
    return res

class LexicalFeaturesTransformer(BaseEstimator,TransformerMixin):

    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    #Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None
        # X.loc[:,'avg_w_len'] = X.loc[:,'no_punct_num_lowercase'].apply(average_word_length)
        # X.loc[:,'tt_ratio'] = X.loc[:,'no_punct_num_lowercase'].apply(typeTokenRatio)
        X.loc[:,'hv_ratio'] = X.loc[:,'no_punct_num_lowercase'].apply(hapaxUniqueRatio)
        X = X.drop('no_punct_num_lowercase', axis = 1)
        #Converting any infinity values in the dataset to Nan
        X = X.replace([ np.inf, -np.inf ], np.nan)
        #returns a numpy array
        return X.loc[:,['hv_ratio']].values