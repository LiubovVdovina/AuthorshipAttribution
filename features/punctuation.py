# extracts given columns from df
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import six

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_names]

# counts the frequency of ! among all the chars
def count_punctuationMark(text, mark):
    counter = 0
    for char in text:
        if char == mark:
            counter += 1
    return counter / len(text)

# Transforming column of text data into frequencies of сhars

class PunctuationTransformer(BaseEstimator,TransformerMixin):

    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    #Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None
        X.loc[:,'exclamations'] = X.loc[:,'text'].apply(count_punctuationMark, mark='!')
        X.loc[:, 'questions'] = X.loc[:,'text'].apply(count_punctuationMark, mark='?')
        X.loc[:, 'commas'] = X.loc[:,'text'].apply(count_punctuationMark, mark=',')
        X.loc[:, 'periods'] = X.loc[:,'text'].apply(count_punctuationMark, mark='.')
        X.loc[:, 'column'] = X.loc[:,'text'].apply(count_punctuationMark, mark=':')
        X.loc[:, 'semi-column'] = X.loc[:,'text'].apply(count_punctuationMark, mark=';')
        X = X.drop('text', axis = 1)
        #Converting any infinity values in the dataset to Nan
        X = X.replace([ np.inf, -np.inf ], np.nan)
        #returns a numpy array
        return X.loc[:,['exclamations', 'questions', 'commas', 'periods', 'column', 'semi-column']].values


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(six.itervalues(word2vec)))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(six.itervalues(word2vec)))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])