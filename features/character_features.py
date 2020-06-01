import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from features.punctuation import ColumnSelector
from sklearn import svm

def uppercase_count(text):
    uppercase_counter = 0
    for char in text:
        if char.isupper():
            uppercase_counter += 1
    res = uppercase_counter/len(text)
    return(res)

def digits_count(text):
    digits_counter = 0
    for char in text:
        if char.isdigit():
            digits_counter += 1
    res = digits_counter/len(text)
    return(res)

def alphabetic_count(text):
    characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    charcounts = {}
    for char in characters: charcounts[char] = text.count(char)
    for char in characters: charcounts[char] = charcounts[char]/len(text)
    return list(charcounts.values())

def count_character(text, character):
    counter = 0
    for char in text.lower():
        if char == character:
            counter += 1
    return counter / len(text)*1000

class CharacterFeaturesTransformer(BaseEstimator,TransformerMixin):

    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    #Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None
        # X.loc[:,'uppercase'] = X.loc[:,'text'].apply(uppercase_count)
        # X.loc[:,'digits'] = X.loc[:,'text'].apply(digits_count)
        # X.loc[:,'blanks'] = X.loc[:,'text'].apply(count_character,character=' ')
        characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        for letter in characters:
            X.loc[:,letter] = X.loc[:,'text'].apply(count_character,character=letter)
        X = X.drop('text', axis = 1)
        #Converting any infinity values in the dataset to Nan
        X = X.replace([ np.inf, -np.inf ], np.nan)
        #returns a numpy array
        return X.loc[:,['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']].values

upper_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
special_characters = ['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '|']
digits = ['0','1','2','3','4','5','6','7','8','9']

character_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('uppercase', Pipeline([
            ('extract', ColumnSelector('text')),
            ('uppers', CountVectorizer(analyzer='char',vocabulary=upper_letters,lowercase=False))
        ])),
        ('digits', Pipeline([
            ('extract', ColumnSelector('text')),
            ('digs', CountVectorizer(analyzer='char',vocabulary=digits))
        ])),
        ('blanks', Pipeline([
            ('extract', ColumnSelector('text')),
            ('blank', CountVectorizer(analyzer='char',vocabulary=' '))
        ])),
        ('letters', Pipeline([
            ('extract', ColumnSelector('text')),
            ('alph', CountVectorizer(analyzer='char',vocabulary=letters))
        ])),
        ('special', Pipeline([
            ('extract', ColumnSelector('text')),
            ('spcs', CountVectorizer(analyzer='char',vocabulary=special_characters))
        ]))
    ])),
    ('classifier', svm.SVC(kernel='linear'))
])