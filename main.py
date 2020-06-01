from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import coo_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
from normalizer import normalize, get_only_tokens, lemmatize

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from gensim.test.utils import datapath, get_tmpfile
from tools.loader import getData
from features.punctuation import ColumnSelector, MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer, count_punctuationMark, PunctuationTransformer
from lexical_features import LexicalFeaturesTransformer, average_word_length
from character_features import CharacterFeaturesTransformer, character_pipeline, count_character

blogsDataFrame = getData(force_reload=False)
#     print(blogsDataFrame)
# define data set
# X = blogsDataFrame['normal_tokens_as_string']

# print(type(X), X.shape)
#
# # define labels set; transform non-numerical labels to numerical labels
labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder \
    .fit(blogsDataFrame['author'].unique()) \
    .transform(blogsDataFrame['author'].values)

# training model
# glove_file = '/Users/lyuba/PycharmProjects/SVM/glove.6B.100d.txt'
# word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
# glove2word2vec(glove_file, word2vec_glove_file)
#
# model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
#
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))

# pipeline = Pipeline([('extract_essays', ColumnSelector('normal_tokens')),("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), ('classifier', svm.SVC(kernel='linear'))])
# scores_pipe = cross_val_score(pipeline, blogsDataFrame, y, scoring='accuracy', cv=10)
# mean_pipe_score = scores_pipe.mean()
# print("Accuracy for pipeline:", mean_pipe_score)

columns = ['no_punct_num_lowercase']
# letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
print(blogsDataFrame.shape)
pipe = Pipeline([('extract_essays', ColumnSelector(columns)),('character', LexicalFeaturesTransformer()), ('classifier', svm.SVC(kernel='linear'))])
# pipe = Pipeline([('extract_essays', ColumnSelector(columns)),('character', CountVectorizer(analyzer='char',vocabulary='a',lowercase=True)), ('classifier', svm.SVC(kernel='linear'))])
scores_pipe = cross_val_score(pipe, blogsDataFrame, y, scoring='accuracy', cv=10)
mean_pipe_score = scores_pipe.mean()
print("Accuracy for TT_RATIO:", mean_pipe_score)

# scores_pipe = cross_val_score(character_pipeline, blogsDataFrame, y, scoring='accuracy', cv=10)
# mean_pipe_score = scores_pipe.mean()
# print("Accuracy for character features:", mean_pipe_score)

# blogsDataFrame['average_word_length'] = blogsDataFrame['no_punct_num_lowercase'].apply(average_word_length)
# blogsDataFrame['len'] = blogsDataFrame['text'].apply(len)
# letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# for letter in letters:
#     blogsDataFrame[letter] = blogsDataFrame['text'].apply(count_character,character=letter)
#
# X = blogsDataFrame.loc[:,['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']]
# print(X,X.shape)
# print(y.shape)
# clf = svm.SVC(kernel='linear')
#
# print(cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())