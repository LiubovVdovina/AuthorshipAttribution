# import gensim
# import numpy as np
# import logging
# from normalizer import normalize, get_only_tokens
# import nltk
# from gensim.models import Word2Vec
# wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
# wv.init_sims(replace=True)
#
# def word_averaging(wv, words):
#     all_words, mean = set(), []
#
#     for word in words:
#         if isinstance(word, np.ndarray):
#             mean.append(word)
#         elif word in wv.vocab:
#             mean.append(wv.syn0norm[wv.vocab[word].index])
#             all_words.add(wv.vocab[word].index)
#
#     if not mean:
#         logging.warning("cannot compute similarity with no input %s", words)
#         # FIXME: remove these examples in pre-processing
#         return np.zeros(wv.vector_size,)
#
#     mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
#     return mean
#
# def  word_averaging_list(wv, text_list):
#     return np.vstack([word_averaging(wv, post) for post in text_list ])
#
# def w2v_tokenize_text(text):
#     tokens = []
#     text = get_only_tokens(normalize(text))
#     return tokens
#
# vector = wv['computer']
# data=wv.most_similar('science')
# print(data)

from sklearn import preprocessing
from tools.loader import getData

df = getData(force_reload=False)

X = df['all_normal_tokens_as_string']

print(type(X), X.shape)
#
# # define labels set; transform non-numerical labels to numerical labels
labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder \
    .fit(df['author'].unique()) \
    .transform(df['author'].values)

from gensim.models.phrases import Phrases, Phraser


sent = [row.split() for row in df['all_normal_tokens_as_string']]

phrases = Phrases(sent, min_count=30, progress_per=10000)

sentences = bigram[sent]

bigram = Phrases(sentence_stream)

