
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
tokenizer = RegexpTokenizer("[\w']+")

# #actually without normailzation
# def normalize(text):
#     tokens = [token for token in tokenizer.tokenize(text.lower())]
#     tokens_parsed = nltk.pos_tag(tokens)
#     return [(str(t[0]), str(t[1])) for t in tokens_parsed]

def normalize(text):
    tokens = [token for token in tokenizer.tokenize(text.lower())]
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for token in tokens:
        if token not in stopwords.words('english') and token.isalpha():
            finalWord = word_Lemmatized.lemmatize(token)
            Final_words.append(finalWord)
    tokens_parsed = nltk.pos_tag(Final_words)
    return [(str(t[0]), str(t[1])) for t in tokens_parsed]

def not_normalize(text):
    tokens = [token for token in text.split()]
    return [t for t in tokens]

# print(letterSplit("Hi. My name is Elena, I'm here\n Bye"))