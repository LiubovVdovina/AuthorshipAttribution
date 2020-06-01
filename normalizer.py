from nltk.corpus import stopwords
import re
import en_core_web_sm
import pymorphy2

numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
           'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand']
ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                   'eleventh', 'twelfth','thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
                   'eighteenth', 'nineteenth', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth',
                   'seventieth', 'eightieth', 'ninetieth', 'hundredth', 'thousandth', 'etc']

additional_words = ['us','mine']
stopwords = stopwords.words('english') + additional_words

def normalize(text):
    #replace all symbols except for alphabetic and \n with space in order to separate contrations 've, 're
    text = re.sub('[^a-zA-Z\n]', ' ', text)
    nlp = en_core_web_sm.load()
    # requirments spacy
    # make sure to download the english model with "python -m spacy download en"
    tokens = []
    for token in nlp(text.lower()):
        if str(token) not in stopwords and token.lemma_ not in stopwords and str(token).isalpha() and len(token.lemma_)>2:
            w = []
            w.append(token.lemma_)
            w.append(token.pos_)
            tokens.append(w)
    return tokens

morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    #replace all symbols except for alphabetic and \n with space in order to separate contrations 've, 're
    # text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
    nlp = en_core_web_sm.load()
    # requirments spacy
    # make sure to download the english model with "python -m spacy download en"
    tokens = []
    for token in nlp(text.lower()):
        if str(token).isalpha():
            if(token.lemma_ == "-PRON-"):
                tokens.append(morph.parse(str(token))[0].normal_form)
            else:
                tokens.append(token.lemma_)
    return tokens

def get_only_tokens(tokens_parsed=[]):
    return [t[0] for t in tokens_parsed]

def get_only_pos(tokens_parsed=[]):
    return [t[1] for t in tokens_parsed]

def remove_punctuation_numbers_case(text):
    text = re.sub('[^a-zA-Z\n]', ' ', text).lower()
    return re.sub(' +', ' ',text)

def tokenize(text):
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    return tokens

# print(tokenize("Mike s my!!.. 50 husband."))

#https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936
#customize_stop_words = [
# 'computing', 'filtered'
# ]
# for w in customize_stop_words:
#     spacy_nlp.vocab[w].is_stop = True
# doc = spacy_nlp(article)
# tokens = [token.text for token in doc if not token.is_stop]
# print('Original Article: %s' % (article))
# print()
# print(tokens)