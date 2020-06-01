from tools.loader import getData
from normalizer import get_only_pos, normalize
import os

# df = getData(force_reload=False)
# df['pos_tags'] = df['parsed_tokens'].apply(get_only_pos)
# # unique_pos = set(df['pos_tags'])
# for author_i in df.author.unique():
#
#         for i in range(len(df.author.unique)):
#             all_tags = df[df.author==author_i].iloc[i]['pos_tags']
#     print(type(all_tags))
# print(unique_pos)

def pos_tags_count(text):
    pos_counts = {}
    parsed_tokens = normalize(text)
    pos_tags = get_only_pos(parsed_tokens)
    unique_tags = ['VERB','ADJ','NOUN','ADV','NUM','SCONJ','CCONJ','CONJ']
    for tag in unique_tags:
        pos_counts[tag] = pos_tags.count(tag)/len(pos_tags)
    pos_counts['CONJ'] = pos_counts['SCONJ']+pos_counts['CCONJ']
    del pos_counts['SCONJ']
    del pos_counts['CCONJ']
    return pos_counts

def pos_words(parsed_tokens,POS):
    words = []
    for token in parsed_tokens:
        if token[1]==POS:
            words.append(token[0])
    return words

def words_count(tokens,N):
    wordcounts = {}
    unique_words = set(tokens)
    for token in unique_words:
        wordcounts[token] = tokens.count(token)
    wordcounts = sorted(wordcounts.items(), key=lambda x: x[1], reverse=True)
    res = wordcounts[:N]
    for i in range(len(res)):
        print(res[i][0])
    return res

corpus_dir = "/Users/lyuba/PycharmProjects/SVM/corpus/"
authors = [f for f in os.listdir(corpus_dir)] #creating a list of txt filenames in the current folder
authors.sort() #sort the file names in alphabetical order

for author in authors:
    print("\nAuthor:",author)
    text = open("/Users/lyuba/PycharmProjects/SVM/corpus/"+author).read()
    parsed_tokens = normalize(text)
    print("Nouns, verbs, adjectives and adverbs:")
    words_count(pos_words(parsed_tokens,'NOUN'),5)
    words_count(pos_words(parsed_tokens,'VERB'),5)
    words_count(pos_words(parsed_tokens,'ADJ'),5)
    words_count(pos_words(parsed_tokens,'ADV'),5)


POS = ['PRON', 'ADJ', 'NOUN', 'ADV', 'INTJ', 'AUX', 'CCONJ', 'PROPN', 'NUM', 'SCONJ', 'VERB', 'ADP', 'DET', 'X']