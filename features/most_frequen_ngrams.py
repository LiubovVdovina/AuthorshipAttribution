from tools.loader import getData
from collections import Counter
import os

# df = getData(force_reload=False)

# def char_ngram(string, n=3):
#     return([string[i:i+n] for i in range(len(string)-n+1)])
#
#
# #For n = 2-4
# for author_i in df.author.unique():
#     print(author_i)
#     # at is 'numpy.ndarray' with all texts for the current author, len(at) - total number of texts
#     at = df[df.author == author_i].text.values
#     for counter in range(4):
#         tmp_dict = {}
#         counter = counter+1
#         for i in range(len(at)):
#             tmp = char_ngram(at[i], n=counter)
#             for tmp1 in tmp:
#                 if tmp1 in tmp_dict:
#                     tmp_dict[tmp1] += 1
#                 else:
#                     tmp_dict[tmp1] = 1
#         d = Counter(tmp_dict)
#         for k, v in d.most_common(20):
#             print('%s: %i' % (k, v))
#         print("\n")

corpus_dir = "/Users/lyuba/PycharmProjects/SVM/corpus/"
authors = [f for f in os.listdir(corpus_dir)] #creating a list of txt filenames in the current folder
authors.sort() #sort the file names in alphabetical order

def alphabetic_count(text):
    characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    charcounts = {}
    for char in characters: charcounts[char] = text.count(char)
    for char in characters: charcounts[char] = charcounts[char]/len(text)
    print(*charcounts.values(),sep="\n")
    return charcounts

def uppercase_count(text):
    uppercase_counter = 0
    for char in text:
        if char.isupper():
            uppercase_counter += 1
    res = uppercase_counter/len(text)
    print(res)
    return(res)

def digits_count(text):
    digits_counter = 0
    for char in text:
        if char.isdigit():
            digits_counter += 1
    res = digits_counter/len(text)
    print(res)
    return(res)

for author in authors:
    # print("Author:",author)
    text = open("/Users/lyuba/PycharmProjects/SVM/corpus/"+author).read()
    digits_count(text)