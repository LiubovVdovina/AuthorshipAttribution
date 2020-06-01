import numpy as np
from tools.loader import getData

def average_word_length(text):
    avg_word_len = np.mean([len(w) for w in str(text).split()])
    return avg_word_len

print(average_word_length("Moooom I love you"))

def avg_word_len_df():
    df = getData(force_reload=False)
    df['avg_word_len'] = df['no_punct_num_lowercase'].apply(average_word_length)
    objects = {}
    for author_i in df.author.unique():
        objects[author_i] = sum(df[df.author==author_i]['avg_word_len'])/len(df[df.author==author_i])
    return objects

print(*avg_word_len_df().values(),sep="\n")
