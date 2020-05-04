import glob
import os
import pandas as pd
from normalizer import normalize, not_normalize
from corpus_divider import txtsToCsv

# from tools.stop_words import strip_stopwords

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, '*.csv')


def get_normal_tokens(tokens_parsed=[]):
    return [t[0] for t in tokens_parsed]


def getData(force_reload = False) -> pd.DataFrame:

    txtsToCsv(os.path.abspath(os.path.join(os.getcwd()))+"/corpus/")
    pickle_path = os.path.join(current_dir, 'blogs.pkl')
    pickle_exist = os.path.isfile(pickle_path)

    if not force_reload and pickle_exist:
        return pd.read_pickle(pickle_path)
    else:
        if pickle_exist:
            os.remove(pickle_path)

        blogsDataFrame = pd.concat([pd.read_csv(f, encoding='utf-8') for f in glob.glob(path)],
                                ignore_index=True).dropna(how='any')

        # blogsDataFrame['punctuation_tokens'] = blogsDataFrame['text'].apply(lambda text: not_normalize(text))

        # blogsDataFrame['text'] - text as a string with punctuaction, uppercase, etc., ej. 1 In fact, I'm much less connected to my money a...
        # blogsDataFrame['parsed_tokens'] -  text in tokens [0] in lowercase  + postag [1], ej. 1 [(in, IN), (fact, NN), (i'm, RB), (much, RB), ...
        blogsDataFrame['parsed_tokens'] = blogsDataFrame['text'].apply(lambda text: normalize(text))

        # blogsDataFrame['normal_tokens'] - only normalized tokens [0] without POS, ej. 1 [in, fact, i'm, much, less, connected, to, my,...
        blogsDataFrame['normal_tokens'] = blogsDataFrame['parsed_tokens'].apply(get_normal_tokens)

        # blogsDataFrame['normal_tokens_as_string'] - normalized tokens without punctuation concatenated back into one string, ej. 1 in fact i'm much less connected to my money an..
        blogsDataFrame['normal_tokens_as_string'] = blogsDataFrame['normal_tokens'].apply(lambda tokens: ' '.join(tokens))

        blogsDataFrame.to_pickle(pickle_path)

        return blogsDataFrame