import glob
import os
import pandas as pd
from normalizer import normalize, get_only_tokens, lemmatize, remove_punctuation_numbers_case
from tools.corpus_divider import txtsToCsv
# from tools.stop_words import strip_stopwords

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, '*.csv')


def getData(force_reload = False) -> pd.DataFrame:

    pickle_path = os.path.join(current_dir, 'blogs.pkl')
    pickle_exist = os.path.isfile(pickle_path)

    if not force_reload and pickle_exist:
        return pd.read_pickle(pickle_path)
    else:
        if pickle_exist:
            os.remove(pickle_path)
        txtsToCsv(os.path.abspath(os.path.join(os.getcwd()))+"/corpus/")

        blogsDataFrame = pd.concat([pd.read_csv(f, encoding='utf-8') for f in glob.glob(path)],
                                ignore_index=True).dropna(how='any')


        blogsDataFrame['no_punct_num_lowercase'] = blogsDataFrame['text'].apply(remove_punctuation_numbers_case)
        blogsDataFrame['parsed_tokens'] = blogsDataFrame['text'].apply(normalize)

        # blogsDataFrame['normal_tokens'] - only normalized tokens [0] without POS, ej. 1 [in, fact, i'm, much, less, connected, to, my,...
        blogsDataFrame['normal_tokens'] = blogsDataFrame['parsed_tokens'].apply(get_only_tokens)

        # blogsDataFrame['normal_tokens_as_string'] - normalized tokens without punctuation concatenated back into one string, ej. 1 in fact i'm much less connected to my money an..
        blogsDataFrame['normal_tokens_as_string'] = blogsDataFrame['normal_tokens'].apply(lambda tokens: ' '.join(tokens))
        blogsDataFrame['all_normal_tokens'] = blogsDataFrame['text'].apply(lemmatize)
        blogsDataFrame['all_normal_tokens_as_string'] = blogsDataFrame['text'].apply(lemmatize).apply(lambda tokens: ' '.join(tokens))

        blogsDataFrame.to_pickle(pickle_path)

        return blogsDataFrame