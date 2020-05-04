def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """return n-gram counts in descending order of counts"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    results=[]

    # word index, count i
    for idx, count in sorted_items:

        # get the ngram name
        n_gram=feature_names[idx]

        # collect as a list of tuples
        results.append((n_gram,count))

    return results