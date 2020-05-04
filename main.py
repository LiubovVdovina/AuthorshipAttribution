from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.loader import getData

RANDOM_STATE = 1500

blogsDataFrame = getData(force_reload=False)

# transform non-numerical labels to numerical labels
labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder \
    .fit(blogsDataFrame['author'].unique()) \
    .transform(blogsDataFrame['author'].values)


# Split data set into training set and test set, ej. 20% training and 80% test
X_train, X_test, y_train, y_test = train_test_split(blogsDataFrame['normal_tokens_as_string'], y, test_size=0.2,random_state=109, stratify=y)

#Text char/word n-grams vectorization concerning their frequency
# count_vect = CountVectorizer(lowercase='false',analyzer='char',ngram_range=(1,2))
# X_train_charCounts = count_vect.fit_transform(X_train)
# X_test_charCounts = count_vect.transform(X_test)

# 1st - order the counts in descending order, then from this list, each feature name is extracted and returned with corresponding counts
# sorted_items=sort_coo(X_train_charCounts[0].thisocoo())
# feature_names=count_vect.get_feature_names()
# n_grams=extract_topn_from_vector(feature_names,sorted_items,10)
# print("Top n most frequent n-grams: ",n_grams)
# print("Shape of count vector: (X - number of train docs and Y number of unique words/characters/n-grams: ",X_train_charCounts.shape)
# print("Resulting vocabulary; the numbers are not counts, they are the position in the sparse vector: ",count_vect.vocabulary_)
# print("Train count vectors for each document: ",X_train_charCounts.toarray())


# Text tf-idf vectorization (weights concerning frequency of a token in one doc corresponding to frequency in all docs)
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(blogsDataFrame['normal_tokens_as_string'])
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# print("Type of X_train_tfidf is ", type(X_train_tfidf), "value of X_train_tfidf is ", X_train_tfidf)

# OPTION #1 - Create a SVM Classifier
svm_clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
svm_clf.fit(X_train_tfidf, y_train)

#Predict the response for test dataset
predictions_SVM = svm_clf.predict(X_test_tfidf)

report = classification_report(y_test, predictions_SVM, target_names=labelEncoder.classes_, digits=5)
print("Report for SVM Classifier:\n",report)

# OPTION #2 create a NAIVE BAYES Classifier
# naive_clf = naive_bayes.MultinomialNB()
# # fit the training dataset on the Naive Bayes classifier
# naive_clf.fit(X_train_tfidf, y_train)
#
# # print("X_train_tfidf: ",X_train_tfidf)
# # predict the labels on validation dataset
# predictions_NB = naive_clf.predict(X_test_tfidf)
#
# # Use classification_report function to get the full report
# report = classification_report(y_test, predictions_NB, target_names=labelEncoder.classes_, digits=5)
# print("Report for Naive Bayes Classifier:\n",report)

# OPTION #3 create a Dummy Classifier sth is wrong - gives a warning
# dummy_clf = DummyClassifier(strategy='uniform', random_state=RANDOM_STATE)
# dummy_clf.fit(X_train_tfidf, y_train)
# predictions_dummy = dummy_clf.predict(X_test_tfidf)
# report = classification_report(y_test, predictions_dummy, target_names=labelEncoder.classes_, digits=5)
# print("Report for a Dummy Classifier:\n",report)
#
# # OPTION #4 - LogisticRegression NOT FINISHED
# log_clf = LogisticRegression()
