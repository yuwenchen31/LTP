# -*- coding: utf-8 -*-

# run on https://colab.research.google.com/
from google.colab import drive

drive.mount('/content/gdrive')

import pickle

with open('/content/gdrive/My Drive/LTP_Colab/genderList', 'rb') as f:
    genderList = pickle.load(f)

with open('/content/gdrive/My Drive/LTP_Colab/authorTextList', 'rb') as f:
    authorTextList = pickle.load(f)

#%% join 100 tweets per person

for i in range(0,len(authorTextList)):
    authorTextList[i] = " ".join(authorTextList[i])

#%%
import numpy as np

from keras.preprocessing.text import Tokenizer

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(authorTextList)
vocab_size = len(t.word_index) + 1
print(t.word_index)

#%% integer encode the documents
encoded_authorTextList = t.texts_to_sequences(authorTextList)

#%% load the whole embedding into memory, adapted from: adapted from: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
embeddings_index = dict()
f = open('/content/gdrive/My Drive/LTP_Colab/glove.6B.100d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        

#%% average word vectors into sentence vectors
x = []

for a in authorTextList:
    words = a.split(' ')
    vector = np.zeros((100))
    word_num = 0
    for b in words:
        if str(b) in embeddings_index:
            vector += embeddings_index[str(b)]
            word_num += 1
    if word_num > 0:
        vector = vector/word_num
    x.append(vector)    

x = np.asarray(x)


#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, genderList, test_size = 0.20, stratify=genderList)

#%% grid search
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 0.1, 1], 'C': [1, 100]}]

scores = ['precision', 'recall']

# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r")
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()


#%%

from sklearn.metrics import confusion_matrix

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))



clf = SVC(C=100, gamma=0.1, kernel='rbf') #best found by grid search

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

cf = confusion_matrix(y_test, y_pred, labels = ['female', 'male'])
print(classification_report(y_test, y_pred))
print (cf)

