#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:28:00 2019

@author: chenfish
"""
#%% cd
import os

os.chdir("/Users/chenfish/Desktop/LTP")
#%%

import pickle

#%%
#open
with open('authorIDList', 'rb') as f:
     authorIDList = pickle.load(f)

with open('genderList', 'rb') as f:
     genderList = pickle.load(f)
     
with open('authorTextList', 'rb') as f:
     authorTextList = pickle.load(f)
     

#%% NN model 1

import numpy, json, argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#%% Read in word embeddings

def read_embeddings(embeddings_file):
	print('Reading in embeddings from {0}...'.format(embeddings_file))
	embeddings = json.load(open(embeddings_file, 'r'))
	embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
	print('Done!')
	return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
	vectorized_words = []
	for word in words:
		try:
			vectorized_words.append(embeddings[word.lower()])
		except KeyError:
			vectorized_words.append(embeddings['UNK'])
	return numpy.array(vectorized_words)



#%% Have to fix this!!! 
    
import numpy as np
from functools import reduce

from itertools import chain 
flatten_list = list(chain.from_iterable(authorTextList))

from nltk.tokenize import TweetTokenizer
import operator

tok_list = []

tknzr = TweetTokenizer()

for i in range(0,len(flatten_list)):
   tok_list.append(tknzr.tokenize(flatten_list[i]))


for i in range(0,3600):
        tok_list[i] = reduce(operator.concat, tok_list[i:i+100])

        


#for i in range(0,len(authorTextList)):
    #authorTextList[i] = "".join(authorTextList[i])
   
#%%

parser = argparse.ArgumentParser(description='KerasNN parameters')
parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
args = parser.parse_args()
    
# Read in the data and embeddings
embeddings = read_embeddings(args.embeddings)


#%%

X = tok_list
Y = genderList

#%%

# Transform words to embeddings
X = vectorizer(Ｘ, embeddings)
    
# Transform string labels to one-hot encodings
encoder = LabelBinarizer()
    
#Use encoder.classes_ to find mapping of one-hot indices to string labels
Y = encoder.fit_transform(Y) 

if args.binary:
    Y = numpy.where(Y == 1, [0,1], [1,0])
    
        
#%% Split in training and test data
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.2)


#%%

# Create model 
model = Sequential()
model.add(Dense(input_dim = X.shape[1], units = Y.shape[1]))
model.add(Activation("linear"))
sgd = SGD(lr = 0.001)
loss_function = 'mean_squared_error'
model.compile(loss = loss_function, optimizer = sgd, metrics=['accuracy'])  
    

#%%
# Fit the model
model.fit(X_train, y_train, verbose = 1, epochs = 20, batch_size = 160)
    

#%%
# Get predictions
y_pred = model.predict(X_test)
    
#%%
# Convert to numerical labels to get scores with sklearn in 6-way setting
y_pred = numpy.argmax(y_pred, axis = 1)
y_test = numpy.argmax(y_test, axis = 1)
print('Classification accuracy on test: {0}'.format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))


#%% NN model 2 

from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score

#%%

EMBEDDING_DIM = 50
N_CLASSES = 1
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(50,), dtype='int32')

#%%
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='softmax')(average)

model = Model(sequence_input, predictions)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
#%%

# Fit the model
model.fit(X_train, y_train, validation_split=0.1,
          nb_epoch=10, batch_size=128)

#%%

# Get prediction 
output_test = model.predict(X_test)

#%%
print("test auc:", roc_auc_score(y_test,output_test[:,0]))


