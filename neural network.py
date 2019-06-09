# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% cd
import os

os.chdir("C:\Ddrive\Master\Language Technology Project\Project")
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
     
     

#%%
#tfidf example
'''
corpus = ['This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?']
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
vectorizer.fit_transform(corpus).toarray()
'''

#%% join 100 tweets per person

for i in range(0,len(authorTextList)):
    authorTextList[i] = " ".join(authorTextList[i])



#%%

from sklearn.feature_extraction.text import TfidfVectorizer

#%%
vec = TfidfVectorizer(min_df=0.005, ngram_range=(1,2))

tfidf_mat = vec.fit_transform(authorTextList).toarray()

#%% convert gender list to int
for (i, item) in enumerate(genderList):
    if item == 'female':
        genderList[i] = 1
    else:
        genderList[i] = 0

#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tfidf_mat, genderList, test_size = 0.20)

#grid_search.fit(x_train, y_train)  


#%%
# build model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

#%%
model = Sequential()
model.add(Dense(18000,input_shape=(16094,)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(15000))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(12000))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(10000))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(7500))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(5000))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(2500))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#%%
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=5)