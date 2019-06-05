# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% cd
import os

os.chdir("/home/eileen/Documents/Language Technology Project")
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

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

#%%
#tfidf example

corpus = ['This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?']
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
vectorizer.fit_transform(corpus).toarray()


#%%
#2-d to 1-d array
from itertools import chain 
flatten_list = list(chain.from_iterable(authorTextList))

#%%
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(flatten_list)
vectorizer.get_feature_names()
vectorizer.fit_transform(flatten_list).toarray()