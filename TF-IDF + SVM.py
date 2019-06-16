# -*- coding: utf-8 -*-
# run on https://colab.research.google.com/
#%% cd
drive.mount('/content/gdrive')

import pickle

with open('/content/gdrive/My Drive/LTP_Colab/genderList', 'rb') as f:
    genderList = pickle.load(f)

with open('/content/gdrive/My Drive/LTP_Colab/authorTextList', 'rb') as f:
    authorTextList = pickle.load(f)
#%%

import pickle

#%%
#open

with open('genderList', 'rb') as f:
    genderList = pickle.load(f)
     
with open('authorTextList', 'rb') as f:
    authorTextList = pickle.load(f)


#%% join 100 tweets per person

for i in range(0,len(authorTextList)):
    authorTextList[i] = " ".join(authorTextList[i])

#%%

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#%%
tfidf = TfidfVectorizer()
svclassifier = SVC(C=1)

pipe = Pipeline([
        ('vect', tfidf),
        ('clf', svclassifier)
        ])


    
tuned_parameters= {
        'vect__min_df': (0, 0.005, 0.05), #0.005
        'vect__ngram_range': [(1, 2), (1, 3)], #(1,2)
        'clf__kernel': ('rbf', 'linear', 'sigmoid'), #linear
        'clf__gamma': (1e-3, 1e-4), #0.001
        'clf__C': (0.1, 1, 10),#1
        'clf__tol': (0.001, 0.0001) #0.001
        }

#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(authorTextList, genderList, test_size = 0.20)


#%%model fit

grid_search = GridSearchCV(estimator=pipe,
                       param_grid=tuned_parameters,
                       cv=5,
                       n_jobs=-1,
                       verbose=10)

grid_search.fit(x_train, y_train)  


#%% model results

from sklearn.metrics import classification_report

#best score
grid_search.best_score_

#best parameters
grid_search.best_estimator_

# Prediction performance on test set
grid_search.score(x_test, y_test) 
y_true, y_pred = y_test, grid_search.predict(x_test)
print(classification_report(y_test, y_pred))

