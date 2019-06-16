# -*- coding: utf-8 -*-
# run on https://colab.research.google.com/
from google.colab import drive

drive.mount('/content/gdrive')

import pickle

with open('/content/gdrive/My Drive/LTP_Colab/authorTextList', 'rb') as f:
    authorTextList = pickle.load(f)
    
with open('/content/gdrive/My Drive/LTP_Colab/genderList', 'rb') as f:
    genderList = pickle.load(f)

#%% join 100 tweets per person

for i in range(0,len(authorTextList)):
    authorTextList[i] = " ".join(authorTextList[i])

from sklearn.feature_extraction.text import TfidfVectorizer


#%% convert gender list to int
for (i, item) in enumerate(genderList):
    if item == 'female':
        genderList[i] = 1
    else:
        genderList[i] = 0


#%% tfidf vectorize
vec = TfidfVectorizer(min_df=0.005, ngram_range=(1,2))

tfidf_mat = vec.fit_transform(authorTextList).toarray()

import numpy as np


#%% split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tfidf_mat, genderList, test_size = 0.20, shuffle=True, stratify=genderList)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#%%reshape for RNN
step = 2
feature = x_train.shape[1]//step
x_train_reshape = np.reshape(x_train, (x_train.shape[0], step, feature))
x_test_reshape = np.reshape(x_test, (x_test.shape[0], step, feature))


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, SimpleRNN
from keras.optimizers import Adam


#%% feed forward model
model_f = Sequential()
model_f.add(Dense(500,input_shape=(16094,)))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(100))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(50))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(10))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(5))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(2))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))
model_f.add(Dense(1))
model_f.add(Activation('sigmoid'))
model_f.summary()
model_f.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#%% RNN model
model_r = Sequential()
model_r.add(SimpleRNN(1000, input_shape=(step, feature))) 
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(500))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(100))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(50))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(10))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(5))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(2))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))
model_r.add(Dense(1))
model_r.add(Activation('sigmoid'))
model_r.summary()


model_r.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['acc'])

#%%
from keras.callbacks import EarlyStopping, ModelCheckpoint

#after the point that validation loss started to degrade
mc_r = ModelCheckpoint('best_model_r', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
mc_f = ModelCheckpoint('best_model_f', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#%% fit model
history_r = model_r.fit(x_train_reshape, y_train, validation_data=(x_test_reshape, y_test), epochs=50, verbose=1, callbacks=[mc_r], batch_size=256)
history_f = model_f.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1, callbacks=[mc_f], batch_size=256)

#%% plot
import matplotlib.pyplot as plt
from keras.models import load_model

plt.plot(history_f.history['acc'], linestyle='--', color='lime')
plt.plot(history_f.history['val_acc'], color='lime')
plt.plot(history_r.history['acc'], linestyle='--', color='purple')
plt.plot(history_r.history['val_acc'],color='purple')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Feed-forward train', 'Feed-forward test', 'RNN train', 'RNN test'], loc='bottom right') 
plt.show()

plt.plot(history_f.history['loss'], linestyle='--', color='lime')
plt.plot(history_f.history['val_loss'], color='lime')
plt.plot(history_r.history['loss'], linestyle='--', color='purple')
plt.plot(history_r.history['val_loss'],color='purple')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Feed-forward train', 'Feed-forward test', 'RNN train', 'RNN test'], loc='upper left') 
plt.show()

#%% evaluate the models

loss_r, accuracy_r = model_r.evaluate(x_test_reshape, y_test, verbose=0)
print(accuracy_r)
print(loss_r)

loss_f, accuracy_f = model_f.evaluate(x_test, y_test, verbose=0)
print(accuracy_f)
print(loss_f)

#%% load the best models
saved_model_r = load_model('best_model_r')
saved_model_f = load_model('best_model_f')

#%% evaluate the best models

best_loss_r, best_acc_r = saved_model_r.evaluate(x_test_reshape, y_test, verbose=0)
print(best_acc_r)
print(best_loss_r)

best_loss_f, best_acc_f = saved_model_f.evaluate(x_test, y_test, verbose=0)
print(best_acc_f)
print(best_loss_f)

