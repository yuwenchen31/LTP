# -*- coding: utf-8 -*-
# run on https://colab.research.google.com/
from google.colab import drive

drive.mount('/content/gdrive')

import pickle

with open('/content/gdrive/My Drive/LTP_Colab/genderList', 'rb') as f:
    genderList = pickle.load(f)

with open('/content/gdrive/My Drive/LTP_Colab/authorTextList', 'rb') as f:
    authorTextList = pickle.load(f)

# join 100 tweetw together
for i in range(0,len(authorTextList)):
    authorTextList[i] = " ".join(authorTextList[i])

#%% convert gender list to int
for (i, item) in enumerate(genderList):
    if item == 'female':
        genderList[i] = 1
    else:
        genderList[i] = 0

#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

t = Tokenizer()
t.fit_on_texts(authorTextList)
vocab_size = len(t.word_index) + 1

encoded_authorTextList = t.texts_to_sequences(authorTextList)

#%% get max length for padding
from nltk.tokenize import word_tokenize 

get_length = []
for i in range(0,len(authorTextList)):
    authorTextList[i] = word_tokenize(authorTextList[i])
    length=len(authorTextList[i])
    get_length.append(length)
    authorTextList[i] = " ".join(authorTextList[i])

max_length = max(get_length)

#%%
padded_authorTextList = pad_sequences(encoded_authorTextList, maxlen=max(get_length), padding='post')

#%% load GloVe, adapted from: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
embeddings_index = dict()
f = open('/content/gdrive/My Drive/LTP_Colab/glove.6B.100d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


#%% split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(padded_authorTextList, genderList, test_size = 0.20, shuffle=True, stratify=genderList)

#%%
from keras.callbacks import ModelCheckpoint

mc_f = ModelCheckpoint('best_model_f', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
mc_r = ModelCheckpoint('best_model_r', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, SimpleRNN
from keras.optimizers import Adam

#%% feed forward model
model_f = Sequential()
e_f = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=max(get_length), trainable=False)
model_f.add(e_f)
model_f.add(Flatten())

model_f.add(Dense(100))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(50))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(40))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(30))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(20))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(10))
model_f.add(Dropout(0.2))
model_f.add(Activation('relu'))

model_f.add(Dense(1, activation='sigmoid'))
model_f.compile(loss='binary_crossentropy',
             optimizer=Adam(lr=0.001),
              metrics=['acc'])
model_f.summary()

#%% RNN model
model_r = Sequential()
e_r = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model_r.add(e_r)

model_r.add(SimpleRNN(100))
model_r.add(Dropout(0.5))
model_r.add(Activation('relu'))

model_r.add(Dense(60))
model_r.add(Dropout(0.5))
model_r.add(Activation('relu'))

model_r.add(Dense(40))
model_r.add(Dropout(0.5))
model_r.add(Activation('relu'))

model_r.add(Dense(20))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))

model_r.add(Dense(10))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))

model_r.add(Dense(5))
model_r.add(Dropout(0.2))
model_r.add(Activation('relu'))


model_r.add(Dense(1, activation='sigmoid'))
model_r.compile(loss='binary_crossentropy',
             optimizer=Adam(lr=0.0001),
              metrics=['acc'])
model_r.summary()



#%% fit model
history_f = model_f.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1, callbacks=[es, mc_f], shuffle=True, batch_size=256)
history_r = model_r.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1, callbacks=[es, mc_r], shuffle=True, batch_size=256)

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

loss_r, accuracy_r = model_r.evaluate(x_test, y_test, verbose=0)
print(accuracy_r)
print(loss_r)

loss_f, accuracy_f = model_f.evaluate(x_test, y_test, verbose=0)
print(accuracy_f)
print(loss_f)

#%% load the best models
saved_model_r = load_model('best_model_r')
saved_model_f = load_model('best_model_f')

#%% evaluate the best models
best_loss_r, best_acc_r = saved_model_r.evaluate(x_test, y_test, verbose=0)
print(best_acc_r)
print(best_loss_r)


best_loss_f, best_acc_f = saved_model_f.evaluate(x_test, y_test, verbose=0)
print(best_acc_f)
print(best_loss_f)

