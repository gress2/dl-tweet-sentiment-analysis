from gensim import models
from gensim.corpora import dictionary
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense
from keras.layers.core import Reshape, Flatten
from keras.utils import to_categorical
import numpy as np
from pandas import read_pickle
from sklearn.model_selection import train_test_split

df = read_pickle('../data/10k-cleaned.pd')

train, test = train_test_split(df)

dictionary_obj = dictionary.Dictionary(list(train['tokens']))
dictionary_size = len(dictionary_obj)

x_train = list()
x_test = list()

for row in train['tokens']:
    cur = np.zeros(dictionary_size)
    for idx in dictionary_obj.doc2idx(row):
        cur[idx] = 1
    x_train.append(cur)
    
for row in test['tokens']:
    cur = np.zeros(dictionary_size)
    for idx in dictionary_obj.doc2idx(row):
        cur[idx] = 1
    x_test.append(cur)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = to_categorical(np.array(train['Sentiment']))
y_test = to_categorical(np.array(test['Sentiment']))

drop = 0.5
epochs = 6

model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(drop))
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(drop))
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(drop))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
model.save('mlp_one_hot.h5')

