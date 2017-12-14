from gensim import models
from gensim.corpora import dictionary
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling2D, Concatenate, Dropout, Dense
from keras.layers.core import Reshape, Flatten
from keras.utils import to_categorical
import numpy as np
from pandas import read_pickle
from sklearn.model_selection import train_test_split
import h5py

df = read_pickle('../data/10k-cleaned.pd')

train, test = train_test_split(df)

dictionary_obj = dictionary.Dictionary(list(train['tokens']))
dictionary_size = len(dictionary_obj)

x_train = np.array([np.array(dictionary_obj.doc2idx(document=tokens)) for tokens in train['tokens']]) 
x_test = np.array([np.array(dictionary_obj.doc2idx(document=tokens)) for tokens in test['tokens']])

y_train = to_categorical(np.array(train['Sentiment']))
y_test = to_categorical(np.array(test['Sentiment']))

sequence_length = x_train.shape[1]
embedding_dim = 256
filter_sizes = [5, 6, 7]
num_filters = 50
drop = 0.5
epochs = 6

inputs = Input(shape=(sequence_length,), dtype='int32')
reshape = Reshape((sequence_length, 1))(inputs)

conv0 = Conv1D(num_filters, (filter_sizes[0]), 
        activation='relu', padding='valid', kernel_initializer='random_uniform')(inputs)
conv1 = Conv1D(num_filters, (filter_sizes[1]), 
        activation='relu', padding='valid', kernel_initializer='random_uniform')(inputs)
conv2 = Conv1D(num_filters, (filter_sizes[2]), 
        activation='relu', padding='valid', kernel_initializer='random_uniform')(inputs)

maxpool0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid', data_format='channels_last')(conv0)
maxpool1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid', data_format='channels_last')(conv1)
maxpool2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid', data_format='channels_last')(conv2)

merged = Concatenate(axis=1)([maxpool0, maxpool1, maxpool2])
flattened = Flatten()(merged)
dropout = Dropout(drop)(flattened)
output = Dense(units=2, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
model.save('cnn.h5')
