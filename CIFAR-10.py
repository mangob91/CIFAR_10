# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:51:26 2018

@author: leeyo
"""

#%%
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
#%%    
def import_data(filePath):
    feature = np.zeros(shape = (10000,3072), dtype = np.uint8)
    label = np.zeros(shape = (10000,1), dtype = np.int32)
    with open(filePath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    temp_keys = list(dict.keys())
    feature = dict[temp_keys[2]]
    label = np.array(dict[temp_keys[1]]).reshape(len(dict[temp_keys[1]]),1)
    yield feature, label

#%% importing train data

filePath = r'C:\Users\leeyo\OneDrive\Documents\Columbia Univ\Advanced Machine Learning\Team Project\Image Dataset\cifar-10-batches-py'
fileName = 'data_batch_'
fullPath = os.path.join(filePath, fileName)
#%%
image_train = np.empty((0,3072), dtype = np.uint8)
label_train = np.empty((0,1))
for i in range(1,6):
    for feature, label in import_data(fullPath + str(i)):
        image_train = np.vstack((feature,image_train))
        label_train = np.vstack((label,label_train))
#%%
train_mean = np.mean(image_train, axis = 1).reshape(len(image_train),1)
image_train = np.subtract(image_train, train_mean)
#%% loading test data
image_test = np.empty((0,3072), dtype = np.uint8)
label_test = np.empty((0,1))
fileName_test = 'test_batch'
fullPath_test = os.path.join(filePath, fileName_test)
for feature, label in import_data(fullPath_test):
    image_test = feature
    label_test = label
#%% min max scaler
scaler = MinMaxScaler(copy = False)
scaler.fit(image_train)
scaler.transform(image_train)
scaler.transform(image_test)
#%%
from sklearn.linear_model import LogisticRegression
seed = 42
solver = 'saga'
max_iter = 200
logistic = LogisticRegression(random_state = seed, solver = solver, max_iter = max_iter, multi_class = 'auto')
logistic.fit(image_train, label_train.ravel())
#%%
from sklearn.metrics import accuracy_score
predicted = logistic.predict(image_test)
accuracy_score(label_test, predicted)
#%%
def one_hot_transform(label):
    oh = OneHotEncoder()
    oh.fit(label_train)
    oh_trainlabel = oh.transform(label_train)
    return oh_trainlabel

def to_tensor(feature):
    nrow = 32
    ncol = 32
    nchannel = 3
    tensor = np.zeros((feature.shape[0], nrow, ncol, nchannel))
    for i in range(nchannel):
        tensor[:,:,:,i] = feature[:,(ncol**2)*i:(ncol**2)*(i+1)].reshape((-1, nrow, ncol))
    return tensor
#%%
label_keras = one_hot_transform(label_train)
feature_keras = to_tensor(image_train)
#%%
import matplotlib.pyplot as plt
plt.imshow(feature_keras[2])
#%%
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#%%
#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

baseMapNum = 32
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)

#training
batch_size = 64
epochs=25
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(x_test,y_test))
model.save_weights('cifar10_normal_rms_ep75.h5')

opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
model.save_weights('cifar10_normal_rms_ep100.h5')

opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
model.save_weights('cifar10_normal_rms_ep125.h5')

#testing - no kaggle eval
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))