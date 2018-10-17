import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
# from keras.datasets import mstar
from keras.layers import Input, ZeroPadding2D, Dense, Activation, BatchNormalization, Dropout, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
import os
import h5py
from keras import regularizers as reg
import util



def InitModel(M,N,C,num_classes):
    input_shape = (M,N,C)
    #Neural Network Architecture Main:
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), name='max_pool1')(X)

    X = Conv2D(64, (5, 5), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), name='max_pool2')(X)

    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(128, activation = 'relu', name='fc0')(X)
    X = Dropout(.2)(X)
    X = Dense(num_classes, activation='softmax', name='fc2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    Qnet = Model(inputs=X_input, outputs=X, name='HappyModel')

    #optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)    optimizer = SGD(lr=.01, decay=1e-6, momentum=0.9, nesterov=True)
    Qnet.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return Qnet

