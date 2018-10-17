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

def TrainModel(loadpath,savepath, epochs, batch_size,train_size,
    #Load the Data to fit the model
    f=h5py.File(loadpath + 'images_shuf.hdf5','r')
    g= h5py.File(loadpath + 'labels_shuf.hdf5','r')
    X = np.array(f.get('images'))
    Y = np.array(g.get('labels'))
    f.close()
    g.close()

    num_classes = len(np.unique(Y))
    epochs = epochs
    batch_size = batch_size
    (L,M,N,C) = X.shape
    train_size = train_size
    test_size = L-train_size

    #Data Preprocessing
    X = X.astype('float32')
    #X /= 255 #normalize grayscale
    # X = np.reshape(X,(len(X),M,N,C,1)) #reshape in order to give the "dummy dimension" that would normally be occupied by RGB
    Y = keras.utils.to_categorical(Y, num_classes) #y must be zero indexed 

    x_train = X[0:train_size]
    y_train = Y[0:train_size]
    x_test = X[train_size:L]
    y_test = Y[train_size:L]

    #Data Output Initialize:
    save_dir = savepath + '\\Logs'
    #conditional statement necessary if this is an existing path, error is thrown otherwise.
    if os.path.exists(savepath)==False:
        os.makedirs(savepath)        
    csv_logger = CSVLogger(savepath + '\\acclosslogdo04.csv')

    #Training/Fitting:
    hist = Qnet.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs =epochs, 
                    verbose=1, 
                    validation_data=(x_test, y_test),
                    callbacks=[csv_logger])
    print(hist.history)


    score = Qnet.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)

    Qnet.save(savepath + '\\Qnetdo04.h5")
              
              
