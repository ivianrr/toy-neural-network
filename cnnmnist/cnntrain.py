import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from toynetwork import Network, Layer,Functions
import numpy as np
import pickle
import matplotlib.pyplot as plt
from util import networkfunctions as Functions
from util import mnistdataset as mnist
from pathlib import Path
import pandas as pd
#-------------------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

#from scipy.stats import pearsonr
#import seaborn as sns
from scipy.stats import kde

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.losses import Huber
from tensorflow.keras import Model, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, concatenate, LeakyReLU, Convolution1D, MaxPooling1D, BatchNormalization, SpatialDropout1D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
#-------------------------


def architecture(iptdim): # spc dim, imput data (10 params) 

    ipt       = Input(shape=(iptdim, 1))     #INPUT DIMENSION OF THE SPECTROSCOPIC CHANNEL
    ipt_all = [ipt]

    x = Convolution1D(16, 10, padding='valid', name = 'convol_1')(ipt)   #CONVOLUTION LAYER 1
    x = LeakyReLU()(x)
    x = SpatialDropout1D(0.2)(x)
    x = Convolution1D(8, 10, padding='valid', name = 'convol_2')(x)     #CONVOLUTION LAYER 2
    x = LeakyReLU()(x)
    x = SpatialDropout1D(0.2)(x)
    x = Convolution1D(4, 10, padding='valid', name = 'convol_3')(x)     #CONVOLUTION LAYER 3
    x = LeakyReLU()(x)
    x = SpatialDropout1D(0.2)(x)

    x = Dropout(0.2)(x)    #DROPOUT LAYER

    x = Flatten()(x)
    x = Dense(32)(x)       #FULLY CONNECTED LAYER
    x = LeakyReLU()(x)

    x = Dense(16)(x)
    x = LeakyReLU()(x)

    # cnnout = Dense(y_train.shape[1], activation='linear')(x)  #OUTPUT LAYER
    cnnout = Dense(10, activation='softmax')(x)  #OUTPUT LAYER
    cnn = Model(ipt_all, cnnout)

    return cnn


if __name__ == "__main__":
    limit=None
    v="cnn2"
    training=True
    n_epoch_max=30
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')


    path=Path.cwd()/"cnnmnist"
    print(Path.cwd())
    # Load dataset
    Y_T, X_T, nrows, ncols = mnist.load_data()
    # X_T = X_T.T
    Y_V, X_V, _, _ = mnist.load_test_data()
    # X_V = X_V.T

    ### RESCALE LABELS FROM 0 TO 1
    scaler_y = preprocessing.MinMaxScaler()
    y_train  = scaler_y.fit_transform(Y_T.reshape(-1,1))[:limit,:]
    # print(Y_T.reshape(-1,1).shape) # (60000,1)
    y_test   = scaler_y.transform(Y_V.reshape(-1,1))

    ### RESCALE PARAMS FROM 0 TO 1
    scaler_x = preprocessing.MinMaxScaler()
    # print(X_T.shape) #(60000, 784)
    X_train   = scaler_x.fit_transform(X_T)[:limit,:]
    X_test    = scaler_x.transform(X_V)

    cnn = architecture(X_train.shape[1])
    cnn.summary()


    cnn.compile(loss='sparse_categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])



    if training:
        history = cnn.fit([X_train], y_train,
                        batch_size=16,
                        epochs=n_epoch_max,
                        validation_data=([X_test], y_test),
                        verbose=1,
                        callbacks=[early_stop])
        cnn.save(path/"Models"/f'model_{v}')
        history_df=pd.DataFrame.from_dict(history.history)
        history_df.to_csv(path/"models"/f'histories_{v}.csv')
    else:
        cnn=load_model(path/"Models"/f'model_{v}')
    saved_h = pd.read_csv(path/"models"/f'histories_{v}.csv')
            
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(20,7))
    plt.subplot(121)
    plt.plot(saved_h["loss"], c='orange')
    plt.plot(saved_h["val_loss"], c='b')
    plt.yscale('log')
    plt.xlabel("N_epoch")
    plt.ylabel("Loss")
    plt.legend(['loss','val_loss'])
    #SAVE THE CNN MODEL
    # cnn.save('Models/model_version_' + v + '_' + str(use_phot) + '_%03d' % index_run)  # creates a HDF5 file
    plt.show(block=True)