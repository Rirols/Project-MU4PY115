#!/usr/bin/env python3

#import matplotlib.pyplot as plt
#import tensorflow as tf
#import keras, sklearn, pickle
#from dscribe.descriptors import SOAP
#from ase.io import read
#from ase.build import molecule
#from ase import Atoms

import data
import numpy as np

soap_params = {
	'sigma': 0.01,
	'nmax': 3, 
	'lmax': 3,
	'rcut': 5
}

limit=10000

# Load dataset and compute descriptors
descriptors, energies = data.load_and_compute(
	dataset='zundel', soap_params=soap_params, limit=limit, parallelize=True)

print(np.shape(descriptors))
print(np.shape(energies))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def create_sub_DNN(dropout_rate, descript, optimizer=keras.optimizers.Adam()):
    # instantiate model
    model = Sequential()
    # 2 hidden layers, 30 neurons each; 
    model.add(Dense(30,input_shape=(np.shape(descript)), activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    # soft-max layer
    model.add(Dense(1, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


#
##Step 4: Train the model
#    
## training parameters
#batch_size = 64
#epochs = 10
#
## create the deep neural net
#model_DNN=compile_model()
#
## train DNN and store training info in history
#history=model_DNN.fit(descriptors_train, energies_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(descriptors_test, energies_test))
#
##Step 5: evaluate performance
#
##Step 6: Optimize
