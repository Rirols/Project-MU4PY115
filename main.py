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

# Load dataset and compute descriptors
descriptors, energies = data.load(
	dataset = 'zundel', soap_params=soap_params, limit=1000)

print(np.shape(descriptors))
print(np.shape(energies))

#Create NN: input = descriptors, output = system's total energy

#Step 1: process data

#Import data

#Take fraction of the data

#Process data: positions => descriptors

#Step 2: Define NN and architecture

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#
#def create_DNN():
#    # instantiate model
#    model = Sequential()    
#    return model
#
##Step 3: Optimizer and cost function
#
#def compile_model(optimizer=keras.optimizers.Adam()):
#    # create the mode
#    model=create_DNN()
#    # compile the model
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=optimizer,
#                  metrics=['accuracy'])
#    return model
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
