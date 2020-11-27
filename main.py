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



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

#Model for the SubNN

def create_sub_DNN(input_shape=np.shape(descriptors)[2], optimizer=keras.optimizers.Adam()):
    # instantiate model
    model = Sequential()
    # 2 hidden layers, 30 neurons each; 
    model.add(Dense(30,input_shape=(input_shape), activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    # soft-max layer
    model.add(Dense(1, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

Sub_NN_O = create_sub_DNN()

Sub_NN_H = create_sub_DNN()

A=[Sub_NN_O, Sub_NN_O, Sub_NN_H, Sub_NN_H, Sub_NN_H, Sub_NN_H, Sub_NN_H]



#A.keras.layers.Add(Dense(1,input_shape=np.shape(A), activation='tanh'))