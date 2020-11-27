#!/usr/bin/env python3

import data
import NN
import numpy as np

soap_params = {
	'sigma': 0.01,
	'nmax': 3, 
	'lmax': 3,
	'rcut': 5
}

limit=1000

# Load dataset and compute descriptors
descriptors, energies = data.load_and_compute(
	dataset='zundel', soap_params=soap_params, limit=limit)

model = NN.create(n_types=2, atoms=[0,0,1,1,1,1,1], desc_length=np.shape(descriptors)[2])
#model.build(input_shape=(84,))
print(model.summary())

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten

##Model for the SubNN
#
#def create_sub_DNN(input_shape=np.shape(descriptors)[2], optimizer=keras.optimizers.Adam()):
#    # instantiate model
#    model = Sequential()
#    # 2 hidden layers, 30 neurons each; 
#    model.add(Dense(30,input_shape=(input_shape), activation='tanh'))
#    model.add(Dense(30, activation='tanh'))
#    # soft-max layer
#    model.add(Dense(1, activation='softmax'))
#    
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=optimizer,
#                  metrics=['accuracy'])
#    
#    return model
#
#Sub_NN_O = create_sub_DNN()
#
#Sub_NN_H = create_sub_DNN()
#
#A=[Sub_NN_O, Sub_NN_O, Sub_NN_H, Sub_NN_H, Sub_NN_H, Sub_NN_H, Sub_NN_H]



#A.keras.layers.Add(Dense(1,input_shape=np.shape(A), activation='tanh'))
