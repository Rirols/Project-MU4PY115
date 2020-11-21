#!/usr/bin/env python3
import numpy as np
import keras,sklearn,pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from dscribe.descriptors import SOAP
from ase.io import read
from ase.build import molecule
from ase import Atoms

#SOAP parameters
species = ["H", "O"]
#sigma=
nmax = 3
lmax = 8
rcut = 5

#Configure SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="outer",
    sparse=False
)

#Create NN: input = descriptors, output = system's total energy

#Step 1: process data

#Import data
energies = pickle.load(open('./data/zundel_100K_energy', 'rb'))
pos = pickle.load(open('./data/zundel_100K_pos', 'rb'))

#Take fraction of the data

#Process data: positions => descriptors

#Step 2: Define NN and architecture

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_DNN():
    # instantiate model
    model = Sequential()    
    return model

#Step 3: Optimizer and cost function

def compile_model(optimizer=keras.optimizers.Adam()):
    # create the mode
    model=create_DNN()
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

#Step 4: Train the model
    
# training parameters
batch_size = 64
epochs = 10

# create the deep neural net
model_DNN=compile_model()

# train DNN and store training info in history
history=model_DNN.fit(descriptors_train, energies_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(descriptors_test, energies_test))

#Step 5: evaluate performance

#Step 6: Optimize