#!/usr/bin/env python3

from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense

def create_submodel(neurons=30, activation='tanh'):
	model = Sequential()
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(1, activation=activation))
	return model

def create(n_types, atoms, desc_length, neurons=30, activation='tanh'):
	submodels = []
	for i in range(n_types):
		submodels.append(
			create_submodel(neurons=neurons, activation=activation))

	inputs = []
	for i in range(len(atoms)):
		inputs.append(Input(shape=(desc_length,)))

	l1 = submodels[0](inputs[0])
	#l2 = submodels[0](inputs[2])
	
	NN = Model(inputs, l1)
	return NN
