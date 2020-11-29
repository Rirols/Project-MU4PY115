#!/usr/bin/env python3

from collections import Counter
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Add

def create_submodel(comp_params, neurons=30, activation='tanh'):
	model = Sequential()
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(1, activation=activation))
	model.compile(**comp_params)

	return model

def create(atoms, desc_length, comp_params, sub_comp_params, neurons=30, activation='tanh'):
	n_types = len(Counter(atoms).keys())
	submodels = []
	for i in range(n_types):
		submodels.append(
			create_submodel(sub_comp_params, neurons=neurons, activation=activation))

	inputs = []
	for i in range(len(atoms)):
		inputs.append(Input(shape=(desc_length,)))

	layers = []
	for i in range(len(atoms)):
		layers.append(submodels[atoms[i]](inputs[i]))
	layers.append(Add()(layers))

	model = Model(inputs, layers)
	model.compile(**comp_params)

	return model
