#!/usr/bin/env python3

from collections import Counter
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Add
from keras import losses, optimizers

def create_submodel(neurons=30, activation='tanh', optimizer=optimizers.Adam()):
	model = Sequential()
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(1, activation=activation))

	model.compile(
		optimizer=optimizer,
		loss=losses.categorical_crossentropy,
		metrics=['accuracy']
	)

	return model

def create(atoms, desc_length, neurons=30, activation='tanh', optimizer=optimizers.Adam()):
	n_types = len(Counter(atoms).keys())
	submodels = []
	for i in range(n_types):
		submodels.append(
			create_submodel(neurons=neurons, activation=activation, optimizer=optimizer))

	inputs = []
	for i in range(len(atoms)):
		inputs.append(Input(shape=(desc_length,)))

	layers = []
	for i in range(len(atoms)):
		layers.append(submodels[atoms[i]](inputs[i]))
	layers.append(Add()(layers))

	model = Model(inputs, layers)

	model.compile(
		optimizer=optimizer,
		loss=losses.categorical_crossentropy,
		metrics=['accuracy']
	)

	return model
