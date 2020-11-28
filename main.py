#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow

import data
import NN
import numpy as np

soap_params = {
	'sigma': 0.01,
	'nmax': 3, 
	'lmax': 3,
	'rcut': 5
}

limit = 10000
train_percent = 0.8

# Load dataset and compute descriptors
descriptors, energies = data.load_and_compute(
	dataset='zundel', soap_params=soap_params, limit=limit)

train_size = int(train_percent * np.shape(descriptors)[0])

X_train, y_train = descriptors[:train_size], energies[:train_size]
X_test, y_test = descriptors[train_size:], energies[train_size:]

def convert_to_inputs(raw):
	raw_t = raw.transpose(1, 0, 2)
	X = []
	for i in range(np.shape(raw_t)[0]):
		X.append(raw_t[i])
	return X

X_train = convert_to_inputs(X_train)
X_test = convert_to_inputs(X_test)

# Create model and train it
model = NN.create(atoms=[0,0,1,1,1,1,1], desc_length=np.shape(descriptors)[2])
#print(model.summary())

model.fit(
	X_train,
	y_train,
	batch_size=64,
	epochs=2,
	validation_data=(X_test, y_test)	
)
