#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow

import data
import NN
import numpy as np
from keras import losses, optimizers

params = {
	'dataset': 'zundel',
	'dataset_size_limit': 10000,
	'soap': {
		'sigma': 0.01,
		'nmax': 3, 
		'lmax': 3,
		'rcut': 5
	},
	'train_set_size_ratio': 0.8,
	'submodel': {
		'hidden_layers': {
			'units': 30,
			'activation': 'tanh',
			'use_bias': True,
			'kernel_initializer': 'glorot_uniform',
			'kernel_regularizer': None,
			'bias_regularizer': None,
			'activity_regularizer': None,
			'kernel_constraint': None,
			'bias_constraint': None
		},
		'output_layer': {
			'activation': 'tanh',
			'use_bias': True,
			'kernel_initializer': 'glorot_uniform',
			'kernel_regularizer': None,
			'bias_regularizer': None,
			'activity_regularizer': None,
			'kernel_constraint': None,
			'bias_constraint': None
		},
		'compilation': {
			'optimizer': optimizers.Adam(
				learning_rate=0.001,
				beta_1=0.9,
				beta_2=0.999,
				epsilon=1e-07,
				amsgrad=False
			),
			'loss': losses.CategoricalCrossentropy(
				from_logits=False,
				label_smoothing=0,
				reduction='auto'
			),
			'metrics': ['accuracy'],
			'loss_weights': None,
			'sample_weight_mode': None,
			'weighted_metrics': None,
			'target_tensors': None
		}
	},
	'model': {
		'compilation': {
			'optimizer': optimizers.Adam(
				learning_rate=0.001,
				beta_1=0.9,
				beta_2=0.999,
				epsilon=1e-07,
				amsgrad=False
			),
			'loss': losses.CategoricalCrossentropy(
				from_logits=False,
				label_smoothing=0,
				reduction='auto'
			),
			'metrics': ['accuracy'],
			'loss_weights': None,
			'sample_weight_mode': None,
			'weighted_metrics': None,
			'target_tensors': None
		}
	},
	'fit': {
		'batch_size': 64,
		'epochs': 2,
		'shuffle': True,
		'class_weight': None,
		'sample_weight': None,
		'initial_epoch': 0,
		'steps_per_epoch': None,
		'validation_steps': None
	}
}

# Load dataset and compute descriptors
descriptors, energies = data.load_and_compute(
	dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])

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
model = NN.create(
	atoms=[0,0,1,1,1,1,1],
	desc_length=np.shape(descriptors)[2],
	comp_params=params['model']['compilation'],
	sub_hidden_layers_params=params['submodel']['hidden_layers'],
	sub_output_layer_params=params['submodel']['output_layer'],
	sub_comp_params=params['submodel']['compilation']
)

model.fit(
	X_train,
	y_train,
	validation_data=(X_test, y_test),
	**params['fit']
)

# Calculate and print scores
scores = model.evaluate(X_test, y_test, verbose=0)

max_metrics_name_length = len(max(model.metrics_names, key=len))
print('\n')
print(' Scores '.center(max_metrics_name_length + 12, '='))
line = '{:<%i} : {:.3e}' % max_metrics_name_length
for i in range(len(model.metrics_names)):
	print(line.format(model.metrics_names[i], scores[i]))

