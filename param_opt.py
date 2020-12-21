#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

import data
import NN
import numpy as np
from keras import losses, optimizers, metrics, callbacks
from sklearn.preprocessing import scale

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

space={'learning_rate': hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.1])}

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def create_set(X, y, indexes):
    X_data, y_data = X[indexes[0]:indexes[1]], y[indexes[0]:indexes[1]]
    X_data = convert_to_inputs(X_data)
    for i in range(len(X_data)):
        X_data[i] = scale(X_data[i])
    y_data = scale(y_data)
    return X_data, y_data

def objective(target):
    
    params = {
        'dataset': 'zundel',
        'dataset_size_limit': 100000,
        'soap': {
            'sigma': 0.01,
            'nmax': 3, 
            'lmax': 3,
            'rcut': 5
            },
        'train_set_size_ratio': 0.6,
        'validation_set_size_ratio': 0.2,
        'submodel': {
            'hidden_layers': {
                'units': 30,
                'activation': 'tanh',
                'use_bias': True,
                'kernel_initializer': None,
                'kernel_regularizer': 'l1',
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
                },
            'output_layer': {
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': 'glorot_uniform',
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
                },
            },
        'model': {
            'compilation': {
                'optimizer': optimizers.Adam(
                    learning_rate=target['learning_rate']
                    ),
                'loss': losses.MeanSquaredError(),
                'metrics': metrics.MeanSquaredError(),
                }
            },
        'fit': {
            'batch_size': 32,
            'callbacks' : callbacks.EarlyStopping(monitor='loss', patience=3),
            'epochs': 100
            }
}   

# Load dataset and compute descriptors
    descriptors, energies = data.load_and_compute(
    dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

    train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])
    validation_size = int(params['validation_set_size_ratio'] * np.shape(descriptors)[0])
    
    X_train, y_train = create_set(descriptors, energies, (0, train_size))
    X_validation, y_validation = create_set(
        descriptors, energies, (train_size, train_size + validation_size))
    X_test, y_test = create_set(descriptors, energies, (train_size + validation_size, -1))
# Create model and train it
    model = NN.create(
    atoms=[0,0,1,1,1,1,1],
    desc_length=np.shape(descriptors)[2],
    comp_params=params['model']['compilation'],
    sub_hidden_layers_params=params['submodel']['hidden_layers'],
    sub_output_layer_params=params['submodel']['output_layer']
)

    history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    verbose=0,
    **params['fit']
    )
    
    loss=history.history['val_loss'][-1]
    
    return {'loss': loss, 'status': STATUS_OK }


trials=Trials()
best = fmin(objective,
            space, 
            algo=tpe.suggest, 
            trials=trials, 
            max_evals=100
            )

print (best)
print (trials.best_trial)
