#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#hyperopt optimization: minimize loss


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow

import data
import NN
import main
import numpy as np
from keras import losses, optimizers, metrics, callbacks
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {'learning_rate': hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.1])}

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
            'kernel_regularizer': None,
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
                learning_rate=['learning_rate']
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

def minimized_f(parameters):
    # Load dataset and compute descriptors
    print('Computing descriptors...')
    descriptors, energies = data.load_and_compute(
        dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

    train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])
    validation_size = int(params['validation_set_size_ratio'] * np.shape(descriptors)[0])

    print('Generating train, validation and test sets...')
    X_train, y_train = main.create_set(descriptors, energies, (0, train_size))
    X_validation, y_validation = main.create_set(
        descriptors, energies, (train_size, train_size + validation_size))
    X_test, y_test = main.create_set(descriptors, energies, (train_size + validation_size, -1))

    # Create model and train it
    print('Creating neural network...')
    model = NN.create(
        atoms=[0,0,1,1,1,1,1],
        desc_length=np.shape(descriptors)[2],
        comp_params=params['model']['compilation'],
        sub_hidden_layers_params=params['submodel']['hidden_layers'],
        sub_output_layer_params=params['submodel']['output_layer']
        )

    print('Training network (this might take a while)...')
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

result=fmin(
    fn=minimized_f(params),
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=100
    )

print (result)
print (trials.best_trial)