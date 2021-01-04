#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize the parameters used in our NN with hyperopt: gice the possible values
for the target parameters in space and call them in params before running
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

import data
import preprocessing
import NN
import numpy as np
from keras import losses, optimizers, metrics, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

space={'learning_rate': hp.choice('learning_rate', [0.001, 0.1, 1]),
       'batch_size': hp.choice('batch_size', [16, 32, 48, 64]),
       'patience_1': hp.choice('patience_1', [3, 5, 7, 10]),
       'patience_2': hp.choice('patience_2', [3, 5, 7, 10])
}
"""
        'rcut': hp.choice('rcut',[8, 9, 10, 11])
        'sigma': hp.choice('sigma', [0.1, 1, 3, 5]),
        'nmax': hp.choice('nmax', [2, 3, 4, 5]),
        'lmax': hp.choice('lmax', [1, 2, 3, 4, 5])
        'desc_scaler_type': hp.choice('desc_scaler_type', [StandardScaler, MinMaxScaler]),
        'energies_scaler': hp.choice('energies_scaler', [StandardScaler, MinMaxScaler])
        'kernel_regularizer': hp.choice('kernel_regularizer', [None, 'l1', 'l2', 'l1_l2']),
        'bias_regularizer': hp.choice('bias_regularizer', [None, 'l1', 'l2', 'l1_l2']),
        'activity_regularizer': hp.choice('activity_regularizer', [None,'l1', 'l2', 'l1_l2']),
        'kernel_initializer': hp.choice('kernel_initializer', 
                [None, 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'Zeros',
                 'Ones', 'GlorotNormal', 'GlorotUniform', 'Identity', 'Orthogonal',
                 'Constant', 'VarianceScaling']),
       """
def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def objective(target):
    
    params = {
        'dataset': 'zundel_100k',
        'dataset_size_limit': 100000,
        'soap': {
            # https://singroup.github.io/dscribe/latest/tutorials/soap.html
            'sigma': 1, #initial: 0.01
            'nmax': 5,  #3
            'lmax': 2,  #3
            'rcut': 9   #7
            },
        'pca': {
            # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            'variance': 0.999999
            },
        'scalers': {
            # https://scikit-learn.org/stable/modules/preprocessing.html
            'desc_scaler_type': StandardScaler,
            'energies_scaler': MinMaxScaler()
            },
        'train_set_size_ratio': 0.6,
        'validation_set_size_ratio': 0.2,
        'submodel': {
            # https://keras.io/guides/sequential_model/
            'hidden_layers': {
                'units': 30,
                'activation': 'tanh',
                'use_bias': True,
                'kernel_initializer': 'GlorotUniform',
                'kernel_regularizer': None,
                'bias_regularizer': 'l2',
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
                },
            'output_layer': {
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': 'Zeros',
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
                    learning_rate=0.001
                    ),
                'loss': losses.MeanSquaredError(),
                'metrics': metrics.MeanSquaredError(),
                }
            },
        'fit': {
            'batch_size': 32,
            'epochs': 100,
            'callbacks' : [
                callbacks.EarlyStopping(monitor='loss', patience=7), 
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
                ]
            },
        'Monte-Carlo': {
            'temperature': 100,
            'Number_of_steps': 100000,
            'box_size': 2,
            }
        }

    # Load dataset and compute descriptors

    descriptors, energies = data.load_and_compute(
        dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

    train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])
    validation_size = int(params['validation_set_size_ratio'] * np.shape(descriptors)[0])

    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocessing.generate_scaled_sets(
            atoms=data.get_atoms_list(params['dataset']),
            desc=descriptors,
            energies=energies,
            ratios=(params['train_set_size_ratio'], params['validation_set_size_ratio']),
            pca_params=params['pca'],
            desc_scaler_type=params['scalers']['desc_scaler_type'],
            energies_scaler=params['scalers']['energies_scaler'])

    X_train = convert_to_inputs(X_train)
    X_validation = convert_to_inputs(X_validation)
    X_test = convert_to_inputs(X_test)

    # Create model and train it
    model = NN.create(
        atoms=data.get_atoms_list(params['dataset']),
        desc_length=np.shape(X_train[0])[1],
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

print ("Best result:", best)
print (trials.best_trial)
