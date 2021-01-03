#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow
#import warnings
#warnings.filterwarnings('ignore', message='Numerical issues were encountered')

import data
import preprocessing
import NN
import numpy as np
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics, callbacks
from sklearn.preprocessing import StandardScaler
import monte_carlo

params = {
    'dataset': 'zundel',
    'dataset_size_limit': 100000,
    'atoms': np.array([0,0,1,1,1,1,1]),
    'soap': {
        'sigma': 1, #initial: 0.01
        'nmax': 2,  #3
        'lmax': 5,  #3
        'rcut': 9   #7
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    'pca': {
        'n_components': None,
        'svd_solver': 'auto'
    },
    'scalers': {
        'desc_scaler': StandardScaler(),
        'energies_scaler': StandardScaler()
    },
    'train_set_size_ratio': 0.6,
    'validation_set_size_ratio': 0.2,
    'submodel': {
        'hidden_layers': {
            'units': 30,
            'activation': 'tanh',
            'use_bias': True,
            'kernel_initializer': 'GlorotUniform',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None
        },
        'output_layer': {
            'activation': 'linear',
            'use_bias': True,
            'kernel_initializer': 'GlorotUniform',
            'kernel_regularizer': None,
            'bias_regularizer': 'l1',
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
            callbacks.EarlyStopping(monitor='loss', patience=10), 
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
print('Computing descriptors...')
descriptors, energies = data.load_and_compute(
    dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

#print('Applying PCA...')
#descriptors = preprocessing.pca(
#    atoms=params['atoms'],
#    descriptors=descriptors,
#    params=params['pca']
#)

train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])
validation_size = int(params['validation_set_size_ratio'] * np.shape(descriptors)[0])

print('Generating train, validation and test sets...')
X_train, y_train, X_validation, y_validation, X_test, y_test = preprocessing.generate_scaled_sets(
        atoms=params['atoms'],
        desc=descriptors,
        energies=energies,
        ratios=(params['train_set_size_ratio'], params['validation_set_size_ratio']),
        pca_params=params['pca'],
        desc_scaler=params['scalers']['desc_scaler'],
        energies_scaler=params['scalers']['energies_scaler'])

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

X_train = convert_to_inputs(X_train)
X_validation = convert_to_inputs(X_validation)
X_test = convert_to_inputs(X_test)

# Create model and train it
print('Creating neural network...')
model = NN.create(
    atoms=params['atoms'],
    desc_length=np.shape(X_train[0])[1],
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

# Calculate and print scores
scores = model.evaluate(X_test, y_test, verbose=0)
print()
print('Test loss:', scores[0])
print('MSE:', scores[1])
print('Number of epochs run:', len(history.history['loss']))

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_test=params['scalers']['energies_scaler'].inverse_transform(y_test)
y_test_pred=params['scalers']['energies_scaler'].inverse_transform(y_test_pred)
y_train_pred=params['scalers']['energies_scaler'].inverse_transform(y_train_pred)

plt.plot(y_test, y_test_pred[-1], '.')
#plt.plot(y_train, y_train_pred[-1], '.')
plt.plot(y_test, y_test)
plt.xlabel('True value of energy')
plt.ylabel('Predicted value')
plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.ylabel('model loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()

"""
max_metrics_name_length = len(max(model.metrics_names, key=len))
print('\n')
print(' Scores '.center(max_metrics_name_length + 12, '='))
line = '{:<%i} : {:.3e}' % max_metrics_name_length
for i in range(len(model.metrics_names)):
    print(line.format(model.metrics_names[i], scores[i]))
"""

MC_pos, taux = monte_carlo.MC_loop(params, model)
