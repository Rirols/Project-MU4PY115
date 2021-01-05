#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable annoying warnings from tensorflow

import data
import preprocessing
import NN
import numpy as np
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import monte_carlo

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
            'metrics': [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()],
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
print('Computing descriptors...')
descriptors, energies = data.load_and_compute(
    dataset=params['dataset'], soap_params=params['soap'], limit=params['dataset_size_limit'])

train_size = int(params['train_set_size_ratio'] * np.shape(descriptors)[0])
validation_size = int(params['validation_set_size_ratio'] * np.shape(descriptors)[0])

print('Generating train, validation and test sets...')
X_train, y_train, X_validation, y_validation, X_test, y_test = preprocessing.generate_scaled_sets(
    atoms=data.get_atoms_list(params['dataset']),
    desc=descriptors,
    energies=energies,
    ratios=(params['train_set_size_ratio'], params['validation_set_size_ratio']),
    pca_params=params['pca'],
    desc_scaler_type=params['scalers']['desc_scaler_type'],
    energies_scaler=params['scalers']['energies_scaler']
)

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
    atoms=data.get_atoms_list(params['dataset']),
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

print('Number of epochs run:', len(history.history['loss']))

# Calculate and print scores
print('Evaluating model performance on test set...')
scores = model.evaluate(X_test, y_test, verbose=0)

max_metrics_name_length = len(max(model.metrics_names, key=len))
print()
print(' Scores '.center(max_metrics_name_length + 13, '='))
line = '{:<%i} : {:.4e}' % max_metrics_name_length
for i in range(len(model.metrics_names)):
    print(line.format(model.metrics_names[i], scores[i]))
print()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train = params['scalers']['energies_scaler'].inverse_transform(y_train)
y_test = params['scalers']['energies_scaler'].inverse_transform(y_test)
y_train_pred = params['scalers']['energies_scaler'].inverse_transform(y_train_pred)
y_test_pred = params['scalers']['energies_scaler'].inverse_transform(y_test_pred)

plt.plot(y_test, y_test_pred, '+')
plt.plot(y_train, y_train_pred, '+')
plt.plot(y_test, y_test)
plt.xlabel('True value of energy')
plt.ylabel('Predicted value')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

acceptance_rate, positions_history, energy_history = monte_carlo.MC_loop(params, model)
