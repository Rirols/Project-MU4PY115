#!/usr/bin/env python3

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

def pca(atoms, descriptors, params):
    n_config, n_elts, n_dim = np.shape(descriptors)
    result = np.empty((
        n_config, n_elts,
        n_dim if params['n_components'] == None else params['n_components']
    ))

    n_types = len(np.unique(atoms))
    for i in range(n_types):
        subdesc = descriptors[:, atoms == i]
        indexes = np.where(atoms == i)[0]

        n_sub_elts = np.shape(subdesc)[1]

        pca = PCA(**params)
        pca.fit(subdesc.reshape((n_config * n_sub_elts, n_dim)))

        for j in range(n_sub_elts):
            result[:, indexes[j]] = pca.transform(subdesc[:, j])

    return result


def scale(atoms, descriptors, scale_func):
    n_config, n_elts, n_dim = np.shape(descriptors)
    result = np.empty((n_config, n_elts, n_dim))

    n_types = len(np.unique(atoms))
    for i in range(n_types):
        indexes = np.where(atoms == i)[0]
        n_sub_elts = len(indexes)

        # Scale on all similar elements
        result[:, indexes] = np.reshape(
            scale_func(descriptors[:, atoms == i].reshape((n_config * n_sub_elts, n_dim))),
            (n_config, n_sub_elts, n_dim)
        )

    return result


def scale_sets(atoms, sets, scaler):
    result = []
    for i in range(len(sets)):
        result.append(
            scale(atoms, sets[i], scaler.fit_transform if i == 0 else scaler.transform))

    return result


def generate_scaled_sets(
    atoms, desc, energies, ratios, pca_params, desc_scaler, energies_scaler):
    train_size = int(ratios[0] * np.shape(desc)[0])
    validation_size = int(ratios[1] * np.shape(desc)[0])
    cumul_size = train_size + validation_size

    X_train, y_train = desc[:train_size], energies[:train_size]
    X_validation, y_validation = desc[train_size:cumul_size], energies[train_size:cumul_size]
    X_test, y_test = desc[cumul_size:], energies[cumul_size:]

    X_train = pca(atoms, X_train, pca_params)
    X_validation = pca(atoms, X_validation, pca_params)
    X_test = pca(atoms, X_test, pca_params)

    X_train, X_validation, X_test = scale_sets(
        atoms, [X_train, X_validation, X_test], desc_scaler) 

    y_train = energies_scaler.fit_transform(y_train.reshape(-1,1))
    y_validation = energies_scaler.transform(y_validation.reshape(-1,1))
    y_test = energies_scaler.transform(y_test.reshape(-1,1))

    return X_train, y_train, X_validation, y_validation, X_test, y_test
