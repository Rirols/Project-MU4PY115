#!/usr/bin/env python3

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def iterate(atoms, descriptors, func, args):
    """
    Iterate over all sub similar elements of an array of descriptors
    and execute func
    """

    n_config, n_elts, n_dim = np.shape(descriptors)

    n_types = len(np.unique(atoms))
    for i in range(n_types):
        # Create a subarray of descriptors with only similar elements
        # and get associated indexes
        subdesc = descriptors[:, atoms == i]
        indexes = np.where(atoms == i)[0]
        n_sub_elts = len(indexes)

        func(
            n_config=n_config,
            n_dim=n_dim,
            n_elts=n_elts,
            n_sub_elts=n_sub_elts,
            subdesc=subdesc,
            indexes=indexes,
            sub_elts_id=i,
            args=args
        )

def create_fitted_transformers(atoms, descriptors, trans_type, trans_params=None):
    """
    Call the fit method of transformers to set them up using a train set
    (transformers can be scalers or PCA instances)
    """

    def func(n_config, n_dim, n_elts, n_sub_elts, subdesc, indexes, sub_elts_id, args):
        if args['trans_params'] == None:
            trans = args['trans_type']()
        else:
            trans = args['trans_type'](**args['trans_params'])
        # Fit transformers on all similar sub elements
        trans.fit(subdesc.reshape(n_config * n_sub_elts, n_dim))
        args['trans_list'].append(trans)

    trans = []
    iterate(atoms, descriptors, func, {
        'trans_type': trans_type, 'trans_params': trans_params, 'trans_list': trans
    })
    return trans
    
def transform_set(atoms, descriptors, transformers):
    def func(n_config, n_dim, n_elts, n_sub_elts, subdesc, indexes, sub_elts_id, args):
        args['result'][:, indexes] = np.reshape(
            args['transformers'][sub_elts_id].transform(
                subdesc.reshape((n_config * n_sub_elts, n_dim))),
            (n_config, n_sub_elts, np.shape(args['result'])[2])
        )

    shape = np.shape(descriptors)
    if (type(transformers[0]) is PCA):
        shape = (shape[0], shape[1], transformers[0].n_components_)
    result = np.empty(shape)

    iterate(atoms, descriptors, func, {
        'transformers': transformers, 'result': result })
    return result

def apply_pca_on_sets(atoms, sets, pca_params):
    pcas = create_fitted_transformers(
        atoms=atoms,
        descriptors=sets[0],
        trans_type=PCA,
        trans_params=pca_params
    )

    result = []
    for i in range(len(sets)):
        n_config, n_elts, n_dim = np.shape(sets[i])
        n_dim = pcas[0].n_components_
        result.append(transform_set(atoms, sets[i], pcas))
    return result

def scale_sets(atoms, sets, scaler_type):
    scalers = create_fitted_transformers(
        atoms=atoms,
        descriptors=sets[0],
        trans_type=scaler_type
    )

    result = []
    for i in range(len(sets)):
        result.append(transform_set(atoms, sets[i], scalers))
    return result

def generate_scaled_sets(
    atoms, desc, energies, ratios, pca_params, desc_scaler_type, energies_scaler):
    train_size = int(ratios[0] * np.shape(desc)[0])
    validation_size = int(ratios[1] * np.shape(desc)[0])
    cumul_size = train_size + validation_size

    X_train, y_train = desc[:train_size], energies[:train_size]
    X_validation, y_validation = desc[train_size:cumul_size], energies[train_size:cumul_size]
    X_test, y_test = desc[cumul_size:], energies[cumul_size:]

    #X_train, X_validation, X_test = apply_pca_on_sets(
    #    atoms, [X_train, X_validation, X_test], pca_params
    #)

    X_train, X_validation, X_test = scale_sets(
        atoms, [X_train, X_validation, X_test], desc_scaler_type
    ) 

    y_train = energies_scaler.fit_transform(y_train.reshape(-1,1))
    y_validation = energies_scaler.transform(y_validation.reshape(-1,1))
    y_test = energies_scaler.transform(y_test.reshape(-1,1))

    return X_train, y_train, X_validation, y_validation, X_test, y_test
