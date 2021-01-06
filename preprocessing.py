#!/usr/bin/env python3

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

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

def generate_fitted_transformers(atoms, descriptors, trans_type, trans_params=None):
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

def generate_pca_transformers(atoms, train_desc, pca_params):
    pcas = generate_fitted_transformers(atoms, train_desc, PCA)

    n_components_list = []
    for i in range(len(pcas)):
        n_components_list.append(
            np.where(
                np.cumsum(pcas[i].explained_variance_ratio_) > pca_params['variance'])[0][0]
        )
    n_components = max(n_components_list)

    return generate_fitted_transformers(
        atoms, train_desc, PCA, { 'n_components': n_components }
    )
    
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
        'transformers': transformers, 'result': result 
    })
    return result

def transform_sets(atoms, sets, transformers):
    result = []
    for i in range(len(sets)):
        result.append(transform_set(atoms, sets[i], transformers))
    return result

def scale_energies(sets, scaler):
    transforms = [scaler.transform] * (len(sets) - 1)
    transforms.insert(0, scaler.fit_transform)
    return [transforms[i](sets[i]) for i in range(len(sets))]

def generate_sets(desc, energies, ratios):
    train_size = int(ratios[0] * np.shape(desc)[0])
    validation_size = int(ratios[1] * np.shape(desc)[0])
    cumul_size = train_size + validation_size

    X_train, y_train = desc[:train_size], energies[:train_size]
    X_validation, y_validation = desc[train_size:cumul_size], energies[train_size:cumul_size]
    X_test, y_test = desc[cumul_size:], energies[cumul_size:]

    y_train = y_train.reshape(-1, 1)
    y_validation = y_validation.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_validation, y_validation, X_test, y_test
