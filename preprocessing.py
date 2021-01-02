#!/usr/bin/env python3

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as SCALE

def pca(atoms, descriptors, params):
    n_config, n_elts, n_dim = np.shape(descriptors)
    result = np.ones((
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

def scale(atoms, descriptors):
    n_config, n_elts, n_dim = np.shape(descriptors)
    result = np.ones((n_config, n_elts, n_dim))

    n_types = len(np.unique(atoms))
    for i in range(n_types):
        indexes = np.where(atoms == i)[0]
        n_sub_elts = len(indexes)

        result[:, indexes] = np.reshape(
            SCALE(descriptors[:, atoms == i].reshape((n_config * n_sub_elts, n_dim))),
            (n_config, n_sub_elts, n_dim)
        )

    return result
