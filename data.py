#!/usr/bin/env python3

from os.path import join
import copy
import multiprocessing
import pickle
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.preprocessing import scale

data_path='data'

datasets = {
    'zundel': {
        'atoms': 7,
        'symbols': 'O2H5',
        'data': {
            'pos': join(data_path, 'zundel_100K_pos'),
            'energies': join(data_path, 'zundel_100K_energy')
        },
        'soap': {
            'species': ['H', 'O'],
            'periodic': False
        }
    }
}


def load(dataset='zundel', limit=None):
    params = datasets[dataset]
    
    pos = pickle.load(open(params['data']['pos'], 'rb'))
    energies = pickle.load(open(params['data']['energies'], 'rb'))
    pos, energies = pos[::100], energies[::100]
    energies=scale(energies)

    if limit != None:
        pos, energies = pos[:limit], energies[:limit]

    tot_time = np.shape(pos)[0]
    molecs = np.empty(tot_time, dtype=object)
    for t in range(tot_time):
        molecs[t] = Atoms(params['symbols'], positions=pos[t])

    return molecs, energies


def compute_desc(molecs, dataset='zundel', soap_params=None, parallelize=True):
    params = copy.deepcopy(datasets[dataset])
    if soap_params != None:
        params['soap'].update(soap_params)
    
    tot_time = np.shape(molecs)[0]

    soap = SOAP(**params['soap'])
    descriptors = soap.create(molecs,
        positions=[np.arange(params['atoms']) for i in range(tot_time)],
        n_jobs=multiprocessing.cpu_count() if parallelize else 1)

    return np.reshape(descriptors,
        (tot_time, params['atoms'], np.shape(descriptors)[1]))


def load_and_compute(dataset='zundel', limit=None, soap_params=None, parallelize=True):
    """
    Load a specific dataset and compute descriptors

    params:
        - dataset: the name of the dataset
        - limit: reduce the size of the dataset
        - soap_params: an object containing custom soap parameters
          (species and periodic are already set)
    """

    molecs, energies = load(dataset=dataset, limit=limit)
    descriptors = compute_desc(
        molecs, dataset=dataset, soap_params=soap_params, parallelize=parallelize)
    
    return descriptors, energies
