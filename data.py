#!/usr/bin/env python3

from os.path import join
import copy
import multiprocessing
import pickle
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

data_path='data'

datasets = {
    'zundel_100k': {
        'atoms': 7,
        'symbols': 'O2H5',
        'list': np.array([0, 0, 1, 1, 1, 1, 1]),
        'data': {
            'pos': join(data_path, 'zundel_100K_pos'),
            'energies': join(data_path, 'zundel_100K_energy'),
            'thinning_step': 100
        },
        'soap': {
            'species': ['H', 'O'],
            'periodic': False
        }
    }
}

def get_atoms_list(dataset='zundel_100k'):
    return np.copy(datasets[dataset]['list'])

def load_pos(dataset='zundel_100k', limit=None):
    params = datasets[dataset]

    pos = pickle.load(open(params['data']['pos'], 'rb'))
    energies = pickle.load(open(params['data']['energies'], 'rb'))

    if (dataset == 'zundel_100k'):
        pos = pos[:-1]
        energies = energies[1:]

    #step = params['data']['thinning_step']
    #pos, energies = pos[::step], energies[::step]

    if limit != None:
        pos, energies = pos[:limit], energies[:limit]

    return pos, energies

def load(dataset='zundel_100k', limit=None):
    params = datasets[dataset]

    pos, energies = load_pos(dataset, limit)
    
    tot_time = np.shape(pos)[0]
    molecs = np.empty(tot_time, dtype=object)
    for t in range(tot_time):
        molecs[t] = Atoms(params['symbols'], positions=pos[t])

    return molecs, energies

def compute_desc(molecs, dataset='zundel_100k', soap_params=None, parallelize=True):
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

def load_and_compute(dataset='zundel_100k', limit=None, soap_params=None, parallelize=True):
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
