#!/usr/bin/env python3

from os.path import join
import copy
import multiprocessing
import pickle
import numpy as np
from ase import Atoms
from ase.build import bulk
from dscribe.descriptors import SOAP

def load_pickle(dataset, args):
    pos = pickle.load(open(args['data']['pos'], 'rb'))
    energies = pickle.load(open(args['data']['energies'], 'rb'))

    if (dataset == 'zundel_100k'):
        pos = pos[:-1]
        energies = energies[1:]

    step = args['thinning_step']
    pos, energies = pos[::step], energies[::step]

    return pos, energies

def load_txt(dataset, args):
    pos = np.loadtxt(args['data']['pos'], unpack=False)
    energies = np.loadtxt(args['data']['energies'], unpack=True)

    energies = energies[args['en_column']]
    n_config = np.shape(energies)[0]
    pos = np.reshape(pos, (n_config, args['atoms'], 3))

    step = args['thinning_step']
    pos, energies = pos[::step], energies[::step]

    return pos, energies

data_path='data'

datasets = {
    'zundel_100k': {
        'loading': {
            'func': load_pickle,
            'args': {
                'data': {
                    'pos': join(data_path, 'zundel_100K_pos'),
                    'energies': join(data_path, 'zundel_100K_energy')
                },
                'thinning_step': 5,
            }
        },
        'atoms': 7,
        'symbols': 'O2H5',
        'list': np.array([0, 0, 1, 1, 1, 1, 1]),
        'ase': {},
        'soap': {
            'species': ['H', 'O'],
            'periodic': False
        }
    },
    'CO2': {
        'loading': {
            'func': load_txt,
            'args': {
                'data': {
                    'pos': join(data_path, 'TRAJEC_db'),
                    'energies': join(data_path, 'ENERGIES_db')
                },
                'thinning_step': 1,
                'en_column': 3,
                'atoms': 96
            }
        },
        'atoms': 96,
        'symbols': 'C32O64',
        'list': np.array([0]*32 + [1]*64),
        'ase': {
            'cell': [9.8] * 3
        },
        'soap': {
            'species': ['C', 'O'],
            'periodic': True
        }
    }
}

def get_atoms_list(dataset='zundel_100k'):
    return np.copy(datasets[dataset]['list'])

def get_symbols(dataset='zundel_100k'):
    return datasets[dataset]['symbols']

def get_n_atoms(dataset='zundel_100k'):
    return datasets[dataset]['atoms']

def load_pos(dataset='zundel_100k', limit=None):
    params = datasets[dataset]

    pos, energies = params['loading']['func'](
        dataset, params['loading']['args']
    )

    if limit != None:
        pos, energies = pos[:limit], energies[:limit]

    return pos, energies

def load(dataset='zundel_100k', limit=None):
    params = datasets[dataset]

    pos, energies = load_pos(dataset, limit)
    
    tot_time = np.shape(pos)[0]
    molecs = np.empty(tot_time, dtype=object)
    for t in range(tot_time):
        molecs[t] = Atoms(params['symbols'], positions=pos[t], **params['ase'])

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
