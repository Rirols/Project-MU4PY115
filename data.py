#!/usr/bin/env python3

from os.path import join
import copy
import pickle
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

data_path='./data'

datasets = {
	'zundel': {
		'atoms': [8, 8, 1, 1, 1, 1, 1],
		'data': {
			'pos': join(data_path, 'zundel_100K_pos'),
			'energies': join(data_path, 'zundel_100K_energy')
		},
		'soap': {
			'species': [1, 8],
			'periodic': False,
		}
	}
}

def load(dataset='zundel', limit=None, soap_params=None):
	"""
	Load a specific dataset and compute descriptors

	params:
		- dataset: the name of the dataset
		- limit: reduce the size of the dataset
		- soap_params: an object containing custom soap parameters
		  (species and periodic are already set)
	"""

	params = copy.deepcopy(datasets[dataset])
	if soap_params != None:
		params['soap'].update(soap_params)

	pos = pickle.load(open(params['data']['pos'], 'rb'))
	energies = pickle.load(open(params['data']['energies'], 'rb'))

	if limit != None:
		pos, energies = pos[:limit], energies[:limit]

	tot_time = np.shape(pos)[0]
	molecs = np.empty(tot_time, dtype=object)
	for t in range(tot_time):
		molecs[t] = Atoms(params['atoms'], positions=pos[t])

	soap = SOAP(**params['soap'])
	descriptors = soap.create(molecs)

	return descriptors, energies

