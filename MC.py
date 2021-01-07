#!/usr/bin/env python3

import data
import preprocessing
import numpy as np
from ase import Atoms
from math import exp
from random import random
from tqdm import tqdm

def simulate(
    init_pos, init_en, model, pcas, desc_scalers, en_scaler, soap,
    steps=100000, delta=0.04, T=100, dataset='zundel_100k'):
    """
    Launch a Monte-Carlo simulation using a trained keras model
    """

    hartree = 4.3597443419e-18
    kb = 1.381e-23 / hartree
    beta = 1 / (kb * T)

    n_atoms = data.get_n_atoms(dataset)
    atoms = data.get_atoms_list(dataset)
    symbols = data.get_symbols(dataset)

    acceptance = 0
    cur_pos = np.copy(init_pos)
    cur_en = init_en

    pos_history = np.empty((steps, n_atoms, 3))
    en_history = np.empty(steps)

    for i in tqdm(range(steps)):
        dr = np.random.random((n_atoms, 3)) * 2 * delta - delta
        try_pos = np.copy(cur_pos) + dr

        molec = np.empty(1, dtype=object)
        molec[0] = Atoms(symbols, positions=try_pos)

        desc = data.compute_desc(
            molec, dataset=dataset, soap_params=soap
        )

        desc = preprocessing.transform_set(
            atoms=atoms, descriptors=desc, transformers=pcas
        )

        desc = preprocessing.transform_set(
            atoms=atoms, descriptors=desc, transformers=desc_scalers
        )

        desc = preprocessing.convert_to_inputs(desc)

        try_en = model.predict(desc)
        try_en = en_scaler.inverse_transform(try_en)[0, 0]
        
        if try_en < cur_en or exp(-beta * (try_en - cur_en)) >= random():
            acceptance += 1
            cur_pos, cur_en = try_pos, try_en

        pos_history[i] = cur_pos
        en_history[i] = cur_en

    return pos_history, en_history, float(acceptance) / steps

