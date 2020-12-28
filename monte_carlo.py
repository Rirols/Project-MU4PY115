#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Goal: Monte Carlo all at once to test accuracy of NN model

import numpy as np
import data

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def MC_loop(delta, number, model, dataset, limit,  loop_size):
    """
    

    Parameters
    ----------
    delta : float
        Size of the box in which the particles move randomly.
    number: integer
        Number of particles 
    model : tensorflow.python.keras.engine.functional.Functional
        NN we want to test. Takes particlespositions as input 
        and total energy as output
    dataset: 
        energies and positions
    limit: 
        reduces the size of the dataset
    loop_size : int
        Optimize.

    Returns positions
    """
    

    all_positions, all_energies = data.load(dataset=dataset, limit=limit)
    positions= all_positions[0]
    energy = all_energies[0]
    
    for i in range(loop_size):
        
        #Random position
        try_positions = positions + ((np.random.random((number,3))*2*delta)-delta)
        
        #Energy of the new position
        descriptors = data.compute_desc(positions, dataset='zundel', 
                                       soap_params=None, parallelize=True)
        descriptors=convert_to_inputs(descriptors)
        try_energy=model.fit(descriptors)
    
        diff_E = energy - try_energy

        if diff_E < 0 : 
            energy = try_energy
            positions = try_positions
            
        elif np.exp(-delta * (diff_E)) >= np.random.random():
            energy = try_energy
            positions = try_positions
        else:
            pass
    return positions