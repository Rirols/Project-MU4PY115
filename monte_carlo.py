#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Goal: Monte Carlo all at once to test accuracy of NN model

import numpy as np
import data
import pickle
from ase import Atoms
from sklearn.decomposition import PCA

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def MC_loop(parameters, model):
    
    #set up constants
    delta = parameters['Monte-Carlo']['box_size']
    kb = 1.381e-23*2.294e+17 # conversion Hartree/K# conversion Hartree/K
    beta = 1/(kb*parameters['Monte-Carlo']['temperature'])
    acceptance = 0
    
    #load MD data
    all_positions, all_energies = data.load_pos(parameters['dataset'], limit=10000)

    #Set up random initial configuration
    init_time = np.random.randint(np.shape(all_positions)[0]-1)
    positions= all_positions[init_time]
    energy = all_energies[init_time + 1]

    molecs = np.empty(1, dtype=object)
    molecs[0] = Atoms('O2H5', positions=positions)
    
    #Set up lists to record evolution
    positions_history=np.empty((parameters['Monte-Carlo']['Number_of_steps'],7,3))
    energy_history = np.empty(parameters['Monte-Carlo']['Number_of_steps'])
        
    for i in range(parameters['Monte-Carlo']['Number_of_steps']):
        
        #Random position
        try_positions = positions + np.random.random((7,3))*delta-delta/2
        
        #convert random position into input
        descriptor = data.compute_desc(molecs,
            dataset=parameters['dataset'], 
            soap_params=parameters['soap'])

        #PCA + scaling
        n_dim = model.get_layer(index=0).input_shape[0][1] #dimension of preprocessed input
        pca = PCA(n_components=n_dim)
        print(np.shape(descriptor))
        descriptor=pca.fit(descriptor)
        descriptor=parameters['scalers']['desc_scaler_type'].transform(descriptor)
        
        descriptor = convert_to_inputs(descriptor)
        
        try_energy=model.predict(descriptor)
    
        diff_E = energy - parameters['scalers']['energies_scaler'].inverse_transform(try_energy)

        if diff_E < 0 : 
            energy = try_energy
            positions = try_positions
            acceptance += 1
            
        elif np.exp(-beta * (diff_E)) >= np.random.random():
            energy = try_energy
            positions = try_positions
            acceptance += 1
        else:
            pass
        
        positions_history[i]=positions
        energy_history[i]=energy
        
    acceptance_rate = acceptance/parameters['Monte-Carlo']['Number_of_steps']
    
    return acceptance_rate, positions_history, energy_history