#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Goal: Monte Carlo all at once to test accuracy of NN model

import numpy as np
import data
from ase import Atoms
import preprocessing

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def MC_loop(parameters, model, pcas, scalers):
    print("Preparing Monte-Carlo simulation...")
    #set up constants
    delta = parameters['Monte-Carlo']['box_size']
    kb = 1.381e-23
    beta = 1/(kb*parameters['Monte-Carlo']['temperature'])
    acceptance = []
    
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
    
    print("Computing Monte-Carlo simulation...(this might take a while)")
    for i in range(parameters['Monte-Carlo']['Number_of_steps']):
        
        #Random position
        try_positions = positions + np.random.random((7,3))*delta-delta/2
        
        #convert random position into input
        descriptor = data.compute_desc(molecs,
            dataset=parameters['dataset'], 
            soap_params=parameters['soap'])

        #PCA + scaling
        descriptor = preprocessing.transform_set(
            atoms=data.get_atoms_list(parameters['dataset']),
            descriptors=descriptor,
            transformers=pcas
            )
        
        descriptor = preprocessing.transform_set(
            atoms=data.get_atoms_list(parameters['dataset']),
            descriptors=descriptor,
            transformers=scalers
            )
        descriptor = convert_to_inputs(descriptor)
        
        try_energy=model.predict(descriptor)
    
        diff_E = energy - parameters['scalers']['energies_scaler'].inverse_transform(try_energy)
        #diff_E *= 4.35974434e-18 #conversion Hartree => J
        
        if diff_E < 0 : 
            energy = try_energy
            positions = try_positions
            acceptance.append(1)
            
        elif np.exp(-beta * diff_E) >= np.random.random():
            energy = try_energy
            positions = try_positions
            acceptance.append(1)
            
        else:
            acceptance.append(0)
            pass
        
        positions_history[i]=positions
        energy_history[i]=energy
        
    print("acceptance rate=", np.mean(acceptance))

    return np.mean(acceptance), positions_history, energy_history