#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Goal: Monte Carlo all at once to test accuracy of NN model
#Issue: not yet entirely general

import numpy as np
import data
import pickle
from ase import Atoms

def convert_to_inputs(raw):
    raw_t = raw.transpose(1, 0, 2)
    X = []
    for i in range(np.shape(raw_t)[0]):
        X.append(raw_t[i])
    return X

def MC_loop(parameters, model):
    
    delta = parameters['Monte-Carlo']['box_size']
    kb = 1.381e-23*2.294e+17 # conversion Hartree/K# conversion Hartree/K
    beta = 1/(kb*parameters['Monte-Carlo']['temperature'])
    acceptance = 0
    
    pos_history = np.empty((parameters['Monte-Carlo']['Number_of_steps'],7, 3))
    energies_history = np.empty(parameters['Monte-Carlo']['Number_of_steps'])
    

    all_positions = pickle.load(open('./data/zundel_100K_pos', 'rb'))
    all_energies= pickle.load(open('./data/zundel_100K_energy', 'rb'))
    
    step = 100
    all_positions, all_energies = all_positions[::step], all_energies[1::step]
    tot_time = np.shape(all_positions)[0]
    molecs = np.empty(tot_time, dtype=object)
    for t in range(tot_time):
        molecs[t] = Atoms('O2H5', positions=all_positions[t])
    
    init_time = np.random.randint(np.shape(all_positions)[0]-1)
    positions= all_positions[init_time]
    energy = all_energies[init_time + 1]
        
    for i in range(parameters['Monte-Carlo']['Number_of_steps']):
        
        #Random position
        try_positions = positions + np.random.random((7,3))*delta-delta/2
        
        #convert random position into input
        descriptor = #descriptor of config
        #PCA + scaling
        
        descriptor = convert_to_inputs(descriptor)
        print(np.shape(descriptor))
        
        try_energy=model.predict(descriptor)
    
        diff_E = energy - parameters['scalers']['energies_scaler'].inverse_transform(try_energy)
        
        print(np.shape(energy))
        print(np.shape(try_energy))
        print(np.shape(diff_E))

        if diff_E < 0 : 
            energy = try_energy
            positions = try_positions
            acceptance += 1
            pos_history[i] = positions
            energies_history[i] = energy
            
        elif np.exp(-beta * (diff_E)) >= np.random.random():
            energy = try_energy
            positions = try_positions
            pos_history[i] = positions
            energies_history[i] = energy
            
        else:
            pos_history[i] = positions
            energies_history[i] = energy
        
    acceptance_rate = acceptance/parameters['Monte-Carlo']['Number_of_steps']
    return positions, acceptance_rate