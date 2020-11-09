#!/usr/bin/env python3

import pickle

energies = pickle.load(open('./data/zundel_100K_energy', 'rb'))
pos = pickle.load(open('./data/zundel_100K_pos', 'rb'))

print(energies)
print(pos)
