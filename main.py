#!/usr/bin/env python3

import pickle

energies = pickle.load(open('./zundel_100K_energy', 'rb'))
pos = pickle.load(open('./zundel_100K_pos', 'rb'))

print(energies)
print(pos)
