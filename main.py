#!/usr/bin/env python3
import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read
from ase.build import molecule
from ase import Atoms
import pickle

#Import data
energies = pickle.load(open('./data/zundel_100K_energy', 'rb'))
pos = pickle.load(open('./data/zundel_100K_pos', 'rb'))

#SOAP parameters
species = ["H", "O"]
#sigma=
nmax = 3
lmax=8
rcut=5

#Configure SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="outer",
    sparse=False
)

