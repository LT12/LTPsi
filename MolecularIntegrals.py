__author__ = 'larry'

import numpy as np
from atomicParam import *
from cPrimMolInt import primOverlapIntegral, primNuclearAttractionIntegral, primElecRepulInt
from numba import jit


@jit
def overlap_matrix(orbs):
    '''Generate matrix of overlap integrals between each atomic orbital

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        overlapM - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    overlapM = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            overlapM[i, j] = overlap_integral(orbs[i], orbs[j])
            overlapM[j, i] = overlapM[i, j]

    return overlapM


@jit
def kinetic_matrix(orbs):
    '''Generate matrix of kinetic energy integrals

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        kineticM - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    kineticM = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            kineticM[i, j] = kinetic_integral(orbs[i], orbs[j])
            kineticM[j, i] = kineticM[i, j]

    return kineticM


@jit
def nuclear_matrix(orbs):
    '''Generate matrix of nuclear attraction energy integrals

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        nuclearM - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    nuclearM = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            nuclearM[i, j] = nuclear_integral(orbs[i], orbs[j])
            nuclearM[j, i] = nuclearM[i, j]

    return nuclearM


@jit
def coreham_matrix(orbs):
    '''Generate one-electron core Hamiltonian matrix

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        corehamM - Matrix of overlap integrals between all atomic orbitals
    '''''

    corehamM = nuclear_matrix(orbs) + kinetic_matrix(orbs)

    return corehamM
