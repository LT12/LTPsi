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
        overlap_m - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    overlap_m = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            overlap_m[i, j] = overlap_integral(orbs[i], orbs[j])
            overlap_m[j, i] = overlap_m[i, j]

    return overlap_m


@jit
def kinetic_matrix(orbs):
    '''Generate matrix of kinetic energy integrals

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        kinetic_m - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    kinetic_m = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            kinetic_m[i, j] = kinetic_integral(orbs[i], orbs[j])
            kinetic_m[j, i] = kinetic_m[i, j]

    return kinetic_m


@jit
def nuclear_matrix(orbs):
    '''Generate matrix of nuclear attraction energy integrals

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        nuclear_m - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    nuclear_m = np.zeros((N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i + 1):
            nuclear_m[i, j] = nuclear_integral(orbs[i], orbs[j])
            nuclear_m[j, i] = nuclear_m[i, j]

    return nuclear_m


@jit
def coreham_matrix(orbs):
    '''Generate one-electron core Hamiltonian matrix

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        coreham_m - Matrix of overlap integrals between all atomic orbitals
    '''''

    coreham_m = nuclear_matrix(orbs) + kinetic_matrix(orbs)

    return coreham_m

@jit
def two_electron_tensor(orbs):
    '''Generate two-electron repulsion integral tensor

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        ert - Matrix of overlap integrals between all atomic orbitals
    '''''

    N = len(orbs)
    ert = np.zeros((N, N, N, N), dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i+1):
            for k in xrange(N):
                for l in xrange(k+1):
                    if (i + 1) * (j + 1) >= (k + 1) *  (l + 1):
                        ert[i,j,k,l] = two_electron_integral(orbs[i],orbs[j],
                                                             orbs[k],orbs[l])
                        ert[j,i,k,l] = ert[i,j,k,l]
                        ert[i,j,l,k] = ert[i,j,k,l]
                        ert[j,i,l,k] = ert[i,j,k,l]
                        ert[k,l,i,j] = ert[i,j,k,l]
                        ert[l,k,i,j] = ert[i,j,k,l]
                        ert[k,l,j,i] = ert[i,j,k,l]
                        ert[l,k,j,i] = ert[i,j,k,l]
    return ert

@jit
def overlap_integral(orb1,orb2):
    '''Calculate overlap integral between 2 atomic orbitals
    
    :rtype : float
    :param orb1:atomic orbital 1
    :param orb2:atomic orbital 2
    :return:integral: overlap integral
    '''

    integral = 0
    pol = primOverlapIntegral

    nprim1, nprim2 = orb1.nPrim, orb2.nPrim #number of primitive GTOs
    x1, y1, z1 = orb1.qnums #orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums #orbital 2 angular momenta
    d1, d2 = orb1.d, orb2.d #contraction coefficients
    a1, a2 = orb1.a, orb2.a #exponential coefficients
    cent1, cent2 = orb1.center #orbital positions

    for i in xrange(nprim1):
        for j in xrange(nprim2):

            Const = d1[i] * d2[j]   #Product of contraction coefficients

            integral += Const * pol(cent1[0],cent1[1],cent1[2],
                                    cent2[0],cent2[1],cent2[2],
                                    a1[i],a2[j],x1,y1,z1,x2,y2,z2)
    return integral