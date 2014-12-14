__author__ = 'larry'

import numpy as np
from cPrimMolInt import primOverlapIntegral, primNuclearAttractionIntegral, primElecRepulInt
from numba import jit


@jit
def overlap_matrix(orbs):
    """Generate matrix of overlap integrals between each atomic orbital

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        overlap_m - Matrix of overlap integrals between all atomic orbitals
    """

    n = len(orbs)
    overlap_m = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            overlap_m[i, j] = overlap_integral(orbs[i], orbs[j])
            overlap_m[j, i] = overlap_m[i, j]

    return overlap_m


@jit
def kinetic_matrix(orbs):
    """Generate matrix of kinetic energy integrals

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        kinetic_m - Matrix of overlap integrals between all atomic orbitals
    """

    n = len(orbs)
    kinetic_m = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            kinetic_m[i, j] = kinetic_integral(orbs[i], orbs[j])
            kinetic_m[j, i] = kinetic_m[i, j]

    return kinetic_m


@jit
def nuclear_matrix(orbs, cart_matrix, atom_charge):
    """Generate matrix of nuclear attraction energy integrals

    :param orbs: list of atomic orbitals for molecule
    :param cart_matrix: matrix of atomic coordinates
    :param atom_charge: list of nuclear charges for each atom
    :return: nuclear_m: matrix of nuclear attraction integrals
    """

    n = len(orbs)
    nuclear_m = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            nuclear_m[i, j] = nuclear_integral(orbs[i], orbs[j],
                                               cart_matrix, atom_charge)
            nuclear_m[j, i] = nuclear_m[i, j]

    return nuclear_m


@jit
def coreham_matrix(orbs, cart_matrix, atom_charge):
    """Generate one-electron core Hamiltonian matrix

    :param orbs: list of atomic orbitals for molecule
    :param cart_matrix: matrix of atomic coordinates
    :param atom_charge: list of nuclear charges for each atom
    :return: coreham_m: core Hamiltonian matrix
    """

    coreham_m = nuclear_matrix(orbs) + kinetic_matrix(orbs, cart_matrix, atom_charge)

    return coreham_m


@jit
def two_electron_tensor(orbs):
    """Generate two-electron repulsion integral tensor

        Arguments:
        orbs - list of atomic orbitals for molecule

        Return:
        ert - Matrix of overlap integrals between all atomic orbitals
    """

    n = len(orbs)
    ert = np.zeros((n, n, n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            for k in xrange(n):
                for l in xrange(k + 1):
                    if (i + 1) * (j + 1) >= (k + 1) * (l + 1):
                        ert[i, j, k, l] = two_electron_integral(orbs[i], orbs[j],
                                                                orbs[k], orbs[l])
                        ert[j, i, k, l] = ert[i, j, k, l]
                        ert[i, j, l, k] = ert[i, j, k, l]
                        ert[j, i, l, k] = ert[i, j, k, l]
                        ert[k, l, i, j] = ert[i, j, k, l]
                        ert[l, k, i, j] = ert[i, j, k, l]
                        ert[k, l, j, i] = ert[i, j, k, l]
                        ert[l, k, j, i] = ert[i, j, k, l]
    return ert


@jit
def overlap_integral(orb1, orb2):
    """Calculate overlap integral between 2 atomic orbitals
    
    :rtype : float
    :param orb1:atomic orbital 1
    :param orb2:atomic orbital 2
    :return:integral: overlap integral
    """

    integral = 0
    pol = primOverlapIntegral

    nprim1, nprim2 = orb1.nPrim, orb2.nPrim  # number of primitive GTOs
    x1, y1, z1 = orb1.qnums  # orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums  # orbital 2 angular momenta
    d1, d2 = orb1.d, orb2.d  # contraction coefficients
    a1, a2 = orb1.a, orb2.a  # exponential coefficients
    cent1, cent2 = orb1.center, orb2.center  # orbital positions

    for i in xrange(nprim1):
        for j in xrange(nprim2):
            const = d1[i] * d2[j]  # product of contraction coefficients

            integral += const * pol(cent1[0], cent1[1], cent1[2],
                                    cent2[0], cent2[1], cent2[2],
                                    a1[i], a2[j], x1, y1, z1, x2, y2, z2)
    return integral


@jit
def kinetic_integral(orb1, orb2):
    """Calculate kinetic energy integral between 2 atomic orbitals

    :rtype : float
    :param orb1:atomic orbital 1
    :param orb2:atomic orbital 2
    :return:integral: kinetic energy integral
    """

    integral = 0
    pol = primOverlapIntegral

    nprim1, nprim2 = orb1.nPrim, orb2.nPrim  # number of primitive GTOs
    x1, y1, z1 = orb1.qnums  # orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums  # orbital 2 angular momenta
    d1, d2 = orb1.d, orb2.d  # contraction coefficients
    a1, a2 = orb1.a, orb2.a  # exponential coefficients
    cent1, cent2 = orb1.center, orb2.center  # orbital positions

    for i in xrange(nprim1):
        for j in xrange(nprim2):
            const = d1[i] * d2[j]  # product of contraction coefficients

            p = pol(cent1[0], cent1[1], cent1[2],
                    cent2[0], cent2[1], cent2[2],
                    a1[i], a2[j], x1, y1, z1, x2, y2, z2)

            px2 = pol(cent1[0], cent1[1], cent1[2],
                      cent2[0], cent2[1], cent2[2],
                      a1[i], a2[j], x1, y1, z1, x2 + 2, y2, z2)
            py2 = pol(cent1[0], cent1[1], cent1[2],
                      cent2[0], cent2[1], cent2[2],
                      a1[i], a2[j], x1, y1, z1, x2, y2 + 2, z2)
            pz2 = pol(cent1[0], cent1[1], cent1[2],
                      cent2[0], cent2[1], cent2[2],
                      a1[i], a2[j], x1, y1, z1, x2, y2, z2 + 2)

            pxm2 = pol(cent1[0], cent1[1], cent1[2],
                       cent2[0], cent2[1], cent2[2],
                       a1[i], a2[j], x1, y1, z1, x2 - 2, y2, z2)
            pym2 = pol(cent1[0], cent1[1], cent1[2],
                       cent2[0], cent2[1], cent2[2],
                       a1[i], a2[j], x1, y1, z1, x2, y2 - 2, z2)
            pzm2 = pol(cent1[0], cent1[1], cent1[2],
                       cent2[0], cent2[1], cent2[2],
                       a1[i], a2[j], x1, y1, z1, x2, y2, z2 - 2)

            integral += const * (a2[j] * (2 * (x2 + y2 + z2) + 3) * p -
                                 (2 * a2[j] ** 2) * (px2 + py2 + pz2)
                                 - 0.5 * (x2 * (x2 - 1) * pxm2 +
                                          y2 * (y2 - 1) * pym2 + z2 * (z2 - 1) * pzm2))
    return integral


@jit
def nuclear_integral(orb1, orb2, cart_matrix, atom_charge):
    """Calculate nuclear attraction integral between 2 atomic orbitals

    :type atom_charge: ndarray
    :type cart_matrix: ndarray
    :rtype :float
    :param orb1:atomic orbital 1
    :param orb2:atomic orbital 2
    :return:integral: nuclear attraction integral
    """

    integral = 0
    nai = primNuclearAttractionIntegral

    nprim1, nprim2 = orb1.nPrim, orb2.nPrim  # number of primitive GTOs
    x1, y1, z1 = orb1.qnums  # orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums  # orbital 2 angular momenta
    d1, d2 = orb1.d, orb2.d  # contraction coefficients
    a1, a2 = orb1.a, orb2.a  # exponential coefficients
    cent1, cent2 = orb1.center, orb2.center  # orbital positions

    for i in xrange(nprim1):
        for j in xrange(nprim2):
            const = d1[i] * d2[j]  # product of contraction coefficients

            for idx, Z in enumerate(atom_charge):
                integral += const * Z * nai(cart_matrix[idx, 0], cart_matrix[idx, 1],
                                            cart_matrix[idx, 2],
                                            cent1[0], cent1[1], cent1[2],
                                            cent2[0], cent2[1], cent2[2],
                                            a1[i], a2[j], x1, y1, z1, x2, y2, z2)
    return integral


@jit
def two_electron_integral(orb1, orb2, orb3, orb4):
    """Calculate two-electron electron repulsion integral

    :rtype : float
    :param orb1:atomic orbital 1
    :param orb2:atomic orbital 2
    :param orb3:atomic orbital 3
    :param orb4:atomic orbital 4
    :return:integral: two electron integral
    """

    integral = 0
    eri = primElecRepulInt

    nprim1, nprim2 = orb1.nPrim, orb2.nPrim  # number of primitive GTOs
    nprim3, nprim4 = orb3.nPrim, orb4.nPrim  # number of primitive GTOs
    x1, y1, z1 = orb1.qnums  # orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums  # orbital 2 angular momenta
    x3, y3, z3 = orb3.qnums  # orbital 3 angular momenta
    x4, y4, z4 = orb4.qnums  # orbital 4 angular momenta
    d1, d2, d3, d4 = orb1.d, orb2.d, orb3.d, orb4.d  # contraction coefficients
    a1, a2, a3, a4 = orb1.a, orb2.a, orb3.a, orb4.a  # exponential coefficients
    cent1, cent2 = orb1.center, orb2.center  # orbital positions
    cent3, cent4 = orb3.center, orb4.center  # orbital positions

    # sum over all GTOs in basis set
    for i in xrange(nprim1):
        for j in xrange(nprim2):
            for k in xrange(nprim3):
                for l in xrange(nprim4):
                    # product of all contraction coefficients
                    const = d1[i] * d2[j] * d3[k] * d4[l]

                    integral += const * eri(cent1[0], cent1[1], cent1[2],
                                            cent2[0], cent2[1], cent2[2],
                                            cent3[0], cent3[1], cent3[2],
                                            cent4[0], cent4[1], cent4[2],
                                            a1[i], a2[j], a3[k], a4[l], x1,
                                            x2, x3, x4, y1, y2, y3, y4, z1,
                                            z2, z3, z4)
    return integral