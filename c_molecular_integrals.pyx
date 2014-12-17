#cython: wraparound = False, boundscheck = False
"""cython version of moleculer_integrals

"""

import numpy as np
cimport cython
cimport numpy as np
from cPrimMolInt cimport primOverlapIntegral, primNuclearAttractionIntegral, primElecRepulInt


cpdef np.ndarray overlap_matrix(tuple orbs):
    """Generate matrix of overlap integrals between each atomic orbital

        Parameters
        ----------
        orbs : array-like
               list of atomic orbitals for molecule

        Returns
        -------
        overlap_m_v : ndarray
                    matrix of overlap integrals between all atomic orbitals

        Notes
        -----
        Since the overlap integral is Hermitian only the lower diagonal
        portion of the matrix is calculated.
    """
    cdef int i, j

    cdef int n = len(orbs)
    cdef double [:,:] overlap_m_v = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            overlap_m_v[i, j] = overlap_integral(orbs[i], orbs[j])
            overlap_m_v[j, i] = overlap_m_v[i, j]
    
    cdef np.ndarray overlap_m = np.asarray(overlap_m_v)
    
    return overlap_m


cpdef np.ndarray kinetic_matrix(tuple orbs):
    """Generate matrix of kinetic energy integrals between each atomic orbital

        Parameters
        ----------
        orbs : array-like
               list of atomic orbitals for molecule

        Returns
        -------
        kinetic_m : ndarray
                    matrix of kinetic energy integrals between all atomic orbitals

        Notes
        -----
        Since the kinetic energy integral is Hermitian only the lower
        portion of the matrix is calculated.
    """
    cdef int i, j

    cdef int n = len(orbs)
    cdef double [:,:] kinetic_m_v = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            kinetic_m_v[i, j] = kinetic_integral(orbs[i], orbs[j])
            kinetic_m_v[j, i] = kinetic_m_v[i, j]

    cdef np. ndarray kinetic_m = np.asarray(kinetic_m_v)

    return kinetic_m


cpdef np.ndarray nuclear_matrix(tuple orbs, double[:,:] cart_matrix, long[:] atom_charge):
    """Generate matrix of nuclear attraction energy integrals

    Parameters
    ----------
    orbs : array-like
           list of atomic orbitals for molecule
    cart_matrix : ndarray
                  matrix of atomic coordinates
    atom_charge : array-like
                  list of nuclear charges for each atom
    Returns
    -------
    nuclear_m : ndarray
                matrix of nuclear attraction integral
    Notes
    -----
    Since the nuclear attraction integral is Hermitian, only the lower
    portion of the matrix is calculated.
    """
    cdef int i,j

    cdef int n = len(orbs)
    cdef double [:,:] nuclear_m_v = np.zeros((n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            nuclear_m_v[i, j] = nuclear_integral(orbs[i], orbs[j],
                                               cart_matrix, atom_charge)
            nuclear_m_v[j, i] = nuclear_m_v[i, j]

    cdef np. ndarray nuclear_m = np.asarray(nuclear_m_v)

    return nuclear_m


cpdef np.ndarray coreham_matrix(tuple orbs, double[:,:] cart_matrix,
                                long[:] atom_charge):
    """Generate one-electron core Hamiltonian matrix

        .. math:: H_{core} = KI + NAI

    Parameters
    ----------
    orbs : array-like
           list of atomic orbitals for molecule
    cart_matrix : ndarray
                  matrix of atomic coordinates
    atom_charge : array-like
                  list of nuclear charges for each atom

    Returns
    -------
    coreham_m : ndarray
                core Hamiltonian matrix
    """

    cdef np.ndarray coreham_m = nuclear_matrix(orbs, cart_matrix, atom_charge) + kinetic_matrix(orbs)

    return coreham_m


cpdef np.ndarray two_electron_tensor(orbs):
    """Generate two-electron repulsion integral tensor

    Parameters
    ----------
    orbs : array-like
           list of atomic orbitals for molecule

    Returns
    -------
    ert : ndarray
          two electron integral rank 4 tensor

    Notes
    -----
    The two electron repulsion tensor is a rank 4- tensor
    meaning the ndarray has 4 indicies, e.g. ert[i,j,k,l].

    The two electron integrals have 8-fold permutation
    symmetry, so only the lower octet of the tensor is
    calculated.
    """
    cdef int i, j, k , l

    cdef int n = len(orbs)
    cdef double [:,:,:,:] ert_v = np.zeros((n, n, n, n), dtype=np.float64)

    for i in xrange(n):
        for j in xrange(i + 1):
            for k in xrange(n):
                for l in xrange(k + 1):
                    if (i + 1) * (j + 1) >= (k + 1) * (l + 1):
                        ert_v[i, j, k, l] = two_electron_integral(orbs[i], orbs[j],
                                                                orbs[k], orbs[l])
                        ert_v[j, i, k, l] = ert_v[i, j, k, l]
                        ert_v[i, j, l, k] = ert_v[i, j, k, l]
                        ert_v[j, i, l, k] = ert_v[i, j, k, l]
                        ert_v[k, l, i, j] = ert_v[i, j, k, l]
                        ert_v[l, k, i, j] = ert_v[i, j, k, l]
                        ert_v[k, l, j, i] = ert_v[i, j, k, l]
                        ert_v[l, k, j, i] = ert_v[i, j, k, l]

    cdef np.ndarray ert = np.asarray(ert_v)

    return ert


cdef double overlap_integral(object orb1, object orb2):
    """Calculate overlap integral between 2 atomic orbitals
    
        .. math::
                 < \\psi_ i | \\psi_j >

        Parameters
        ----------
        orb1 : object
               orbital object, atomic orbital 1
        orb2 : object
               orbital object, atomic orbital 2

        Returns
        -------
        float
              overlap integral (S)
    """
    cdef int i, j, x1, y1, z1, x2, y2, z2, nprim1, nprim2
    cdef double [:] d1 ,d2, a1, a2, cent1, cent2
    cdef double const

    cdef double integral = 0
    pol = primOverlapIntegral

    nprim1, nprim2 = orb1.n_prim, orb2.n_prim  # number of primitive GTOs
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


cdef double kinetic_integral(object orb1, object orb2):
    """Calculate kinetic energy integral between 2 atomic orbitals

        .. math::
                 \\frac{1}{2} < \\psi_ i |\\nabla^2 | \\psi_j >

        Parameters
        ----------
        orb1 : object
               orbital object, atomic orbital 1
        orb2 : object
               orbital object, atomic orbital 2

        Returns
        -------
        float
              kinetic energy integral (KI)
    """
    cdef int i, j, x1, y1, z1, x2, y2, z2, nprim1, nprim2
    cdef double [:] d1 ,d2, a1, a2, cent1, cent2
    cdef double const, p, px2, py2, pz2,  pxm2, pym2, pzm2

    cdef double integral = 0
    pol = primOverlapIntegral

    nprim1, nprim2 = orb1.n_prim, orb2.n_prim  # number of primitive GTOs
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


cdef double nuclear_integral(object orb1, object orb2,
                             double [:,:] cart_matrix,
                             long [:] atom_charge):
    """Calculate nuclear attraction integral between 2 atomic orbitals

        .. math::
                 < \\psi_ i |\\sum_N \\frac{-1}{r_N}|\\psi_j >

        Parameters
        ----------
        orb1 : object
               orbital object, atomic orbital 1
        orb2 : object
               orbital object, atomic orbital 2
        cart_matrix : float[:,:]
                      matrix with Cartesian coordinates for each atom
        atom_charge : array-like
                      list of nuclear charges for each atom

        Returns
        -------
        float
              nuclear attraction integral (NAI)
    """
    cdef int i, j, k, x1, y1, z1, x2, y2, z2, nprim1, nprim2
    cdef double [:] d1 ,d2, a1, a2, cent1, cent2
    cdef double const

    cdef double integral = 0
    nai = primNuclearAttractionIntegral

    nprim1, nprim2 = orb1.n_prim, orb2.n_prim  # number of primitive GTOs
    x1, y1, z1 = orb1.qnums  # orbital 1 angular momenta
    x2, y2, z2 = orb2.qnums  # orbital 2 angular momenta
    d1, d2 = orb1.d, orb2.d  # contraction coefficients
    a1, a2 = orb1.a, orb2.a  # exponential coefficients
    cent1, cent2 = orb1.center, orb2.center  # orbital positions

    for i in xrange(nprim1):
        for j in xrange(nprim2):
            const = d1[i] * d2[j]  # product of contraction coefficients

            for k in xrange(len(atom_charge)):
                integral -= const * atom_charge[k] * \
                    nai(cart_matrix[k, 0], cart_matrix[k, 1],
                    cart_matrix[k, 2], cent1[0], cent1[1], cent1[2],
                    cent2[0], cent2[1], cent2[2], a1[i], a2[j], x1, y1,
                    z1, x2, y2, z2)

    return integral


cdef double two_electron_integral(object orb1, object orb2,
                           object orb3, object orb4):
    """Calculate two-electron electron repulsion integral

        ERI = .. math:: (ij|kl)

        Parameters
        ----------
        orb1 : object
               orbital object, atomic orbital 1
        orb2 : object
               orbital object, atomic orbital 2
        orb3 : object
               orbital object, atomic orbital 3
        orb4 : object
               orbital object, atomic orbital 4

        Returns
        -------
        float
              electron repulsion integral (ERI)
    """
    cdef int i, j, x1, y1, z1, x2, y2, z2, nprim1, nprim2
    cdef int k, l, x3, y3, z3, x4, y4, z4, nprim3, nprim4
    cdef double [:] d1 ,d2, a1, a2, cent1, cent2
    cdef double [:] d3 ,d4, a3, a4, cent3, cent4
    cdef double const

    cdef double integral = 0
    eri = primElecRepulInt

    nprim1, nprim2 = orb1.n_prim, orb2.n_prim  # number of primitive GTOs
    nprim3, nprim4 = orb3.n_prim, orb4.n_prim  # number of primitive GTOs
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