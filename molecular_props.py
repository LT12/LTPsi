from __future__ import division
from numba import jit
import numpy as np

__author__ = 'larry'


@jit
def calc_nucl_repulsion(atom_charge, cart_matrix):
    """calculates the nuclear repulsion energy between atoms in system

    Parameters
    ----------
    atom_charge : array-like
                  list of nuclear charges for each atom in system
    cart_matrix : ndarray
                  matrix of coordinates for each atom in system
    Returns
    -------
    nuc_repl : float
               nuclear repulsion energy
    Notes
    -----
    The nuclear repulsion energy can be calculated using the classical
    coulomb potential formula:
    .. math:: E_{nuc} = \sum \frac{Z_i Z_j}{r_{ij}}
    Only the lower diagonal portion of cart_matrix needs to be summed
    due to the symmetry of the problem, and the diagonal should not be
    summed.

    """

    nuc_repl = 0
    n_atom = len(atom_charge)

    if n_atom > 1:
        for i in xrange(n_atom):
            for j in xrange(i):
                r_i, r_j = cart_matrix[i, :], cart_matrix[j, :]
                r_ij = np.sqrt(np.dot(r_i - r_j, r_i - r_j))
                z_ij = atom_charge[i] * atom_charge[j]

                nuc_repl += z_ij / r_ij

    return nuc_repl