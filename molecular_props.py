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

'''
def DipoleMoments(self):

    MuX_e = 2 * np.einsum('ij,ij', self.densityM, self.XDipoleMatrix)
    MuY_e = 2 * np.einsum('ij,ij', self.densityM, self.YDipoleMatrix)
    MuZ_e = 2 * np.einsum('ij,ij', self.densityM, self.ZDipoleMatrix)

    MuX_N, MuY_N, MuZ_N = 0, 0, 0
    Z = getAtomicCharge
    if self.numAtom > 1:
        for i in enumerate(self.atomType):
            MuX_N += Z(i[1]) * self.cartMatrix[i[0], 0]
            MuY_N += Z(i[1]) * self.cartMatrix[i[0], 1]
            MuZ_N += Z(i[1]) * self.cartMatrix[i[0], 2]

    MuX = MuX_e + MuX_N
    MuY = MuY_e + MuY_N
    MuZ = MuZ_e + MuZ_N

    self.DipoleMoments = [MuX, MuY, MuZ]

def MullikenPopulation(self):

    self.MullCharges = [0] * self.numAtom
    Z = getAtomicCharge

    # Gross Orbital Product
    GOP = -2 * np.einsum('ij,ij->i', self.densityM,
                         self.OverlapMatrix)

    for i in xrange(self.numAtom):
        q = Z(self.atomType[i])

        for j in enumerate(self.orbList):
            if j[1].atom == i: q += GOP[j[0]]

        self.MullCharges[i] = q
'''