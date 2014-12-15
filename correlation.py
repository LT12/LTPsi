__author__ = 'larry'
import numpy as np
from numba import jit

@jit
def mp2(ert, orb_e, ao_cof, n_occ, n_orb):
    """Calculates the Moller-Plesset MP2 energy correction

    Parameters
    ----------
    ert : ndarray
          two electron integral tensor (ij|kl)
    orb_e : array-like
            energy of the molecular orbitals for the system
    ao_cof : ndarray
            LCAO coefficents for the molecular orbitals
    n_occ : int
            number of occupied molecular orbitals
    n_orb : int
            total number of molecular orbitals for system

    Returns
    -------
    mp2_energy : float
                 MP2 correction to energy to account for correlation
    """

    # Convert Electron Repulsion Tensor from AO basis to MO basis
    mo_ert = np.einsum('ijkl, ls -> ijks', ert, ao_cof)
    mo_ert = np.einsum('ijks, kr -> ijrs', mo_ert, ao_cof)
    mo_ert = np.einsum('ijrs, jq -> iqrs', mo_ert, ao_cof)
    mo_ert = np.einsum('iqrs, ip -> pqrs', mo_ert, ao_cof)

    mp2_energy = 0

    for i in xrange(n_occ):
        for j in xrange(n_occ):
            for a in xrange(n_occ, n_orb):
                for b in xrange(n_occ, n_orb):

                    mp2_energy += mo_ert[i,a,j,b] *\
                                  (2 * mo_ert[i,a,j,b] - mo_ert[i,b,j,a])/\
                                  (orb_e[i] + orb_e[j] - orb_e[a] - orb_e[b])

    return mp2_energy