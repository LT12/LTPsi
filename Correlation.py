__author__ = 'larry'
import numpy as np
from numba import jit

@jit
def MP2(ert, orb_e, ao_cof, n_occ, n_orb):
    """Calculates the Moller-Plesset MP2 energy correction

    :param ert: electron repulsion tensor
    :param orb_e: orbital energies
    :param ao_cof: atomic orbital coefficents
    :param n_occ: number of occupied orbitals
    :param n_orb: number of orbitals
    :return: mp2_energy: MP2 correction to energy
    """

    # Convert Electron Repulsion Tensor from AO basis to MO basis
    mo_ert = np.einsum('ijkl, ls -> ijks', ert, ao_cof)
    mo_ert = np.einsum('ijks, kr -> ijrs', mo_ert, ao_cof)
    mo_ert = np.einsum('ijrs, jq -> iqrs', mo_ert, ao_cof)
    mo_ert = np.einsum('iqrs, ip -> pqrs', mo_ert, ao_cof)

    mp2_energy = 0

    for i,j,a,b in itertools.product(xrange(nOcc),xrange(nOcc),
                                     xrange(nOcc,N),xrange(nOcc,N)):

        self.MP2Energy += MOERT[i,a,j,b] *\
                          (2 * MOERT[i,a,j,b] - MOERT[i,b,j,a])/\
                          (OrbE[i] + OrbE[j] - OrbE[a] - OrbE[b])

    return mp2_energy