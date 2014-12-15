'''
Created on Sep 28, 2014

@author: Larry
'''
from __future__ import division
import time
import warnings

import numpy as np
from numba import jit
from scipy.linalg import eigh
from numpy.linalg.linalg import LinAlgError

from molecular_integrals import coreham_matrix, two_electron_tensor, overlap_matrix
from molecular_props import calc_nucl_repulsion
from orbital import det_orbs


class SCF():
    """Restricted Hartree-Fock SCF Algorithm

    Parameters
    ----------

    Attributes
    ----------

    Raises
    ------
    Notes
    -----

    """

    def __init__(self, mol, basis="STO3G", mp2=False,
                 dipole=False, cart_matrix=None):

        # Introduce Molecular Parameters
        self.atom_type = mol.atom_type
        self.atom_charge = mol.atom_charge
        if cart_matrix is None:
            self.cart_matrix = mol.cart_matrix
        else:
            self.cart_matrix = cart_matrix

        # Calculate number of occupied orbitals and raise an execption
        # for odd # of e-

        if mol.num_e % 2:
            raise NameError("RHF only works on even electrons systems!")
        self.n_occ = mol.num_e // 2

        # number of atoms in system
        self.num_atom = len(self.atom_type)

        # Calculate nuclear repulsion
        self.nuclear_rep = calc_nucl_repulsion(self.atom_charge,
                                               self.cart_matrix)

        # Determine basis set for molecule
        self.orbs = det_orbs(basis, self.atom_type, self.cart_matrix)
        self.n_orb = len(self.orbs)

        int_time_i = time.time()

        # Calculate molecular integrals
        # overlap integrals
        self.overlap = overlap_matrix(self.orbs)
        # core Hamiltonian
        self.core_ham = coreham_matrix(self.orbs, self.cart_matrix,
                                       self.atom_charge)
        # two electron integrals
        self.ert = two_electron_tensor(self.orbs)

        int_time_f = time.time()

        # Time to calculate integrals
        self.intTime = int_time_f - int_time_i

        # Initialize starting variables
        # Variables described in detail in class docstring
        self.scf_energy_list = []
        self._fock_list = np.ndarray(6, dtype=np.ndarray)
        self._error_list = np.ndarray(6, dtype=np.ndarray)
        self.itercount, self._fock_count = 0, 0
        self._diis_switch = False
        self.total_energy, self.elec_e = None, None
        self.ao_coefs, self.sym_s = None, None
        self.scf_energy, self.fock_m = 0, None
        self.density_m, self.orb_e = None
        self.trans_fock_m = None

        # Start SCF procedure
        self.scf()

        # If selected, perform Moller-Posset (MP2) correction
        if mp2:
            from correlation import mp2

            self.mp2_energy = mp2(self.ert, self.orb_e,
                                  self.ao_coefs, self.n_occ,
                                  self.n_orb)
            self.total_energy += self.mp2_energy
        # If selected, calculate dipole moments
        if dipole:
            pass
            # from molecular_props import calc_dipole
            # self.dipole_moments = calc_dipole()

            # Calculate Mulliken Charges
            # self.MullikenPopulation()

    def scf(self):

        # Construct orthogonalization matrix
        self.sym_orthog_matrix()

        # Initial density guess
        self.trans_fock(self.core_ham)
        self.comp_density_mat()
        self.comp_elec_energy(self.core_ham)

        # append initial energy to list of energies
        self.scf_energy_list.append(self.elec_e + self.nuclear_rep)

        # Start SCF procedure
        while True:
            # SCF convergence conditions
            delta_e = abs(self.scf_energy_list[self.itercount] -
                          self.scf_energy_list[self.itercount - 1])

            if ((delta_e < 1E-12 and self.itercount > 0)
                or self.itercount > 1000):
                break

            # Compute Fock Matrix, F = H + (2J - K)
            self.comp_fock()
            # Compute error matrix, e_i = FDS - SDF
            self.comp_error_mat()
            # Call Direct Inversion on Iterative Subspace procedure
            self.diis()
            # Transform Fock matrix into orthonormal AO basis
            self.trans_fock(self.fock_m)
            # Compute charge density matrix, D
            self.comp_density_mat()
            # Compute electronic SCF energy
            self.comp_elec_energy(self.fock_m)
            # increment iteration counter
            self.itercount += 1
            # Add nuclear repulsion energy to electronic energy
            self.scf_energy = self.elec_e + self.nuclear_rep
            # Store SCF energy for each iteration
            self.scf_energy_list.append(self.scf_energy)

        # Final SCF energy
        self.total_energy = self.scf_energy
        # Throw warning if SCF procedure failed to converge
        if self.itercount > 1000:
            warnings.warn('SCF did not converge!')

    def sym_orthog_matrix(self):
        # Construct symmetric orthogonalization matrix

        eigs, eig_vm = eigh(self.overlap, turbo=True)

        eig_m = np.diag(np.reciprocal(np.sqrt(eigs)))

        self.sym_s = np.dot(eig_vm, np.dot(eig_m, eig_vm.T))

    def comp_fock(self):
        # compute Fock matrix
        self.fock_m = self.core_ham + \
                      2 * np.einsum('kl,ijkl->ij', self.density_m, self.ert) \
                      - np.einsum('kl,ikjl->ij', self.density_m, self.ert)

        # Storing Fock Matricies for DIIS
        self._fock_list[self._fock_count % 6] = np.copy(self.fock_m)
        self._fock_count += 1

    def trans_fock(self, fock):
        # Transform Fock matrix into orthonormal AO basis
        self.trans_fock_m = np.dot(np.dot(self.sym_s.T, fock), self.sym_s)
        # Solve Roothaan - Hall Equation
        self.orb_e, eig_vm = eigh(self.trans_fock_m, turbo=True)
        # Transform AO coefficients back to non-orthogonal basis
        self.ao_coefs = np.dot(self.sym_s, eig_vm)

    def comp_density_mat(self):
        n = self.n_occ
        # Compute charge density matrix with occupied AO coefficents
        # D = C_occ * C_occ^T
        self.density_m = np.dot(self.ao_coefs[:, :n],
                                self.ao_coefs[:, :n].T)

    def comp_elec_energy(self, fock):
        # Compute SCF electronic energy, sum(sum( D * (H + F) ) )
        self.elec_e = np.einsum('ij,ij', self.density_m,
                                self.core_ham + fock)

    def comp_error_mat(self):

        # Compute error matrix FDS - SDF (since D and F should commute)
        error_m = np.dot(self.fock_m, np.dot(self.density_m,
                                             self.overlap)) - \
                  np.dot(self.overlap, np.dot(self.density_m,
                                              self.fock_m))

        # Place error matrix into orthonormal AO basis
        o_error_mat = np.dot(self.sym_s.T, np.dot(error_m, self.sym_s))

        # if max error from error matrix is below threshold
        # start DIIS procedure
        if o_error_mat.ravel().max() < 0.1:
            self._diis_switch = True

        # Store error matricies for DIIS procedure
        self._error_list[(self._fock_count - 1) % 6] = o_error_mat

    @jit
    def diis(self):
        """Direct Inversion in Iterative Subspace
        
           Accelerates SCF convergence by building a Fock matrix
           from previous Fock matrices using coefficients that
           extrapolate error matrix to zero in least-square sense
        """

        # Do not perform DIIS on first SCF cycle
        if self.itercount == 0:
            return

        # Peform Fock matrix averaging when DIIS is not active
        if not self._diis_switch:
            self.fock_m = 0.5 * (self._fock_list[(self._fock_count - 1) % 6]
                                 + self._fock_list[(self._fock_count - 2) % 6])
            return

        # Size of B matrix to be calculated
        n_b = self._fock_count % 6 + 1
        if self._fock_count >= 6:
            n_b = 7

        b_m = np.zeros((n_b, n_b))

        # Compute B matrix
        for i in xrange(n_b - 1):
            for j in xrange(i + 1):
                b_m[i, j] = b_m[j, i] = np.einsum('ij,ij', self._error_list[i],
                                                  self._error_list[j])
            b_m[n_b - 1, i] = b_m[i, n_b - 1] = -1

        # b = [0,0,..,0,-1]
        b = np.zeros(n_b)
        b[n_b - 1] = -1

        try:
            # Solve B * c = b for c to determine extrapolation coefficients
            coefs = np.linalg.solve(b_m, b)
        except LinAlgError:
            return
        # Build new Fock matrix with coefficients
        self.fock_m = np.sum(coefs[:n_b - 1] * self._fock_list[:n_b - 1])
