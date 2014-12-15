'''
Created on Sep 28, 2014

@author: Larry
'''
from __future__ import division
import numpy as np
import scipy
from scipy.linalg import eigh
from atomicparam import *
from molecular_integrals import coreham_matrix, two_electron_tensor
from molecular_props import calc_nucl_repulsion
from orbital import det_orbs
import itertools, time, warnings
from numpy.linalg.linalg import LinAlgError


class SCF():
    """
    Restriced Hartree-Fock SCF Algorithm
    """

    def __init__(self, mol, basis="STO3G", mp2=False,
                 dipole=False, cart_matrix=None):

        # Introduce Molecular Parameters
        self.atom_type = mol.atom_type

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

        intTimeI = time.time()

        # Calculate molecular integrals
        # core Hamiltonian
        self.core_ham = coreham_matrix(self.orbs, self.cart_matrix,
                                       self.atom_charge)
        # two electron integrals
        self.ert = two_electron_tensor(self.orbs)

        intTimeF = time.time()

        # Time to calculate integrals
        self.intTime = intTimeF - intTimeI

        # Start SCF procedure
        self.SCF()

        # If selected, perform Moller-Posset (MP2) correction
        if mp2:
            from correlation import mp2

            self.mp2_energy = mp2(self.ert, self.orb_e,
                                  self.ao_cof, self.n_occ,
                                  self.n_orb)

        # Calculate Mulliken Charges
        self.MullikenPopulation()


    def SCF(self):

        # Initialize starting variables
        self.SCFEnergyList = []
        self._fockList = np.ndarray(6, dtype=np.ndarray)
        self._errorList = np.ndarray(6, dtype=np.ndarray)
        self.itercount, self._FockCount = 0, 0
        self._DIIS_switch = False

        #Construct Orthogonalization Matrix
        self.symOrthogMatrix()

        #initial density guess
        self.transFock(self.CoreHam)
        self.compDensityMatrix()
        self.compElectronicEnergy(self.CoreHam)

        #append initial energy to list of energies
        self.SCFEnergyList.append(self.ElectrEnergy + self.NuclRepEnergy)

        #Start SCF procedure
        while True:
            #SCF convergence conditions
            DeltaE = abs(self.SCFEnergyList[self.itercount] -
                         self.SCFEnergyList[self.itercount - 1])

            if ((DeltaE < 1E-12 and self.itercount > 0)
                or self.itercount > 1000): break

            #Compute Fock Matrix, F = H + (2J - K)
            self.compFock()
            #Compute error matrix, e_i = FDS - SDF
            self.compErrorMatrix()
            #Call Direct Inversion on Iterative Subspace procedure    
            self.DIIS()
            #Transform Fock matrix into orthonormal AO basis
            self.transFock(self.FockM)
            #Compute charge density matrix, D
            self.compDensityMatrix()
            #Compute electronic SCF energy
            self.compElectronicEnergy(self.FockM)
            #increment iteration counter
            self.itercount += 1
            #Add nuclear repulsion energy to electronic energy
            self.SCFEnergy = self.ElectrEnergy + self.NuclRepEnergy
            #Store SCF energy for each iteration
            self.SCFEnergyList.append(self.SCFEnergy)

        #Final SCF energy
        self.TotalEnergy = self.SCFEnergy
        #Throw warning if SCF procedure failed to converge
        if self.itercount > 1000: warnings.warn('SCF did not converge!')

    def symOrthogMatrix(self):
        '''Construct symmetric orthogonalization matrix
        '''

        eigs, eigVM = eigh(self.OverlapMatrix)  # ,turbo = True)

        eigM = sp.diag(np.reciprocal(np.sqrt(eigs)))

        self.symS = sp.dot(eigVM, sp.dot(eigM, eigVM.T))


    def compFock(self):
        '''Compute Fock Matrix
        '''

        self.FockM = self.CoreHam + \
                     2 * np.einsum('kl,ijkl->ij', self.densityM, self.ERT) \
                     - np.einsum('kl,ikjl->ij', self.densityM, self.ERT)

        # Storing Fock Matricies for DIIS
        self._fockList[self._FockCount % 6] = np.copy(self.FockM)
        self._FockCount += 1


    def transFock(self, Fock):

        # Transform Fock matrix into orthonormal AO basis
        self.transFockM = sp.dot(self.symS.T, Fock).dot(self.symS)
        #Solve Roothaan - Hall Equation
        self.OrbE, eigVM = eigh(self.transFockM, turbo=True)
        #Transform AO coefficients back to non-orthogonal basis
        self.AOcoefs = sp.dot(self.symS, eigVM)

    def compDensityMatrix(self):
        N = self.nOcc
        # Compute charge density matrix with occupied AO coefficents
        # D = C_occ * C_occ^T
        self.densityM = sp.dot(self.AOcoefs[:, :N], (self.AOcoefs[:, :N].T))

    def compElectronicEnergy(self, Fock):
        # Compute SCF electronic energy, sum(sum( D * (H + F) ) )
        self.ElectrEnergy = np.einsum('ij,ij', self.densityM,
                                      self.CoreHam + Fock)

    def compErrorMatrix(self):
        # Compute error matrix FDS - SDF (since D and F should commute)
        errorMatrix = self.FockM.dot(self.densityM).dot(
            self.OverlapMatrix) - \
                      self.OverlapMatrix.dot(self.densityM).dot(
                          self.FockM)
        #Place error matrix into orthonormal AO basis
        OerrorMatrix = self.symS.T.dot(errorMatrix).dot(self.symS)

        #if max error from error matrix is below threshold
        #start DIIS procedurre
        if OerrorMatrix.ravel().max() < 0.1: self._DIIS_switch = True

        #Store error matricies for DIIS procedure
        self._errorList[(self._FockCount - 1) % 6] = OerrorMatrix


    def DIIS(self):
        '''Direct Inversion in Iterative Subspace
        
           Accelerates SCF convergence by building a Fock matrix
           from previous Fock matricies using coefficents that 
           extrapolate error matrix to zero in least-square sense
        '''

        # Do not perform DIIS on first SCF cycle
        if self.itercount == 0:
            return

        #Peform Fock matrix averaging when DIIS is not active
        if not self._DIIS_switch:
            self.FockM = 0.5 * (self._fockList[(self._FockCount - 1) % 6]
                                + self._fockList[(self._FockCount - 2) % 6])
            return

        #Size of B matrix to be calculated
        N_B = self._FockCount % 6 + 1
        if self._FockCount >= 6: N_B = 7

        B = np.zeros((N_B, N_B))

        #Compute B matrix
        for i in xrange(N_B - 1):
            for j in xrange(i + 1):
                B[i, j] = B[j, i] = np.einsum('ij,ij', self._errorList[i],
                                              self._errorList[j])
            B[N_B - 1, i] = B[i, N_B - 1] = -1

        #b = [0,0,..,0,-1]
        b = np.zeros(N_B)
        b[N_B - 1] = -1

        try:
            #Solve B * c = b for c to determine extrapolation coefficents
            cofs = np.linalg.solve(B, b)
        except LinAlgError:
            return
        #Build new Fock matrix with coefficients
        self.FockM = np.sum(cofs[:N_B - 1] * self._fockList[:N_B - 1])


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
