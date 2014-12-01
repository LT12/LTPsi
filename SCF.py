'''
Created on Sep 28, 2014

@author: Larry
'''

import numpy as np
import scipy as sp
from scipy.linalg import eigh
from atomicParam import *
from CMolecularIntegrals import MolecularIntegrals
from Orbital import Orbital
import itertools, time, warnings


class SCF():
    '''
    Restriced Hatree-Fock SCF Algorithm
    '''

    def __init__(self, mol, basis = "STO3G", MP2 = False,
                 dipole = False, cartMatrix = None):

        # Introduce Molecular Parameters
        self.atomType = mol.atomType

        if cartMatrix is None:
            self.cartMatrix = mol.cartMatrix
        else:
            self.cartMatrix = cartMatrix

        #Calculate number of occupied orbitals and raise an execption
        #for odd # of e-

        if mol.num_e % 2:
            raise NameError("RHF only works on even electrons systems!")
        self.nOcc = mol.num_e /2

        #number of atoms in system
        self.numAtom = len(self.atomType)

        #Calculate nuclear repulsion
        self.calcNuclRepulsion()
        
        #Determine basis set for molecule
        self.detOrbList(basis)
        self.nOrb = len(self.orbList)

        intTimeI = time.time()

        #Calculate molecular integrals
        if dipole:
            Ints = MolecularIntegrals(self.orbList, self.atomType,
                                      self.cartMatrix,1)
        else:
            Ints = MolecularIntegrals(self.orbList, self.atomType,
                                      self.cartMatrix,0)

        intTimeF = time.time()
        #Time to calculate integrals
        self.intTime = intTimeF - intTimeI

        #unpackage molecular integrals
        self.CoreHam, self.OverlapMatrix, self.ERT = Ints.CoreHam,\
                                                     Ints.OLMatrix,\
                                                     Ints.ERT
        
        #Start SCF procedure
        self.SCF()

        #If selected, perform Moller-Posset (MP2) correction
        if MP2:
            self.MP2()

        #If selected, calculate dipoles for system
        if dipole:

            self.XDipoleMatrix, self.YDipoleMatrix,self.ZDipoleMatrix =\
            Ints.XDipoleMatrix, Ints.YDipoleMatrix,Ints.ZDipoleMatrix

            self.DipoleMoments()
        #Calculate Mulliken Charges
        self.MullikenPopulation()

    def detOrbList(self,basis):
        
        #possibly implement a dictionary import method
        #basis_sets = {"STO3G","STO6G","3-21G","6-31G","6-31G**","6-31++G"}
        
        if basis == "STO3G":     from basis import STO3G   as bs
        elif basis == "STO6G":   from basis import STO6G   as bs
        elif basis == "6-31G":   from basis import b631G   as bs
        elif basis == "3-21G":   from basis import b321G   as bs
        elif basis == "6-31G*": from basis import b631Gs as bs
        elif basis == "6-31G**": from basis import b631Gss as bs
        elif basis == "6-31++G": from basis import b631ppG as bs
        elif basis == "6-311G":  from basis import b6311G  as bs
        elif basis == "6-311++G":  from basis import b6311ppG  as bs
        elif basis == "6-311++G**":  from basis import b6311ppGss  as bs
        elif basis == "DZP":     from basis import DZP     as bs
        elif basis == "DZ":      from basis import DZ      as bs
        elif basis == "MINI":    from basis import MINI    as bs
        else: raise NameError('Bad Basis!')

        self.orbList = []
        append = self.orbList.append
        self.numPrimGauss = 0

        for i in xrange(self.numAtom):
            orb = bs.getOrbs(self.atomType[i])
            N = len(orb)
            numOrbs = 0
            for j in xrange(N):
                if orb[j][0] == "S":
                    append(Orbital(orb[j],self.cartMatrix[i,:],i))
                elif orb[j][0] == "P":
                    append(Orbital(("PX",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("PY",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("PZ",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                elif orb[j][0] == "D":
                    append(Orbital(("DX2",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("DY2",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("DZ2",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("DXZ",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("DXY",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                    append(Orbital(("DYZ",orb[j][1]),
                                    self.cartMatrix[i,:],i))
                else:
                    print orb[j][0]
                    raise NameError("bad orbital!")

        for orb in self.orbList:
            self.numPrimGauss += orb.nPrim

    def calcNuclRepulsion(self):

        self.NuclRepEnergy = 0
        Z = getAtomicCharge

        if self.numAtom > 1:
            for i in xrange(self.numAtom):
                for j in xrange(i):

                    r_i, r_j =  self.cartMatrix[i,:], self.cartMatrix[j,:]
                    nR_ij = np.sqrt(sp.dot(r_i - r_j,r_i - r_j))

                    Z_iZ_j = Z(self.atomType[i]) * Z(self.atomType[j])

                    self.NuclRepEnergy += Z_iZ_j/nR_ij

    def SCF(self):

        #Initialize starting variables
        self.SCFEnergyList = []
        self._fockList = np.ndarray(6,dtype=np.ndarray)
        self._errorList = np.ndarray(6,dtype=np.ndarray)
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

        eig,eigVM = eigh(self.OverlapMatrix,turbo = True)

        eigM = sp.diag(np.reciprocal(np.sqrt(eig)))

        self.symS = sp.dot(eigVM,sp.dot(eigM,eigVM.T))


    def compFock(self):
        '''Compute Fock Matrix
        '''

        self.FockM = self.CoreHam +\
                     2 * np.einsum('kl,ijkl->ij',self.densityM,self.ERT)\
                       - np.einsum('kl,ikjl->ij',self.densityM,self.ERT)

        #Storing Fock Matricies for DIIS    
        self._fockList[self._FockCount % 6] = np.copy(self.FockM)
        self._FockCount += 1


    def transFock(self,Fock):

        #Transform Fock matrix into orthonormal AO basis
        self.transFockM = sp.dot(self.symS.T,Fock).dot(self.symS)
        #Solve Roothaan - Hall Equation
        self.eig,eigVM = eigh(self.transFockM,turbo=True)
        #Transform AO coefficients back to non-orthogonal basis
        self.AOcoefs = sp.dot(self.symS,eigVM)

    def compDensityMatrix(self):
        N = self.nOcc
        # Compute charge density matrix with occupied AO coefficents
        # D = C_occ * C_occ^T
        self.densityM = sp.dot(self.AOcoefs[:,:N],(self.AOcoefs[:,:N].T))

    def compElectronicEnergy(self, Fock):
        #Compute SCF electronic energy, sum(sum( D * (H + F) ) )
        self.ElectrEnergy = np.einsum('ij,ij', self.densityM,
							    self.CoreHam + Fock)
                                               
    def compErrorMatrix(self):
        #Compute error matrix FDS - SDF (since D and F should commute)
        errorMatrix = self.FockM.dot(self.densityM).dot(
                                     self.OverlapMatrix) -\
                      self.OverlapMatrix.dot(self.densityM).dot(
                                                             self.FockM)
        #Place error matrix into orthonormal AO basis
        OerrorMatrix = self.symS.T.dot(errorMatrix).dot(self.symS)
        
        #if max error from error matrix is below threshold
        #start DIIS procedurre
        if OerrorMatrix.ravel().max() < 0.1: self._DIIS_switch = True
        
        #Store error matricies for DIIS procedure
        self._errorList[(self._FockCount - 1) % 6 ] = OerrorMatrix
       

    def DIIS(self):
        '''Direct Inversion in Iterative Subspace
        
           Accelerates SCF convergence by building a Fock matrix
           from previous Fock matricies using coefficents that 
           extrapolate error matrix to zero in least-square sense
        '''

        #Do not perform DIIS on first SCF cycle
        if self.itercount == 0:
            return
            
        #Peform Fock matrix averaging when DIIS is not active
        if not self._DIIS_switch:
            self.FockM =  0.5 * (self._fockList[(self._FockCount - 1) % 6]
                          + self._fockList[(self._FockCount - 2) % 6]) 
            return

        #Size of B matrix to be calculated
        N_B = self._FockCount % 6 + 1
        if self._FockCount>= 6: N_B = 7
            
        B = np.zeros((N_B,N_B))
        
        #Compute B matrix
        for i in xrange(N_B-1):
            for j in xrange(i+1):

                B[i,j] = B[j,i] = np.einsum('ij,ij',self._errorList[i],
                                                    self._errorList[j])
            B[N_B-1,i] = B[i,N_B-1] = -1
        
        #b = [0,0,..,0,-1]
        b = np.zeros(N_B)
        b[N_B-1] = -1
        
        #Solve B * c = b for c to determine extrapolation coefficents
        cofs = np.linalg.solve(B,b)
        #Build new Fock matrix with coefficients
        self.FockM = np.sum(cofs[:N_B-1] * self._fockList[:N_B-1])

    def MP2(self):

        #define N-number of orbitals, nOcc-number of occupied orbitals,
        #MOERT-ERT in MO basis, Orbenergy- energy of each orbital in
        #ordered manner, C-Matrix of AO coefficients
        N,nOcc,MOERT,OrbEnergy,C = len(self.orbList),self.nOcc,\
                                   np.copy(self.ERT),self.eig,\
                                   self.AOcoefs

        #Convert Electron Repulsion Tensor from AO basis to MO basis
        MOERT = np.einsum('ijkl, ls -> ijks',MOERT,C)
        MOERT = np.einsum('ijks, kr -> ijrs',MOERT,C)
        MOERT = np.einsum('ijrs, jq -> iqrs',MOERT,C)
        MOERT = np.einsum('iqrs, ip -> pqrs',MOERT,C)

        self.MP2Energy = 0

        for i,j,a,b in itertools.product(xrange(nOcc),xrange(nOcc),
                                         xrange(nOcc,N),xrange(nOcc,N)):

            self.MP2Energy += MOERT[i,a,j,b] *\
                              (2 * MOERT[i,a,j,b] - MOERT[i,b,j,a])/\
                              (OrbEnergy[i] + OrbEnergy[j]
                               - OrbEnergy[a] - OrbEnergy[b])

        self.TotalEnergy += self.MP2Energy

    def DipoleMoments(self):

        MuX_e = 2 * np.einsum('ij,ij',self.densityM,self.XDipoleMatrix)
        MuY_e = 2 * np.einsum('ij,ij',self.densityM,self.YDipoleMatrix)
        MuZ_e = 2 * np.einsum('ij,ij',self.densityM,self.ZDipoleMatrix)

        MuX_N, MuY_N, MuZ_N = 0, 0, 0
        Z = getAtomicCharge
        if self.numAtom > 1:
            for i in enumerate(self.atomType):
                MuX_N += Z(i[1])*self.cartMatrix[i[0],0]
                MuY_N += Z(i[1])*self.cartMatrix[i[0],1]
                MuZ_N += Z(i[1])*self.cartMatrix[i[0],2]

        MuX = MuX_e + MuX_N
        MuY = MuY_e + MuY_N
        MuZ = MuZ_e + MuZ_N

        self.DipoleMoments = [MuX,MuY,MuZ]

    def MullikenPopulation(self):

        self.MullCharges = [0] * self.numAtom
        Z = getAtomicCharge

        #Gross Orbital Product
        GOP = -2 *  np.einsum('ij,ij->i',self.densityM,
                                         self.OverlapMatrix)

        for i in xrange(self.numAtom):
            q = Z(self.atomType[i])

            for j in enumerate(self.orbList):
                if j[1].atom == i: q += GOP[j[0]]

            self.MullCharges[i] = q
