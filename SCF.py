'''
Created on Sep 28, 2014

@author: Larry
'''

import numpy as np
from scipy.linalg import eigh
from atomicParam import *
from CMolecularIntegrals import MolecularIntegrals
from Orbital import Orbital
import itertools
import time


class SCF():
    '''
    classdocs
    '''

    def __init__(self, mol, basis = "", MP2 = False, MullPop = False,
                 dipole = False, cartMatrix = None):

        # Introduce Molecular Parameters
        self.atomType = mol.atomType

        if cartMatrix is None:
            self.cartMatrix = mol.cartMatrix
        else:
            self.cartMatrix = cartMatrix

        #Calculate number of occupied orbitals and raise an execption
        #for odd # of e-
        num_e = sum([getAtomicCharge(i) for i in self.atomType])\
                    - mol.molecularCharge

        if num_e & 1:
            raise NameError("RHF only works on even electrons systems!")
        self.nOcc = num_e /2

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
        #Calculate
        self.MullikenPopulation()

    def detOrbList(self,basis):
        Z = getAtomicCharge

        if basis == "sto3G":     from basis import sto3g as bs
        elif basis == "6-31G":   from basis import p631 as bs
        elif basis == "3-21G":   from basis import p321 as bs
        elif basis == "6-31G**": from basis import p631ss as bs
        elif basis == "6-31++G": from basis import p631ppss as bs
        elif basis == "dzvp":    from basis import dzvp as bs
        else: from basis import sto3g as bs

        self.orbList = []
        append = self.orbList.append
        self.numPrimGauss = 0

        for i in xrange(self.numAtom):
            orb = bs.getOrbs(Z(self.atomType[i]))
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

                    R_ij =  self.cartMatrix[i,:] - self.cartMatrix[j,:]
                    nR_ij = np.sqrt(np.dot(R_ij,R_ij))

                    Z_iZ_j = Z(self.atomType[i]) * Z(self.atomType[j])

                    self.NuclRepEnergy += Z_iZ_j/nR_ij

    def SCF(self):

        self.symOrthogMatrix()
        self.transFock(self.CoreHam)
        self.densityMatrix()
        self.ElectronicEnergy(self.CoreHam)
        SCFitercount, prevElecEnergy = 0, 0

        while True:

            if abs(self.ElectrEnergy-prevElecEnergy) < 1E-10: break
            prevElecEnergy = self.ElectrEnergy
            self.Fock()
            self.transFock(self.FockM)
            self.densityMatrix()
            self.ElectronicEnergy(self.FockM)
            SCFitercount += 1

        self.TotalEnergy = self.ElectrEnergy + self.NuclRepEnergy

    def symOrthogMatrix(self):

        eig,eigVM = eigh(self.OverlapMatrix,turbo = True)

        eigT = np.diag(np.reciprocal(np.sqrt(eig)))

        self.symS = np.dot(eigVM,np.dot(eigT,eigVM.T))


    def Fock(self):
        self.FockM = self.CoreHam +\
                    2 * np.einsum('kl,ijkl->ij',self.densityM,self.ERT)\
                      - np.einsum('kl,ikjl->ij',self.densityM,self.ERT)

    def transFock(self, Fock):

        self.transFockM = np.dot(self.symS.T,np.dot(Fock,self.symS))

        self.eig,eigVM = eigh(self.transFockM,turbo=True)

        self.AOcoefs = np.dot(self.symS,eigVM)

    def densityMatrix(self):
        self.densityM = np.einsum('im ,jm ->ij',
                                  self.AOcoefs[:,:self.nOcc],
                                  self.AOcoefs[:,:self.nOcc])

    def ElectronicEnergy(self, Fock):
        self.ElectrEnergy = np.einsum('ij,ij', self.densityM,
                                               self.CoreHam + Fock)


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

