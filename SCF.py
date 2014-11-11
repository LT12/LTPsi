'''
Created on Sep 28, 2014

@author: Larry
'''

import numpy as np
from scipy.linalg import eigh
from atomicParam import *
from CMolecularIntegrals import MolecularIntegrals
from Orbital import Orbital
import time as t

class SCF():
    '''
    classdocs
    '''

    def __init__(self, mol, basis="", MP2=False, MullPop=False, dipole=False):
        
        # Introduce Molecular Parameters
        self.atomType,self.cartMatrix,self.distanceMatrix = mol.atomType, mol.cartMatrix, mol.DistanceMatrix 
        
        #Calculate number of occupied orbitals and raise an execption for odd # of e-
        num_e = (sum([getAtomicCharge(i) for i in self.atomType])-mol.molecularCharge)
        if num_e & 1: raise NameError("RHF only works on even electrons systems!")
        self.nOcc = num_e /2
        
        self.numAtom = len(self.atomType)
        
        self.calcNuclRepulsion()
        self.detOrbList(basis)
        self.nOrb = len(self.orbList)
        
        print "Number of Orbitals: " + str(self.nOrb)
        print "Number of Primitive Gaussians: " + str(self.numPrimGauss)
        t1 = t.time()
        if dipole:
            Ints = MolecularIntegrals(self.orbList, self.atomType, self.cartMatrix,1)
        else:
            Ints = MolecularIntegrals(self.orbList, self.atomType, self.cartMatrix,0)
        t2 = t.time()
        print "Integration Complete in "+str(t2-t1)+" seconds"
        self.CoreHam, self.OverlapMatrix, self.ERT = Ints.CoreHam, Ints.OLMatrix, Ints.ERT
        np.savetxt("bithme.csv", self.OverlapMatrix,delimiter=",")
        self.SCF()
        if MP2:
            self.MP2()
        if dipole:
            self.XDipoleMatrix, self.YDipoleMatrix,self.ZDipoleMatrix = Ints.XDipoleMatrix, Ints.YDipoleMatrix,Ints.ZDipoleMatrix
            self.DipoleMoments()
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
        self.atomOrbInfo = [0]*self.numAtom
        self.numPrimGauss = 0
        
        for i in xrange(self.numAtom):
            orb = bs.getOrbs(Z(self.atomType[i]))
            N = len(orb)
            numOrbs = 0 
            for j in xrange(N):
                if orb[j][0] == "S":
                    append(Orbital(orb[j],self.cartMatrix[i,:]))
                    numOrbs += 1 
                elif orb[j][0] == "P":
                    append(Orbital(("PX",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("PY",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("PZ",orb[j][1]),self.cartMatrix[i,:]))
                    numOrbs += 3
                elif orb[j][0] == "D":
                    append(Orbital(("DX2",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("DY2",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("DZ2",orb[j][1]),self.cartMatrix[i,:]))                
                    append(Orbital(("DXZ",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("DXY",orb[j][1]),self.cartMatrix[i,:]))
                    append(Orbital(("DYZ",orb[j][1]),self.cartMatrix[i,:]))
                    numOrbs += 6
                else:
                    print orb[j][0] 
                    raise NameError("bad orbital!")
            for  orb in self.orbList:
                self.numPrimGauss += orb.nPrim
            self.atomOrbInfo[i] = (self.atomType[i],numOrbs)          

    def calcNuclRepulsion(self):
        
        self.NuclRepEnergy = 0
        Z = getAtomicCharge
        
        if self.numAtom > 1:
            
            #E_nuc = sum(Z_i * Z_j /(r_i - r_j) for all i,j)
            Eterms = (Z(self.atomType[i])*Z(self.atomType[j])/self.distanceMatrix[i,j]
                              for  i in xrange(self.numAtom) for j in xrange(i))
            
            NumTerms = self.numAtom * (self.numAtom - 1) / 2 #number of terms to be summed
            RepulsionTerms = np.fromiter(Eterms, np.float,NumTerms)
            self.NuclRepEnergy = np.sum(RepulsionTerms)        
  
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

        eig,eigVM = eigh(self.OverlapMatrix,turbo=True,check_finite=False)

        eigT = np.diag(np.reciprocal(np.sqrt(eig)))
        
        self.symS = np.dot(eigVM,np.dot(eigT,eigVM.T))


    def Fock(self):
        self.FockM = self.CoreHam + 2*np.einsum('kl,ijkl->ij',self.densityM,self.ERT)\
                                    -np.einsum('kl,ikjl->ij',self.densityM,self.ERT)

        
    def transFock(self,Fock): 
        
        self.transFockM = np.dot(self.symS.T,np.dot(Fock,self.symS))
        
        self.eig,eigVM = eigh(self.transFockM,turbo=True,check_finite=False)

        self.AOcoefs = np.dot(self.symS,eigVM)
    
    def densityMatrix(self):
        self.densityM = np.einsum('im ,jm ->ij', self.AOcoefs[:,:self.nOcc],
                                                 self.AOcoefs[:,:self.nOcc])   
                
    def ElectronicEnergy(self, Fock):
        self.ElectrEnergy = np.einsum('ij,ij', self.densityM, self.CoreHam + Fock)
       
        
    def MP2(self):
        
        #define N-number of orbitals, nOcc-number of occupied orbitals, MOERT-ERT in MO basis,
        #Orbenergy- energy of each orbital in ordered manner, C-Matrix of AO coefficients
        N,nOcc,MOERT,OrbEnergy,C = len(self.orbList),self.nOcc,np.copy(self.ERT),self.eig,self.AOcoefs
        
        #Convert Electron Repulsion Tensor from AO basis to MO basis
        MOERT = np.einsum('ijkl, ls -> ijks',MOERT,C)
        MOERT = np.einsum('ijks, kr -> ijrs',MOERT,C)
        MOERT = np.einsum('ijrs, jq -> iqrs',MOERT,C)
        MOERT = np.einsum('iqrs, ip -> pqrs',MOERT,C)                      
        
        self.MP2Energy = 0
        
        for i in xrange(nOcc):
            for j in xrange(nOcc):
                for a in xrange(nOcc,N):
                    for b in xrange(nOcc,N):
                        self.MP2Energy += MOERT[i,a,j,b]*(2*MOERT[i,a,j,b] - MOERT[i,b,j,a])/\
                                          (OrbEnergy[i]+OrbEnergy[j]-OrbEnergy[a]-OrbEnergy[b])
       
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
        
        self.DipoleMoments = np.array([MuX,MuY,MuZ])

    def MullikenPopulation(self):
        self.MullCharges = [0]*self.numAtom
        Z = getAtomicCharge
        runningIndex = 0
        P = np.einsum('ij,ij->i',self.densityM,self.OverlapMatrix)
        for i in xrange(self.numAtom):
            q = Z(self.atomOrbInfo[i][0])
            
            for j in xrange(runningIndex,runningIndex + self.atomOrbInfo[i][1]):
                q -= 2*P[j]
                
            self.MullCharges[i] = q
            runningIndex += self.atomOrbInfo[i][1]
        print self.MullCharges