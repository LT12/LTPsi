#cython: profile=True
'''
Created on Oct 20, 2014

@author: Larry
'''
cimport cython
import numpy as np
cimport numpy as np
from atomicParam import *
from CprimMolInt cimport primOverlapIntegral, primNuclearAttractionIntegral,primElecRepulInt



class MolecularIntegrals():
    '''
    classdocs
    '''

    def __init__(self, object orbList, object atomType, np.ndarray[np.float64_t,ndim=2] cartMatrix,int dipole):
        self.atomType, self.cartMatrix = atomType, cartMatrix
        self.NumOrbitals, self.orbList = len(orbList), orbList

        self.OverlapMatrix()
        self.CoreHamiltonian()
        self.ERT = ElectronRepulsionTensor(self.NumOrbitals,orbList)
        
        if dipole:
            self.DipoleMatrix()
        
    def OverlapMatrix(self):
        cdef int N,i,j
        N = self.NumOrbitals # number of orbitals
        self.OLMatrix = np.zeros((N,N),dtype=np.float64) #storage

        for i in xrange(N):
            for j in xrange(i+1):
                
                self.OLMatrix[i,j] = OverlapIntegral(self.orbList[i],self.orbList[j])
                
                self.OLMatrix[j,i] = self.OLMatrix[i,j]
                
    def DipoleMatrix(self):
        cdef int N,i,j
        N = self.NumOrbitals # number of orbitals
        self.XDipoleMatrix = np.zeros((N,N),dtype=np.float64) #storage
        self.YDipoleMatrix = np.zeros((N,N),dtype=np.float64) #storage
        self.ZDipoleMatrix = np.zeros((N,N),dtype=np.float64) #storage

        for i in xrange(N):
            for j in xrange(i+1):
                
                self.XDipoleMatrix[i,j] = -DipoleIntegral(self.orbList[i],self.orbList[j],0)
                self.YDipoleMatrix[i,j] = -DipoleIntegral(self.orbList[i],self.orbList[j],1)
                self.ZDipoleMatrix[i,j] = -DipoleIntegral(self.orbList[i],self.orbList[j],2)
                self.XDipoleMatrix[j,i] = self.XDipoleMatrix[i,j]
                self.YDipoleMatrix[j,i] = self.YDipoleMatrix[i,j]
                self.ZDipoleMatrix[j,i] = self.ZDipoleMatrix[i,j]
                
    def KIMatrix(self):
        '''Constructs matrix of kinetic energy integrals
        '''
        cdef int N,i,j
        N = self.NumOrbitals # number of orbitals
        cdef double [:,:] KineticMatrix = np.zeros((N,N),dtype=np.float64)
        
        for i in xrange(N):
            for j in xrange(i+1):
                
                KineticMatrix[i,j] = KineticIntegral(self.orbList[i],self.orbList[j])
               
        return KineticMatrix
                    
    def NucAtractMatrix(self):
        '''Constructs matrix of nuclear attraction integrals
        '''
        cdef int N,i,j
        N = self.NumOrbitals # number of orbitals
        cdef double [:,:] NuclearMatrix = np.zeros((N,N),dtype=np.float64)
	
        for i in xrange(N):
            for j in xrange(i+1):
            
                NuclearMatrix[i,j] = NuclearIntegral(self.orbList[i],self.orbList[j],self.atomType,self.cartMatrix)
                
        return NuclearMatrix
 


                                    
    def CoreHamiltonian(self):
        '''Constructs core hamiltonian from single-electron integral matricies
        '''
        cdef int N,i,j
        N = self.NumOrbitals
        cdef double [:,:] KI = self.KIMatrix()
        cdef double [:,:] NAI = self.NucAtractMatrix()
        self.CoreHam = np.ndarray((N,N),np.float64)
 
        for i in xrange(N):
            for j in xrange(i+1):
                self.CoreHam[i,j] = KI[i,j] + NAI[i,j]
                self.CoreHam[j,i] = self.CoreHam[i,j]
                
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:,:,:] ElectronRepulsionTensor(int N, object orbList):
    cdef int i,j,k,l
    cdef double [:,:,:,:] ERT = np.ndarray((N,N,N,N),dtype=np.float64)

    for i in xrange(N):
        for j in xrange(i+1):
            for k in xrange(N):
                for l in xrange(k+1):
                    if (i+1) * (j+1) >= (k+1) * (l+1):

                        ERT[i,j,k,l] = ElecRepIntegral(orbList[i],orbList[j],
                                                       orbList[k],orbList[l])
                        ERT[j,i,k,l] = ERT[i,j,k,l]
                        ERT[i,j,l,k] = ERT[i,j,k,l]
                        ERT[j,i,l,k] = ERT[i,j,k,l]
                        ERT[k,l,i,j] = ERT[i,j,k,l]
                        ERT[l,k,i,j] = ERT[i,j,k,l]
                        ERT[k,l,j,i] = ERT[i,j,k,l]
                        ERT[l,k,j,i] = ERT[i,j,k,l]
    return ERT
                

#Integration Methods:
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double OverlapIntegral(object orb1, object orb2):

    cdef int xnum1,ynum1,znum1,xnum2,ynum2,znum2,nPrim1,nPrim2,i,j
    cdef double Integral, Const
    pOL = primOverlapIntegral
    Integral=0

    #get Angular Cartesian numbers for orbitals

    xnum1,ynum1,znum1 = orb1.qnums
    xnum2,ynum2,znum2 = orb2.qnums
    
    #Center of each orbital
    cdef double [:] cent1 = orb1.Center
    cdef double [:] cent2 = orb2.Center
    
    #number of primitive Gaussians for each orbital

    nPrim1,nPrim2 = orb1.nPrim,orb2.nPrim
    
    #Coefficients for primitive Gaussians for each orbital
    cdef double [:] d1 = orb1.d
    cdef double [:] d2 = orb2.d
    cdef double [:] a1 = orb1.a
    cdef double [:] a2 = orb2.a

    #Sum over all GTOs in basis set
    for i in xrange(nPrim1):
        for j in xrange(nPrim2):

            Const = d1[i] * d2[j]   #Product of contraction coefficients
            Integral += Const * pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],
                                    a1[i],a2[j],xnum1,ynum1,znum1,xnum2,ynum2,znum2)            
    return Integral
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double DipoleIntegral(object orb1,object orb2,int cart):

    cdef int xnum1,ynum1,znum1,xnum2,ynum2,znum2,nPrim1,nPrim2,i,j,qnumA,qnumB
    cdef double Integral, Const
    
    pOL = primOverlapIntegral
    Integral = 0

    #get Angular Cartesian numbers for orbitals

    xnum1,ynum1,znum1 = orb1.qnums
    xnum2,ynum2,znum2 = orb2.qnums
    
    #Center of each orbital
    cdef double [:] cent1 = orb1.Center
    cdef double [:] cent2 = orb2.Center
    
    #number of primitive Gaussians for each orbital
    nPrim1,nPrim2 = orb1.nPrim,orb2.nPrim
    
    #Coefficients for primitive Gaussians for each orbital
    cdef double [:] d1 = orb1.d
    cdef double [:] d2 = orb2.d
    cdef double [:] a1 = orb1.a
    cdef double [:] a2 = orb2.a
    
    cdef double P,Pam1,Pbm1,p,X
    #Sum over all GTOs in basis set
    for i in xrange(nPrim1):
        for j in xrange(nPrim2):
            
            Const = d1[i] * d2[j]   #Product of contraction coefficients
            #dipole integral is a linear of combination of overlap integrals
           
            P = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                    ynum1,znum1,xnum2,ynum2,znum2)
            
            if cart == 0:
            
                Pam1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1-1,
                                      ynum1,znum1,xnum2,ynum2,znum2)
                Pbm1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2-1,ynum2,znum2)
                qnumA = xnum1
                qnumB = xnum2
            
            if cart == 1:
            
                Pam1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1-1,znum1,xnum2,ynum2,znum2)
                Pbm1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2-1,znum2)
                qnumA = ynum1
                qnumB = ynum2
            
            if cart == 2:
            
                Pam1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1-1,xnum2,ynum2,znum2)
                Pbm1 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2,znum2-1)
                qnumA = znum1
                qnumB = znum2
                
            p = 1/(a1[i] + a2[j])
            X = (a1[i]*cent1[cart] + a2[j]*cent2[cart]) * p
            Integral += Const*(X * P + 0.5 * p * (qnumA * Pam1 + qnumB * Pbm1))
    
    return Integral

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double KineticIntegral(object orb1,object orb2):

    cdef int xnum1,ynum1,znum1,xnum2,ynum2,znum2,nPrim1,nPrim2,i,j
    cdef double Integral, Const
    
    pOL = primOverlapIntegral
    Integral = 0

    #get Angular Cartesian numbers for orbitals

    xnum1,ynum1,znum1 = orb1.qnums
    xnum2,ynum2,znum2 = orb2.qnums
    
    #Center of each orbital
    cdef double [:] cent1 = orb1.Center
    cdef double [:] cent2 = orb2.Center
    
    #number of primitive Gaussians for each orbital

    nPrim1,nPrim2 = orb1.nPrim,orb2.nPrim
    
    #Coefficients for primitive Gaussians for each orbital
    cdef double [:] d1 = orb1.d
    cdef double [:] d2 = orb2.d
    cdef double [:] a1 = orb1.a
    cdef double [:] a2 = orb2.a
    
    cdef double P,Px2,Py2,Pz2,Pxm2,Pym2,Pzm2
    #Sum over all GTOs in basis set
    for i in xrange(nPrim1):
        for j in xrange(nPrim2):
            
            
            Const = d1[i] * d2[j]   #Product of contraction coefficients
            #Kinetic energy integral is a linear of combination of overlap integrals
           
            P = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                    ynum1,znum1,xnum2,ynum2,znum2)
            
            Px2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2+2,ynum2,znum2)
            Py2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2+2,znum2)
            Pz2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2,znum2+2)
            
            Pxm2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2-2,ynum2,znum2)
            Pym2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2-2,znum2)
            Pzm2 = pOL(cent1[0],cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],a2[j],xnum1,
                                      ynum1,znum1,xnum2,ynum2,znum2-2)

            
            Integral += Const*(a2[j]*(2*(xnum2+ynum2+znum2)+3)*P-(2*a2[j]**2)*(Px2+Py2+Pz2)
                               -0.5*(xnum2*(xnum2-1)*Pxm2 + ynum2*(ynum2-1)*Pym2 + znum2*(znum2-1)*Pzm2))
    
    return Integral
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double NuclearIntegral(object orb1,object orb2,object atomType,double [:,:] cartN):

    cdef int xnum1,ynum1,znum1,xnum2,ynum2,znum2,nPrim1,nPrim2,i,j
    cdef double Integral, Const
    Integral = 0
    #get Angular Cartesian numbers for orbitals
    xnum1,ynum1,znum1 = orb1.qnums
    xnum2,ynum2,znum2 = orb2.qnums
    
    #Center of each orbital
    cdef double [:] cent1 = orb1.Center
    cdef double [:] cent2 = orb2.Center
    
    #number of primitive Gaussians for each orbital
    nPrim1,nPrim2 = orb1.nPrim,orb2.nPrim
    
    #Coefficients for primitive Gaussians for each orbital
    cdef double [:] d1 = orb1.d
    cdef double [:] d2 = orb2.d
    cdef double [:] a1 = orb1.a
    cdef double [:] a2 = orb2.a
    #assign functional calls to avoid looping over them--optimization

    primNuc = primNuclearAttractionIntegral
    #Sum over all GTOs in basis set
    for i in xrange(nPrim1):
        for j in xrange(nPrim2):
            
            Const = d1[i] * d2[j]   #Product of contraction coefficients
            
            for k in enumerate(atomType):
                Z = getAtomicCharge(k[1])
                Integral += -Z * Const * primNuc(cartN[k[0],0],cartN[k[0],1],cartN[k[0],2],cent1[0],
                                                 cent1[1],cent1[2],cent2[0],cent2[1],cent2[2],a1[i],
                                                 a2[j],xnum1,ynum1,znum1,xnum2,ynum2,znum2)
    return Integral

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ElecRepIntegral(object orb1,object orb2,object orb3,object orb4):
    
    cdef double Integral,Const
    cdef int i,j,k,l,nPrim1,nPrim2,nPrim3,nPrim4
    Integral = 0

    #get Angular Cartesian numbers for orbitals
    cdef int x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4
    x1,y1,z1 = orb1.qnums
    x2,y2,z2 = orb2.qnums
    x3,y3,z3 = orb3.qnums
    x4,y4,z4 = orb4.qnums

    #number of primitive Gaussians for each orbital
    nPrim1,nPrim2,nPrim3,nPrim4 = orb1.nPrim,orb2.nPrim,orb3.nPrim,orb4.nPrim
    
    #Coefficients for primitive Gaussians for each orbital
    cdef double [:] d1 = orb1.d
    cdef double [:] d2 = orb2.d
    cdef double [:] d3 = orb3.d
    cdef double [:] d4 = orb4.d
    cdef double [:] a1 = orb1.a
    cdef double [:] a2 = orb2.a
    cdef double [:] a3 = orb3.a
    cdef double [:] a4 = orb4.a
    cdef double [:] Center1 = orb1.Center
    cdef double [:] Center2 = orb2.Center
    cdef double [:] Center3 = orb3.Center
    cdef double [:] Center4 = orb4.Center
    
    #Sum over all GTOs in basis set
    for i in xrange(nPrim1):
        for j in xrange(nPrim2):
            for k in xrange(nPrim3):
                for l in xrange(nPrim4):
                
                    Const = d1[i] * d2[j] * d3[k] * d4[l]

                    Integral += Const * primElecRepulInt(Center1[0],Center1[1],Center1[2],
                                                         Center2[0],Center2[1],Center2[2],
                                                         Center3[0],Center3[1],Center3[2],
                                                         Center4[0],Center4[1],Center4[2],
                                                         a1[i],a2[j],a3[k],a4[l],x1,x2,x3,
                                                         x4,y1,y2,y3,y4,z1,z2,z3,z4)                
    return Integral