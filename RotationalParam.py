'''
Created on Sep 27, 2014

@author: Larry
'''
import numpy as np
import atomicParam as ap

class RotationalParam():
    '''
    Calculates Moment of Interia Tensor, Principle Moments of inertia, and Rotational Constant
    '''

    def __init__(self, cartMatrix, atomType):
        self.cartMatrix = cartMatrix
        self.getInertiaTensor(cartMatrix,atomType)
        self.getPrincipleMoments()
        self.getRotationalConstants(self.PrincipleMoments)
        

    def getInertiaTensor(self,cartMatrix,atomType):
        '''Calculates the Moment of Inertia Tensor of Molecule
            
        '''
        M = ap.getAtomicMass
        self.InertiaTensor=np.zeros((3,3))
        for i in xrange(0,len(atomType)):
            self.InertiaTensor[0,0] += M(atomType[i])*(cartMatrix[i,1]**2+cartMatrix[i,2]**2)
            self.InertiaTensor[1,1] += M(atomType[i])*(cartMatrix[i,0]**2+cartMatrix[i,2]**2)
            self.InertiaTensor[2,2] += M(atomType[i])*(cartMatrix[i,0]**2+cartMatrix[i,1]**2)
            self.InertiaTensor[0,1] += M(atomType[i])*(cartMatrix[i,0]*cartMatrix[i,1])
            self.InertiaTensor[0,2] += M(atomType[i])*(cartMatrix[i,0]*cartMatrix[i,2])
            self.InertiaTensor[1,2] += M(atomType[i])*(cartMatrix[i,1]*cartMatrix[i,2])
        self.InertiaTensor[1,0] = self.InertiaTensor[0,1]
        self.InertiaTensor[2,0] = self.InertiaTensor[0,2]
        self.InertiaTensor[2,1] = self.InertiaTensor[1,2]
            
        
    def getPrincipleMoments(self):
        '''Determines principle moments of inertia by diagonalizing inertia tensor
           and additionally rotates cartesian matrix onto principal axes
        '''
        self.PrincipleMoments,TensorAxes=np.linalg.eig(self.InertiaTensor)
        for i in xrange(0,len(self.cartMatrix[:,0])):
            self.cartMatrix[i,:] = np.transpose(np.dot(TensorAxes,np.transpose(self.cartMatrix[i,:])))
            
        
    def getRotationalConstants(self,PrincipleMoments):
        RotScale = 16.8576 #sets correct units for Principle Moments
        self.RotationalConstants = [RotScale / I for I in PrincipleMoments if I != 0]
