'''
Orbital
@author: Larry
'''

from CAuxFuncs import NormFac
import numpy as np
class Orbital(object):
    '''
    A class containing position and coefficient information on Gaussian-Type Orbitals
    '''
    
    def __init__(self, orbData,center):
    
        #Carteisian center of orbital
        self.Center = center 
        #angular momentum numbers
        self.qnums = getAngMom(orbData[0])
        #exponential coefficients
        self.a = np.array([cof[0] for cof in orbData[1]],dtype=np.float64)
        
        #Number of primitive Gaussians
        self.nPrim = len(self.a)
        #Contraction Coefficients
        self.d = np.array([cof[1]*NormFac(cof[0], self.qnums[0],
                          self.qnums[1], self.qnums[2]) for cof in orbData[1]],dtype=np.float64)

def getAngMom(AngMom):
    AngMomDict = {"S":(0,0,0),"PX":(1,0,0),"PY":(0,1,0),"PZ":(0,0,1),"DX2":(2,0,0),
                 "DY2":(0,2,0),"DZ2":(0,0,2),"DXY":(1,1,0),"DXZ":(1,0,1),"DYZ":(0,1,1),
                 "FX3":(3,0,0),"FX2Y":(2,1,0),"FX2Z":(2,0,1),"FY3":(0,3,0),"FY2X":(1,2,0),
                 "FY2Z":(0,2,1),"FZ3":(0,0,3),"FZ3":(0,0,3),"FZ2X":(1,0,2),"FZ2Y":(0,1,2),
                 "FXYZ":(1,1,1)}
    try:
        return AngMomDict[AngMom]
    except KeyError:
        print AngMom
        raise NameError("Bad Orbital!")