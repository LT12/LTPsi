'''
Created on Feb 20, 2014
@author: Larry
'''
from Molecule import Molecule
from RotationalParam import RotationalParam
from SCF import SCF
import numpy as np
import cProfile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


def main():
    
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    mol = Molecule("molecules\meth.mol")
    print "Molecular Formula:" + mol.molecularFormula
    
#     print "Rotational Constants (cm ^ -1)"
#     print rot.RotationalConstants
    t1 = time.clock()
    scf = SCF(mol,"6-31G", dipole=True)
    t2 = time.clock()
    print t2-t1
    print scf.DipoleMoments
    print scf.TotalEnergy

if __name__ == '__main__':

    cProfile.run('main()')
    #cProfile.run('HydrogenSCFAnalysis(200,.4,7,"G")')
