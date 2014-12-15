'''
Created on Feb 20, 2014
@author: Larry
'''
from Molecule import Molecule
from RotationalParam import RotationalParam
from scf import SCF
import numpy as np


def main():
    
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    mol = Molecule("Molecules/wat.mol")
    print "Molecular Formula:" + mol.molecularFormula

    scf = SCF(mol)
    print scf.TotalEnergy
    print scf.intTime


if __name__ == '__main__':

   main()
    #cProfile.run('HydrogenSCFAnalysis(200,.4,7,"G")')
