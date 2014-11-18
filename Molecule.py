'''
Created on Feb 20, 2014

@author: Larry
'''
import numpy as np
from atomicParam import *
from math import sqrt, pi, acos
import itertools

class Molecule():

    def __init__(self, molFile):

        self.readMolFile(molFile)
        self.compDistanceMatrix()
        #self.compBondAngle()
        self.compMolecularFormula()
        self.compMolecularMass()
        self.compMolecularCharge()
        self.compCenterofMass()

    def readMolFile(self,molFile):
        """reads an input mdl mol file line by line, determining
        the number of bonds and atoms in the molecule, a matrix
        of Cartesian coordinates for each atom, and a matrix with
        the bond connectivity for each atom"""

        mF = open(molFile, 'r')
        lines = mF.readlines()
        """ 3rd line of mol file contains number of atoms and
         number of bonds"""
        atomCounts = lines[3].split()
        self.numAtom = int(atomCounts[0])
        self.numBond = int(atomCounts[1])
        # A list containing the elements of the molecules as ordered
        # in the mol file.
        self.atomType = [0] * self.numAtom
        # Lists any deviations in element atomic mass listed
        # in the mol file.
        self.isotopeDiff = [0] * self.numAtom
        #Lists any charges on an atom in the same order as atomType
        self.atomCharge = [0] * self.numAtom
        self.cartMatrix = np.zeros((self.numAtom,3),dtype=np.float64)
        self.bondMatrix = np.zeros((self.numAtom,self.numAtom))

        for i in xrange(4,4+self.numAtom):

            atomrow = lines[i].split()

            for j in xrange(0,3):
                self.cartMatrix[i-4,j] = atomrow[j]

            self.atomType[i-4] = atomrow[3]
            self.isotopeDiff[i-4] = int(atomrow[4])
            self.atomCharge[i-4] = int(atomrow[5])

        for i in xrange(4 + self.numAtom, 
                        4 + self.numAtom + self.numBond):

            bondrow = lines[i].split()
            row = int(bondrow[0])-1
            col = int(bondrow[1])-1
            self.bondMatrix[row,col] = int(bondrow[2])
            self.bondMatrix[col,row] = int(bondrow[2])

        mF.close()

    #~ def compBondAngle(self):
        #~ """ Determines the bond angle between all the groups of 3 bonded
         #~ atoms in the molecule"""
        #~ bondLength=np.multiply(self.bondMatrix,self.distanceMatrix)
#~ 
        #~ # enumerate all the possible ways to arrange atoms in
        #~ # the molecule into groups of 3 without repetition.
#~ 
        #~ self.bondAngle = np.zeros(numAtom,4)
        #~ for i in xrange(numAtom):
            #~ for j in xrange(i):
                #~ for k in xrange(j):
#~ 
                #~ #determine the distance vectors along the internuclear axis
                #~ #and takes the dot product of the two
                #~ vec1 = np.subtract(self.cartMatrix[i,:],
                                   #~ self.cartMatrix[j,:])
                #~ vec2 = np.subtract(self.cartMatrix[j,:],
                                   #~ self.cartMatrix[k,:])
                #~ dotp = np.dot(vec1,vec2)
                #~ bondang= (180 / pi)*(acos(dotp/(bondLength[i,j]
                                          #~ *bondLength[i,j])))

    def compDistanceMatrix(self):
        """Determines the distance between each atom using the
        distance formula.Takes the Cartesian coordinate matrix,
        bonding matrix, number of bonds and atoms as required inputs.
        """
        #Looped methods, subtracting and norm
        self.distanceMatrix  = np.zeros((self.numAtom,self.numAtom),
                                                      dtype=np.float64)

        for i in xrange(self.numAtom):
            for j in xrange(i + 1):
				
                dist = self.cartMatrix[i,:] - self.cartMatrix[j,:]
                self.distanceMatrix[i,j] = np.sqrt(dist.dot(dist))
                self.distanceMatrix[j,i] = self.distanceMatrix[i,j]

    def compMolecularCharge(self):
        self.molecularCharge = sum(self.atomCharge)

    def compMolecularFormula(self):
        self.molecularFormula = ""

        """sorts the atomType list alphabetically without
        duplicate elements"""
        sortedAtomType = sorted(list(set(self.atomType)))

        #append element and count (if not equal to 1) to
        #the molecular formula in alphabetical order
        for x in xrange(len(sortedAtomType)):
            self.molecularFormula += sortedAtomType[x]
            if self.atomType.count(sortedAtomType[x]) != 1:
                self.molecularFormula += str(self.atomType.count(sortedAtomType[x]))

    def compMolecularMass(self):
        AtomicMasses = [getAtomicMass(self.atomType[i])
										  for i in xrange(self.numAtom)]
        self.molecularMass = sum(self.isotopeDiff) + sum(AtomicMasses)

    def compCenterofMass(self):
        self.CenterofMass = np.zeros((1,3),dtype = np.float64)

        for i in xrange(self.numAtom):
            mass = getAtomicMass(self.atomType[i]) + self.isotopeDiff[i]
            self.CenterofMass += mass * self.cartMatrix[i,:]

        self.CenterofMass /= self.molecularMass
        self.cartMatrix -= self.CenterofMass

    #-------------------------------------------------------#
    #                   Molecule Properties                 #
    #-------------------------------------------------------#

    @property
    def cartMatrix(self):
        return self.cartMatrix

    @cartMatrix.setter
    def cartMatrix(self,cartMatrix):
        self.cartMatrix = cartMatrix
        self.compDistanceMatrix(self.cartMatrix, self.bondMatrix,
                                self.numBond, self.numAtom)
    @property
    def atomType(self):
        return self.atomType

    @atomType.setter
    def atomType(self,atomType):
        self.atomType = atomType
        self.compMolecularMass()
        self.compCenterofMass()
        self.compMolecularFormula()
            
    @property
    def molecularMass(self):
        return self.molecularMass

    @property
    def molecularCharge(self):
        return self.molecularCharge

    @molecularCharge.setter
    def molecularCharge(self,molecularCharge):
        self.molecularCharge = molecularCharge

    @property
    def isotopeDiff(self):
        return self.isotopeDiff
    
    @property
    def distanceMatrix(self):
        return self.distanceMatrix

    @isotopeDiff.setter
    def isotopeDiff(self,isotopeDiff):
        self.isotopeDiff = isotopeDiff
        self.compMolecularMass()
        self.compCenterofMass()

    @property
    def CenterofMass(self):
        return self.CenterofMass
