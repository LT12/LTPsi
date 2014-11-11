'''
Created on Feb 20, 2014

@author: Larry
'''
import numpy as np
import itertools as its
import math as m
import atomicParam as ap
from math import sqrt

class Molecule():
    
    def __init__(self, molFile):
        
        self.readMolFile(molFile)
        self.getDistanceMatrix(self.cartMatrix,self.bondMatrix,self.numBond,
                           self.numAtom)
        #self.getBondAngle(self.cartMatrix,self.bondMatrix,self.numBond,
        #                  self.numAtom,self.DistanceMatrix)
        self.getMolecularFormula(self.atomType)
        self.getMolecularMass(self.atomType,self.isotopeDiff)
        self.getMolecularCharge(self.atomCharge)
        self.getCenterofMass(self.atomType, self.cartMatrix, self.molecularMass)
        self.reposition()
    
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
        self.atomType = []
        # Lists any deviations in element atomic mass listed
        # in the mol file.
        self.isotopeDiff = []
        #Lists any charges on an atom in the same order as atomType
        self.atomCharge = []
        self.cartMatrix = np.zeros((self.numAtom,3),dtype=np.float)
        self.bondMatrix = np.zeros((self.numAtom,self.numAtom),dtype=np.int)
        
        atomTypeAppend,isotopeAppend,atomChargeAppend = self.atomType.append,self.isotopeDiff.append,self.atomCharge.append
       
        for x in xrange(4,4+self.numAtom):
            
            atomrow = lines[x].split()
            
            for y in xrange(0,3):
                self.cartMatrix[x-4,y] = atomrow[y]
                
            atomTypeAppend(atomrow[3])
            isotopeAppend(int(atomrow[4]))
            atomChargeAppend(int(atomrow[5]))
            
        for x in xrange(4+self.numAtom,4+self.numAtom+self.numBond):
            
            bondrow = lines[x].split()
            i = int(bondrow[0])-1
            j = int(bondrow[1])-1
            self.bondMatrix[i,j] = int(bondrow[2])
            self.bondMatrix[j,i] = int(bondrow[2]) 
            
        mF.close()

                    
    def getBondAngle(self,cartMatrix,bondMatrix,numBond,numAtom,DistanceMatrix):
        """ Determines the bond angle between all the groups of 3 bonded
         atoms in the molecule"""
        bondLength=np.multiply(bondMatrix,DistanceMatrix)
       
        # enumerate all the possible ways to arrange atoms in
        # the molecule into groups of 3 without repetition. 
        comb =  list(its.permutations(range(numAtom),3))
        
        self.bondAngle = np.zeros((len(comb),4))
        
        for x in xrange(0,len(comb)):
            
            #only compute the bond angle for bonded groups of 3 atoms.
            if ((bondMatrix[comb[x][0],comb[x][1]] !=  0) and (bondMatrix[comb[x][1],comb[x][2]] !=  0)):
                
                #determine the distance vectors along the internuclear axis
                #and takes the dot product of the two 
                vec1 = (np.subtract(cartMatrix[comb[x][0],:],
                                 cartMatrix[comb[x][1],:]))
                vec2 = (np.subtract(cartMatrix[comb[x][2],:],
                                  cartMatrix[comb[x][1],:]))
                dotp = np.dot(vec1,vec2)
                bondang= (180/np.pi)*(m.acos(dotp/(bondLength[comb[x][1],
                                    comb[x][0]]*bondLength[comb[x][1],comb[x][2]])))
                
                # computes the bond angle by taking the arccosine
                # of the dot product of the two vectors divided by
                # the product of the vector norms
                self.bondAngle[x,0] =  bondang
                self.bondAngle[x,1:] = [comb[x][0],comb[x][1],comb[x][2]]
        j=0;
        for x in xrange(0,np.size(self.bondAngle, 0)):
            if self.bondAngle[x-j,0]==0:
                self.bondAngle=np.delete(self.bondAngle, x-j, 0)
                j+=1
            
    def getDistanceMatrix(self,cartMatrix,bondMatrix,numBond,numAtom):
        """Determines the distance between each atom using the
        distance formula.Takes the Cartesian coordinate matrix,
        bonding matrix, number of bonds and atoms as required inputs.
        """
        #Looped methods, subtracting and norm
        subtract=np.subtract

        
        self.DistanceMatrix  = np.zeros((numAtom,numAtom),dtype=np.float)

        for i in xrange(0,numAtom):
            for j in xrange(0,i + 1):
                dist = subtract(cartMatrix[i,:],cartMatrix[j,:])
                self.DistanceMatrix[i,j] = np.sqrt(dist.dot(dist))
                self.DistanceMatrix[j,i] = self.DistanceMatrix[i,j]
                
                    
    def getMolecularCharge(self,atomCharge):
        self.molecularCharge = sum(atomCharge)
                
    def getMolecularFormula(self,atomType):
        self.molecularFormula = ""
        
        """sorts the atomType list alphabetically without
        duplicate elements"""
        sortedAtomType = sorted(list(set(atomType)))
        
        #append element and count (if not equal to 1) to
        #the molecular formula in alphabetical order
        for x in xrange(0,len(sortedAtomType)):
            self.molecularFormula += sortedAtomType[x]
            if atomType.count(sortedAtomType[x]) != 1:
                self.molecularFormula += str(atomType.count(sortedAtomType[x]))
        
    def getMolecularMass(self,atomType,isotopeDiff):
        AtomicMasses = [ap.getAtomicMass(atomType[i]) for i in xrange(0,self.numAtom)]
        self.molecularMass = sum(isotopeDiff) + sum(AtomicMasses)
        
    def getCenterofMass(self,atomType,cartMatrix,molecularMass):
        self.CenterofMass=np.zeros((1,3))
        for i in xrange(0,self.numAtom ):
            self.CenterofMass[0,0] += ap.getAtomicMass(atomType[i]) * cartMatrix[i,0]
            self.CenterofMass[0,1] += ap.getAtomicMass(atomType[i]) * cartMatrix[i,1]
            self.CenterofMass[0,2] += ap.getAtomicMass(atomType[i]) * cartMatrix[i,2]
        self.CenterofMass /= molecularMass
   
    def reposition(self):
        self.RposList = [np.sqrt(self.cartMatrix[i,:].dot(self.cartMatrix[i,:]))
                        for i in xrange(0,self.numAtom)]
        self.cartMatrix -= self.CenterofMass
        
    def setCartMatrix(self,cartMatrix):
        self.cartMatrix = cartMatrix
        self.getDistanceMatrix(self.cartMatrix,self.bondMatrix,self.numBond,
                           self.numAtom)
    
    def getCartMatrix(self):
        return self.cartMatrix