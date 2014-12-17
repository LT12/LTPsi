"""molecule.py contains the Molecule class which creates the molecule
object. A object containing all of the attributes prescribed to
a molecule, including coordinates for each atom, identity of each atom,
molecular mass, molecular formula, bonding within the molecule, and
other attributes. In order to create a molecule object, a MDL MolFile,
following the MDL MolFile specifications, needs to be passed in,
which is then parsed by methods within the class.
"""
from __future__ import division
import numpy as np
from atomicParam import *
from scipy.constants import physical_constants


class Molecule():
    """Reads and parses a MDL .mol file, generating a molecule object
    
        Parameters
        ----------
        mol_file : .mol file
                   a MDL MolFile containing coordinates and atom info
        units : str
                units for the coordinate system; either Angstrom or Bohr
       
        Attributes
        ----------
        atom_type : array-like
                    a list containing the element of each atom in molecule

        units : str
                units for the coordinate system; either Angstrom or Bohr
        cart_matrix : ndarray
                      matrix containing coordinates for each atom
        bond_matrix : ndarray
                      matrix containing the type of bond between each atom
        molecular_mass : float
                         the molecular weight of the molecule in g per mol
        molecular_form : str
                         molecular formula of the molecule (alphabetically)
        center_of_mass : array-like
                         coordinates of the center of mass of the molecule
        num_atom : int
                   number of atoms in the molecule
        num_e : int
                number of electrons in molecule
        isotope_diff : array-like
                       an array where each element describes the
                       isotopic difference of the element from its
                       most abundant species
        atom_charge : array-like

       
        Notes
        -----
        Currently unstable, so a new Molecule object should be created
        instead of attempting to modify a molecule's parameters,
        which may lead to unexpected results or program crash.

    """

    def __init__(self, mol_file, units = 'bohr'):
        
        if units != 'bohr' and units != 'angstrom':
            raise NameError('Invalid unit selection')
        
        self.units = units

        # Molecule attributes
        # described in class docstring as their respective properties
        self._atom_type = None
        self._cart_matrix = None
        self._bond_matrix = None
        self._molecular_mass = 0
        self._com = None  # center of mass
        self._charge_list = None
        self._molecular_charge = None
        self._molecular_form = None

        self._read_mol_file(mol_file)
        self._comp_distance_matrix()
        # self.compBondAngle()  someday
        self._comp_mol_formula()
        self._comp_mol_mass()
        self._comp_mol_charge()
        self._comp_cent_mass()

    def _read_mol_file(self,mol_file):
        """reads an input mdl mol file line by line, determining
        the number of bonds and atoms in the molecule, a matrix
        of Cartesian coordinates for each atom, and a matrix with
        the bond connectivity for each atom"""

        m_f = open(mol_file, 'r')
        lines = m_f.readlines()
        """ 3rd line of mol file contains number of atoms and
         number of bonds"""
        atomCounts = lines[3].split()
        self.num_atom = int(atomCounts[0])
        self.num_bond = int(atomCounts[1])
        # A list containing the elements of the molecules as ordered
        # in the mol file.
        self._atom_type = [0] * self.num_atom
        # Lists any deviations in element atomic mass listed
        # in the mol file.
        self.isotope_diff = [0] * self.num_atom
        # Lists any charges on an atom in the same order as atom_type
        self._charge_list = [0] * self.num_atom
        self._cart_matrix = np.zeros((self.num_atom,3),dtype=np.float64)
        self._bond_matrix = np.zeros((self.num_atom,self.num_atom))

        for i in xrange(4,4+self.num_atom):

            atomrow = lines[i].split()

            for j in xrange(0,3):
                self._cart_matrix[i-4,j] = atomrow[j]

            self._atom_type[i-4] = atomrow[3]
            self.isotope_diff[i-4] = int(atomrow[4])
            self._charge_list[i-4] = int(atomrow[5])

        for i in xrange(4 + self.num_atom, 
                        4 + self.num_atom + self.num_bond):

            bondrow = lines[i].split()
            row = int(bondrow[0])-1
            col = int(bondrow[1])-1
            self._bond_matrix[row,col] = int(bondrow[2])
            self._bond_matrix[col,row] = int(bondrow[2])

        m_f.close()

    #~ def compBondAngle(self):
        #~ """ Determines the bond angle between all the groups of 3 bonded
         #~ atoms in the molecule"""
        #~ bondLength=np.multiply(self.bond_m,self.d)
#~ 
        #~ # enumerate all the possible ways to arrange atoms in
        #~ # the molecule into groups of 3 without repetition.
#~ 
        #~ self.bondAngle = np.zeros(num_atom,4)
        #~ for i in xrange(num_atom):
            #~ for j in xrange(i):
                #~ for k in xrange(j):
#~ 
                #~ #determine the distance vectors along the internuclear axis
                #~ #and takes the dot product of the two
                #~ vec1 = np.subtract(self.cart_matrix[i,:],
                                   #~ self.cart_matrix[j,:])
                #~ vec2 = np.subtract(self.cart_matrix[j,:],
                                   #~ self.cart_matrix[k,:])
                #~ dotp = np.dot(vec1,vec2)
                #~ bondang= (180 / pi)*(acos(dotp/(bondLength[i,j]
                                          #~ *bondLength[i,j])))

    def _comp_distance_matrix(self):
        """Determines the distance between each atom using the
        distance formula.Takes the Cartesian coordinate matrix,
        bonding matrix, number of bonds and atoms as required inputs.
        """

        self._distance_m  = np.zeros((self.num_atom,self.num_atom),
                                    dtype=np.float64)

        for i in xrange(self.num_atom):
            for j in xrange(i + 1):
                
                dist = self.cart_matrix[i,:] - self.cart_matrix[j,:]
                self._distance_m[i,j] = np.sqrt(dist.dot(dist))
                self._distance_m[j,i] = self._distance_m[i,j]

    def _comp_mol_charge(self):
        self._molecular_charge = sum(self._charge_list)

    def _comp_mol_formula(self):
        self._molecular_form = ""

        """sorts the atom_type list alphabetically without
        duplicate elements"""
        sortedAtomType = sorted(list(set(self._atom_type)))

        # append element and count (if not equal to 1) to
        # the molecular formula in alphabetical order
        for x in xrange(len(sortedAtomType)):
            self._molecular_form += sortedAtomType[x]
            if self.atom_type.count(sortedAtomType[x]) != 1:
                self._molecular_form += str(self.atom_type.count(sortedAtomType[x]))

    def _comp_mol_mass(self):
        atomic_masses = [getAtomicMass(self._atom_type[i])
                                          for i in xrange(self.num_atom)]
        self._molecular_mass = sum(self.isotope_diff) + sum(atomic_masses)

    def _comp_cent_mass(self):
        self._com = np.zeros((1,3),dtype = np.float64)

        for i in xrange(self.num_atom):
            mass = getAtomicMass(self._atom_type[i]) + self.isotope_diff[i]
            self._com += mass * self._cart_matrix[i,:]

        self._com /= self.molecular_mass
        self._cart_matrix -= self._com
    
    def to_bohr(self):
        """Convert units for coordinate system to Bohrs
        """
        if self.units == 'angstrom':
            self.cart_matrix /= (physical_constants['Bohr radius'][0] * 1E+10)
            self.units = 'bohr'
            self._comp_distance_matrix()
            self._comp_cent_mass()

            
    def to_angstrom(self):
        """Convert units for coordinate system to Angstroms
        """
        if self.units == 'bohr':
            self.cart_matrix *= (physical_constants['Bohr radius'][0] * 1E+10)
            self.units = 'angstrom'
            self._comp_distance_matrix()
            self._comp_cent_mass()
    #-------------------------------------------------------#
    #                   Molecule Properties                 #
    #-------------------------------------------------------#

    @property
    def cart_matrix(self):
        return self._cart_matrix

    @cart_matrix.setter
    def cart_matrix(self, cart_m):
        self._cart_matrix = cart_m
        self._comp_cent_mass()
        self._comp_distance_matrix()

    @property
    def distance_matrix(self):
        return self._distance_m

    @property
    def atom_type(self):
        return self._atom_type

    @atom_type.setter
    def atom_type(self,atomtype):
        self._atom_type = atomtype
        self._comp_mol_mass()
        self._comp_cent_mass()
        self._comp_mol_formula()
            
    @property
    def molecular_mass(self):
        return self._molecular_mass

    @property
    def molecular_charge(self):
        return self._molecular_charge

    @molecular_charge.setter
    def molecular_charge(self,molecular_charge):
        self._molecular_charge = molecular_charge

    @property
    def isotope_diff(self):
        return self.isotope_diff

    @isotope_diff.setter
    def isotope_diff(self,isotope_diff):
        self.isotope_diff = isotope_diff
        self._comp_mol_mass()
        self._comp_cent_mass()

    @property
    def center_of_mass(self):
        return self._com

    @property
    def molecular_formula(self):
        return self._molecular_form

    @property
    def num_e(self):
        return sum([getAtomicCharge(atom) for atom in self.atom_type])\
            - self.molecular_charge

    @property
    def atom_charge(self):
        return np.array([getAtomicCharge(atom) for atom in self.atom_type],
                        dtype=np.int64)
