from atomicParam import *
import numpy as np
from visual import *
name = "Molecule"

class renderMolecule():
	
	def __init__(self,mol):
		self.mol = mol
		self.atomType = mol.atomType
		self.render(mol)
		
	def render(self,mol):
		self.atoms = [0] * len(self.atomType)
		self.bonds = []
		for i, cord in enumerate(mol.cartMatrix):

			col = getAtomColor(self.atomType[i])[0:3]
			rad = getAtomRadius(self.atomType[i])
			self.atoms[i] = sphere(pos = cord, color = col, radius = rad)
			
			for j in range(i):
				if mol.bondMatrix[i,j] != 0:
					
					dist = mol.cartMatrix[j,:] - cord
					self.bonds.append((cylinder(pos = cord,axis = dist ,radius = .1 ),i,j))
	
	def update(self,mol):
		
		for i, cord in enumerate(mol.cartMatrix):

			self.atoms[i].pos = cord
			
		for bond,i,j in self.bonds:
			dist = mol.cartMatrix[j,:] - mol.cartMatrix[i,:]
			bond.pos = mol.cartMatrix[i,:]
			bond.axis = dist