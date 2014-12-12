#  Orbital.py
#
#  Copyright 2014 Larry Tesler <ltesler@ufl.edu>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.


from cAuxFuncs import NormFac
import numpy as np


class Orbital(object):

    '''
    A class containing position and coefficient information on
    Contracted Gaussian-Type Orbitals (CGTO)
    
    Orbital i centered on atom A with nuclear coordinates (X_A,Y_A,Z_A):
    phi_i,A = [(x - X_A)^n_i * (y - Y_A)^m_i * (z - Z_A)^L_i] * Sum d_j * exp(-a_j * (x^2 + y^2 + z^2) )
    '''

    def __init__(self, orbData, center, atom):

        #Atom identity
        self.atom = atom
        #Carteisian center of orbital
        self.Center = center
        #angular momentum numbers
        self.qnums = getAngMom(orbData[0])
        #exponential coefficients
        self.a = np.array([cof[0] for cof in orbData[1]])
        #Number of primitive Gaussians
        self.nPrim = len(self.a)
        #Contraction Coefficients
        self.d = np.array([cof[1] * NormFac(cof[0], self.qnums[0],
                          self.qnums[1], self.qnums[2]) for cof in
                          orbData[1]])

    ####################################################################
    #                       Orbital Properties                         #
    ####################################################################

    @property
    def Center(self):
        return self._Center

    @Center.setter
    def Center(self, cent):
        self._Center = cent

    @property
    def qnums(self):
        return self._qnums

    @qnums.setter
    def qnums(self, q):
        self._qnums = q

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, aL):
        self._a = aL

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self,dL):
        self._d = dL

    @property
    def nPrim(self):
        return self._nPrim

    @nPrim.setter
    def nPrim(self,nP):
        self._nPrim = nP

def getAngMom(AngMom):

    AngMomDict = {"S":(0,0,0),
                  "PX":(1,0,0),"PY":(0,1,0),"PZ":(0,0,1),
                  "DX2":(2,0,0),"DY2":(0,2,0),"DZ2":(0,0,2),
                  "DXY":(1,1,0),"DXZ":(1,0,1),"DYZ":(0,1,1),
                  "FX3":(3,0,0),"FX2Y":(2,1,0),"FX2Z":(2,0,1),
                  "FY3":(0,3,0),"FY2X":(1,2,0),"FY2Z":(0,2,1),
                  "FZ3":(0,0,3),"FZ3":(0,0,3),"FZ2X":(1,0,2),
                  "FZ2Y":(0,1,2),"FXYZ":(1,1,1)}
    try:

        return AngMomDict[AngMom]

    except KeyError:

        print AngMom
        raise NameError("Bad Orbital!")
