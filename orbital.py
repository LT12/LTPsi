"""orbital.py contains the Orbital class used for containing of the
attributes of a contracted Gaussian type orbital (GTO). Additionally,
this modules contains functions for calculating normalization constants
for orbitals (NormFac), angular quantum numbers (get_ang_mom), and
generating a list of orbital object for a given molecule and basis set
(det_orbs).
"""

# orbital.py
#
#  Copyright 2014 Larry Tesler <teslerlarry@gmail.com>
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


from __future__ import print_function
from cAuxFuncs import NormFac
import numpy as np
from importlib import import_module


class Orbital(object):
    """
    A class containing position and coefficient information on
    Contracted Gaussian-Type Orbitals (CGTO)

    Parameters
    ----------
    orb_data : list
               contains coefficients and angular momentum of orbital
    center : float ndarray
             coordinates for the center of orbital
    atom : int
           number of the atom on which of the orbital is centered on

    Notes
    -----
    Orbital i centered on atom A with nuclear coordinates (X_A,Y_A,Z_A)

    .. math::
    \phi_{i,A} = [(x - X_A)^n_i * (y - Y_A)^m_i * (z - Z_A)^L_i] * \sum_j d_j * e^(-a_j * (x^2 + y^2 + z^2) )
    """

    def __init__(self, orb_data, center, atom):

        # Atom identity
        self.atom = atom
        # Cartesian center of orbital
        self.center = center
        # angular momentum numbers
        self.qnums = get_ang_mom(orb_data[0])
        # exponential coefficients
        self.a = np.array([cof[0] for cof in orb_data[1]])
        # Number of primitive Gaussians
        self.n_prim = len(self.a)
        # Contraction Coefficients
        self.d = np.array([cof[1] * NormFac(cof[0], self.qnums[0],
                                            self.qnums[1], self.qnums[2]) for cof in
                           orb_data[1]])

    ####################################################################
    #                       Orbital Properties                         #
    ####################################################################

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, cent):
        self._center = cent

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
    def a(self, a_l):
        self._a = a_l

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d_l):
        self._d = d_l

    @property
    def n_prim(self):
        return self._n_prim

    @n_prim.setter
    def n_prim(self, n_p):
        self._n_prim = n_p


def get_ang_mom(ang_mom):
    """Retrieves angular quantum numbers for a given angular momentum

    Parameters
    ----------
    ang_mom : str
              angular momentum: S, P, D, F...

    Returns
    -------
    tuple
          3 angular quantum numbers for each Cartesian direction

    Raises
    ------
    KeyError
            if invalid angular momentum is used
    """
    ang_mom_dict = {"S": (0, 0, 0),
                    "PX": (1, 0, 0),  "PY": (0, 1, 0),   "PZ": (0, 0, 1),
                    "DX2": (2, 0, 0), "DY2": (0, 2, 0),  "DZ2": (0, 0, 2),
                    "DXY": (1, 1, 0), "DXZ": (1, 0, 1),  "DYZ": (0, 1, 1),
                    "FX3": (3, 0, 0), "FX2Y": (2, 1, 0), "FX2Z": (2, 0, 1),
                    "FY3": (0, 3, 0), "FY2X": (1, 2, 0), "FY2Z": (0, 2, 1),
                    "FZ3": (0, 0, 3), "FZ2X": (1, 0, 2), "FZ2Y": (0, 1, 2),
                    "FXYZ": (1, 1, 1)}
    try:

        return ang_mom_dict[ang_mom]

    except KeyError:

        print(ang_mom)
        raise NameError("Bad Orbital!")


def det_orbs(basis, atom_type, cart_matrix):
    """Determines the atomic orbitals to be used for the molecule

    Parameters
    ----------
    basis: str
           Basis set to be used for calculations
    atom_type: str list
               List of all atoms in molecule as element names
    cart_matrix: float ndarray
                 Matrix of Cartesian coordinates for each atom
    Returns
    -------
    orbs : list
           list of atomic orbital objects
    num_prim : int
               total number of primitive GTOs in system

    Notes
    -----
    basis sets can be added by adding an entry
    to 'basis_sets' dictionary. The key should
    be the name of the basis set, while the value
    should be the name of the python module
    containing the basis set.

    Raises
    ------
    KeyError
            if an invalid basis set is chosen
    """

    basis_sets = {"STO3G": "STO3G", "STO6G": "STO6G", "3-21G": "b321G",
                  "6-31G": "b631G", "6-31G*": "b631Gs", "6-31G**": "b631Gss",
                  "6-31++G": "b631ppG", "6-311G": "b6311G", "6-311++G": "b6311ppG",
                  "6-311++G**": "b6311ppGss", "DZ": "DZ", "DZP": "DZP",
                  "MINI": "MINI"}

    try:
        bs = import_module("basis." + basis_sets[basis])
    except KeyError:
        raise NameError("Bad basis set!")

    orbs = []
    append = orbs.append
    num_prim = 0

    for i in xrange(len(atom_type)):
        orb = bs.getOrbs(atom_type[i])
        n = len(orb)
        for j in xrange(n):
            if orb[j][0] == "S":
                append(Orbital(orb[j], cart_matrix[i, :], i))
            elif orb[j][0] == "P":
                append(Orbital(("PX", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("PY", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("PZ", orb[j][1]),
                               cart_matrix[i, :], i))
            elif orb[j][0] == "D":
                append(Orbital(("DX2", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("DY2", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("DZ2", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("DXZ", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("DXY", orb[j][1]),
                               cart_matrix[i, :], i))
                append(Orbital(("DYZ", orb[j][1]),
                               cart_matrix[i, :], i))
            else:
                print (orb[j][0])
                raise NameError("bad orbital!")

    for orb in orbs:
        num_prim += orb.n_prim

    return orbs, num_prim