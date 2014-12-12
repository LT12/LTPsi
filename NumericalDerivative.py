'''
Created on Nov 9, 2014

@author: Larry

'''
from __future__ import print_function
from SCF import SCF
import numpy as np
import scipy as sp

def _Energy(mol,cart,m,d,b):
    res = SCF(mol,basis = b,MP2 = m, dipole = d, cartMatrix = cart)
    return res.TotalEnergy

def _calcDer(mol,atom,q,m,d,b):
    '''Calculate derivative of energy with respect
       to direction q for a specific atom in molecule
    '''

    #differential displacement
    dq = 1E-10

    cart_mdx = mol.cartMatrix
    cart_pdx = np.copy(cart_mdx)

    cart_mdx[atom,q] -= dq
    cart_pdx[atom,q] += dq

    dE = _Energy(mol,cart_pdx,m,d,b) - _Energy(mol,cart_mdx,m,d,b)

    cart_mdx[atom,q] += dq

    return dE/(2*dq)

def _calcGrad(mol,m,d,b):
    '''Calculate energy gradient for molecule
    
        mol - molecule
        m - whether to use MP2, Boolean True or False
        d - whether to calculate dipole moments, True or false
        b - basis set to use
    '''

    Grad = np.ndarray(np.shape(mol.cartMatrix))

    for i in xrange(len(mol.atomType)):
        for j in xrange(3):

            Grad[i,j] = _calcDer(mol,i,j,m,d,b)

    return Grad

def Optimize(mol, MP2 = False, d = False, basis = "STO3G",rend = None):

    cartNew = np.copy(mol.cartMatrix)
    cartOld = np.zeros(np.shape(cartNew))
    cartList = []
    precesion = 1E-5
    eps = 1
    grad = 1
    
    while sp.linalg.norm(grad) > precesion:
        cartOld = cartNew
        grad = _calcGrad(mol,MP2,d,basis)
        print(sp.linalg.norm(grad))
        cartNew = cartOld - eps*grad
        mol.cartMatrix = cartNew
        if rend is not None:
            rend.update(mol)
        cartList.append(cartNew)
        eps = _LS(mol,MP2,d,basis,e=eps)

    return cartList

def _LS(mol,m,d,bs,e=1):
    
    j,c,b = 0,0.3,0.7
    cart = mol.cartMatrix
    grad = _calcGrad(mol,m,d,bs)
    norm2 = sp.linalg.norm(grad)**2
    
    while (_Energy(mol,cart,m,d,bs) - 
                        _Energy(mol,cart-e*grad,m,d,bs) <= c*e*norm2):
        e *= b
        
    return e
