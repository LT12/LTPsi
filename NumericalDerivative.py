'''
Created on Nov 9, 2014

@author: Larry

'''

from SCF import SCF
import numpy as np
import scipy as sp

def _SCFEnergy(mol,cart,m,d,b):
    res = SCF(mol,basis = b,MP2 = m, dipole = d, cartMatrix = cart)
    return res.TotalEnergy

def _calcDer(mol,i,j,m,d,b):

    dx = 1E-10

    cart = mol.cartMatrix
    cart1 = np.copy(cart)

    cart[i,j] -= dx
    cart1[i,j] += dx

    dE = _SCFEnergy(mol,cart1,m,d,b) - _SCFEnergy(mol,cart,m,d,b)

    cart[i,j] += dx

    return dE/(2*dx)

def _calcGrad(mol,m,d,b):

    Grad = np.ndarray(np.shape(mol.cartMatrix))

    for i in xrange(len(mol.atomType)):
        for j in xrange(3):

            Grad[i,j] = _calcDer(mol,i,j,m,d,b)

    return Grad

def Optimize(mol, MP2 = False, d = False, basis = "STO3G"):

    cartNew = np.copy(mol.cartMatrix)
    cartOld = np.zeros(np.shape(cartNew))
    cartList = []
    precesion = 1E-5
    eps = 1
    grad=5
    
    while sp.linalg.norm(grad) > precesion:
        cartOld = cartNew
        grad = _calcGrad(mol,MP2,d,basis)
        print sp.linalg.norm(grad)
        cartNew = cartOld - eps*grad
        mol.cartMatrix = cartNew
        cartList.append(cartNew)
        eps = _LS(mol,MP2,d,basis,e=eps)
        print eps

    return cartList

def _LS(mol,m,d,bs,e=1):
    
    j,c,b = 0,0.4,0.8
    cart = mol.cartMatrix
    grad = _calcGrad(mol,m,d,bs)
    norm2 = sp.linalg.norm(grad)**2
    
    while (_SCFEnergy(mol,cart,m,d,bs) - 
                        _SCFEnergy(mol,cart-e*grad,m,d,bs) <= c*e*norm2):
        e *= b
        
    return e
