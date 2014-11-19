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

    dx = 1E-8

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

def Optimize(mol,m=False,d=False,basis=""):

    cartNew = np.copy(mol.cartMatrix)
    cartOld = np.zeros(np.shape(cartNew))
    cartList = []
    precesion = 0.00001
    eps = 2
    grad=5
    
    while sp.linalg.norm(grad) > precesion:
        cartOld = cartNew
        grad = _calcGrad(mol,m,d,basis)
        print sp.linalg.norm(grad)
        cartNew = cartOld - eps*grad
        mol.cartMatrix = cartNew
        cartList.append(cartNew)
        eps = _LS(mol,m,d,basis,e=eps)
        print eps

    return cartList

def _LS(mol,m,d,bs,e=1):
    
    j,c,b = 0,0.1,0.75
    cart = mol.cartMatrix
    grad = calcGrad(mol,m,d,bs)
    norm2 = sp.linalg.norm(grad)**2
    
    while (SCFEnergy(mol,cart,m,d,bs) - 
                        SCFEnergy(mol,cart-e*grad,m,d,bs) <= c*e*norm2):
        e *= b
        
    return e
