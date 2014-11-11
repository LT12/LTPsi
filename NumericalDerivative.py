'''
Created on Nov 9, 2014

@author: Larry
'''
from SCF import SCF
import numpy as np


def calcDer(mol,i,j,m,d):
    
    dx = 1E-9
    
    cart = mol.getCartMatrix()
    cart1 = np.copy(cart)
    
    x = cart[i,j]
    
    cart1[i,j] = x - dx
    
    mol.setCartMatrix(cart1)    
    scf = SCF(mol,"sto3G")
    E1 = scf.TotalEnergy+scf.MP2()

    cart1[i,j] = x + dx
    mol.setCartMatrix(cart1)
    scf = SCF(mol,"sto3G",MP2=m,dipole=d)
    E2 = scf.TotalEnergy

    mol.setCartMatrix(cart)

    return (E2-E1)/(2*dx)     
    
def calcGrad(mol,m,d):
    
    Grad = np.ndarray(np.shape(mol.getCartMatrix()))
    
    for i in xrange(len(mol.atomType)):
        for j in xrange(3):
            Grad[i,j] = calcDer(mol,i,j,m,d)     
    return Grad

def Optimize(mol,m=False,d=False):
    
    cartNew = np.copy(mol.getCartMatrix())   
    cartOld = np.zeros(np.shape(cartNew))
    cartList = [cartNew]
    
    eps = 1
    precesion = 0.00001

    while abs(np.linalg.norm(cartNew-cartOld))>precesion:
        
        cartOld = cartNew
        cartNew = cartOld - eps*calcGrad(mol,m,d)
        mol.setCartMatrix(cartNew)
        cartList.append(cartNew)
        eps *= 0.9
        
    print cartNew