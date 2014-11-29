from termcolor import cprint
from pyfiglet import figlet_format
import numpy as np


def ResultDisplay(mol,res,MP2=False,dipole=False):
    cprint(figlet_format('LTPSI',font='starwars'))
    print 'Molecular Formula: ' + mol.molecularFormula
    print 'Molecular Charge: ' + str(mol.molecularCharge)
    print 'Molecular Mass: ' + str(mol.molecularMass)
    print 'Number of orbitals: ' + str(res.nOrb)
    print 'Number of occupied orbitals: ' +str(res.nOcc)
    print 'Number of primitive gaussians ' +str(res.numPrimGauss)
    print 'Integration Complete in: '+str(res.intTime)+' seconds'
    print 'SCF complete in '+str(res.itercount)+ ' iterations'
    print 'SCF Energy is equal to '+str(res.SCFEnergy)+' Hartrees'
    if MP2:
        print 'MP2 Energy is equal to '+str(res.MP2Energy)+' Hartrees'
        print 'Total Energy is equal to '+str(res.TotalEnergy)+' Hartrees'
    if dipole:
        print 'Dipole Moments: '+np.array_str(res.DipoleMoments)
    print 'Mulliken Charge Analysis: '
    for i in xrange(len(mol.atomType)):
        print mol.atomType[i]+": "+str(res.MullCharges[i])
    
    
