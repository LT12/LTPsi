'''
Parameters for different atom types
'''

atomicMass = {"He":4.0026,"C":12.0107,"H":1.00794,"O":15.9994, "N":14}
atomicCharge = {"He":2,"C":6,"H":1,"O":8, "N":7}
atomicRadius = {"C":.4,"H":.25,"O":.5, "N":.4}
atomicColor = {"C": [1,1,1,1], "H":[1.5,1.5,1.5,1], "O":[1,0,0,1], "N":[0,0,1,1]}
atomicOrbitals = {"He":["1s"],"H":["1s"],"C":["1s","2s","2px","2py","2pz"],"O":["1s","2s","2px","2py","2pz"]}

def getAtomicMass(atom):
    return atomicMass[atom]

def getAtomicCharge(atom):
    return atomicCharge[atom]

def getAtomRadius(atom):
    return atomicRadius[atom]

def getAtomColor(atom):
    return atomicColor[atom]

def getAtomicOrbitals(atom):
    return atomicOrbitals[atom]
