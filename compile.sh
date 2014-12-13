# ! /bin/bash

cython2 cPrimMolInt.pyx
cython2 cAuxFuncs.pyx
cython2 cMolecularIntegrals.pyx
python2 setup.py build_ext --inplace
