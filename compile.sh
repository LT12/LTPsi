# ! /bin/bash

cython2 cPrimMolInt.pyx
cython2 cMolecularIntegrals.pyx
cython2 c_molecular_integrals.pyx
python2 setup.py build_ext --inplace
