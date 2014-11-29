# ! /bin/bash

cython CprimMolInt.pyx
cython CAuxFuncs.pyx
cython CMolecularIntegrals.pyx
python setup.py build_ext --inplace
