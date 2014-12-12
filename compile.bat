cython cPrimMolInt.pyx
cython cAuxFuncs.pyx
cython cMolecularIntegrals.pyx
python setup.py build_ext --inplace --compiler=mingw32

pause