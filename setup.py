from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("cPrimMolInt", ["cPrimMolInt.c"]),
        Extension("cMolecularIntegrals", ["cMolecularIntegrals.c"],
        include_dirs=[numpy.get_include()]),
        Extension("cAuxFuncs", ["cAuxFuncs.c"],),
        
    ],
)
