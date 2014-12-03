from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("CprimMolInt", ["CprimMolInt.c"]),
        Extension("CMolecularIntegrals", ["CMolecularIntegrals.c"],
        include_dirs=[numpy.get_include()]),
        Extension("CAuxFuncs", ["CAuxFuncs.c"],),
        
    ],
)
