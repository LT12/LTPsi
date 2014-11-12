from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("CprimMolInt", ["CprimMolInt.c"],
                  extra_compile_args=['-O3']),
        Extension("CMolecularIntegrals", ["CMolecularIntegrals.c"],
        extra_compile_args=['-O3'],
        include_dirs=[numpy.get_include()]),
        Extension("CAuxFuncs", ["CAuxFuncs.c"],extra_compile_args=['-O3']),
        
    ],
)
