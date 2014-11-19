LTPsi
=====

Basic Restricted-Hartree Fock Quantum Chemistry Package in Python

Current Features:<br>
-Restricted Hartree Fock (RHF)<br>
-Several Basis Sets - STO-3G, 3-21G, 6-31G<br>
-Fast Two-Electron Integrals using Rys Polynomials (Numerical Rys Quadrature and Chebyshev-Gauss Quadrature)
(Implemented in Cython)

-Numerical Differentiation and Optimization<br>
-Rotational Constants<br>
-Dipole Moment Calculation<br>
-Mulliken Population Analysis<br>
-Moller-Posset Pertubation Theory (MP2) Energy Correction
<br>
Molecule Rendering with OpenGL
<br>
<hr>

Instructions:
<hr>
Windows: python setup.py build_ext --inplace compiler=mingw32<br>
Mac/Linux: python setup.py build_ext --inplace
