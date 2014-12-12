#ifndef __PYX_HAVE__cPrimMolInt
#define __PYX_HAVE__cPrimMolInt


#ifndef __PYX_HAVE_API__cPrimMolInt

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(double) roots[5];
__PYX_EXTERN_C DL_IMPORT(double) weights[5];
__PYX_EXTERN_C DL_IMPORT(double) sRysN[10];
__PYX_EXTERN_C DL_IMPORT(double) sRys[10][10];
__PYX_EXTERN_C DL_IMPORT(double) Sa0[20];

#endif /* !__PYX_HAVE_API__cPrimMolInt */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcPrimMolInt(void);
#else
PyMODINIT_FUNC PyInit_cPrimMolInt(void);
#endif

#endif /* !__PYX_HAVE__cPrimMolInt */
