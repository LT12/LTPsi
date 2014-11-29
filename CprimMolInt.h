#ifndef __PYX_HAVE__CprimMolInt
#define __PYX_HAVE__CprimMolInt


#ifndef __PYX_HAVE_API__CprimMolInt

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

#endif /* !__PYX_HAVE_API__CprimMolInt */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initCprimMolInt(void);
#else
PyMODINIT_FUNC PyInit_CprimMolInt(void);
#endif

#endif /* !__PYX_HAVE__CprimMolInt */
