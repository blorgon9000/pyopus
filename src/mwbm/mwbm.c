/* (c)2013 Arpad Buermen */
/* Interface to the FORTRAN implementation of the More-Wild test functions. */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <stdio.h>
#include "f2c.h"
// Included by f2c.h AB
// #include <math.h> 

/* Debug switch - uncomment to enable debug messages */
/* #undefine PYDEBUG */

/* Debug file */
#define df stdout

#ifdef LINUX
#define __declspec(a) extern
#endif

/* Safeguard against C++ symbol mangling */
#ifdef __cplusplus
extern "C" {
#endif

/* lvumm.c */
/* Number of parameters for the problems */
/*               1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17  18 19 20 21 22 */
int n_problem[]={0,0,0,2,3,4,2,3,4, 3, 0, 3, 2, 4, 0, 0, 5, 11, 0, 0, 0, 8};

static char dfoxs_doc[]=
"Initial points for the More-Wild problems.\n"
"Wraps FORTRAN function dfoxs.\n"
"\n"
"x0=dfoxs(num, scale)\n"
"\n"
"Input\n"
"num   -- problem number (0-21)\n"
"n     -- problem dimension\n"
"scale -- scaling factor\n"
"\n"
"Output\n"
"x0 -- initial point (array)\n"; 

static PyObject *dfoxs_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n; 
	double i_scale;
	integer n, problem;
	doublereal scale, *xini;
	PyArrayObject *Mx0; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "dfoxs: checking arguments\n");
#endif
	if (PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes three arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iid", &i_problem, &i_n, &i_scale)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	
	if (problem<1 || problem>22) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=i_n;
	
	/* Check n */
	if (n_problem[i_problem]!=0 && n_problem[i_problem]!=i_n || i_n<=0) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	/* Set scale */
	scale=i_scale;
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	dfoxs_(&n, xini, &problem, &scale);
	
	return (PyObject *)Mx0; 
}

static char dfovec_doc[]=
"Partial function values for the More-Wild problems.\n"
"Wraps FORTRAN function dfovec.\n"
"\n"
"fi=dfovec(m, n, x, num)\n"
"\n"
"Input\n"
"m   -- number of partial functions\n"
"x   -- point at which the functions are evaluated (array of length n)\n"
"num -- problem number (0-21)\n"
"\n"
"Output\n"
"fi -- array of length m holding the partial function values\n"; 

static PyObject *dfovec_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n, i_m; 
	integer n, m, problem;
	doublereal *f, *x; 
	PyArrayObject *Mx, *Mf; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "dfovec: checking arguments\n");
#endif
	if (PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes three arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iOi", &i_m, &Mx, &i_problem)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	problem=i_problem+1;
	
	if (problem<1 || problem>22) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length n");
		return NULL; 
	}
	
	i_n=PyArray_DIM(Mx, 0);
	
	if (n_problem[i_problem]!=0 && n_problem[i_problem]!=i_n || i_n<=0) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	n=i_n; 
	
	if (i_m<0) {
		PyErr_SetString(PyExc_Exception, "Bad number of partial functions.");
		return NULL; 
	}
	
	m=i_m;
	
	/* Allocate vector for f */
	dims[0]=i_m;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	dfovec_(&m, &n, x, f, &problem);
	
	return (PyObject *)Mf; 
}


/* Methods table */
static PyMethodDef _mwbm_methods[] = {
	{"dfoxs", dfoxs_wrap, METH_VARARGS, dfoxs_doc},
	{"dfovec", dfovec_wrap, METH_VARARGS, dfovec_doc},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_mwbm()  {
	(void) Py_InitModule("_mwbm", _mwbm_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_mwbm_amd64()  {
	(void) Py_InitModule("_mwbm_amd64", _mwbm_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_mwbm_i386()  {
	(void) Py_InitModule("_mwbm_i386", _mwbm_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
