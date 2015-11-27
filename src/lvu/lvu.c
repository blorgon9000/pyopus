/* (c)2013 Arpad Buermen */
/* Interface to the FORTRAN implementation of the Luksan-Vlcek test functions. */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include "f2c.h"

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

/* This zeros a vector - it is an external dependency of test28.f */
int uxvset_(integer *n, doublereal *val, doublereal *x) {
	integer i;
	for(i=0;i<*n;i++) {
		x[i]=*val;
	}
	return 0;
}

/* test28.c */

static char tiud28_doc[]=
"Unconstrained problems - problem information.\n"
"Wraps FORTRAN function tiud28.\n"
"\n"
"data=tiud28(num)\n"
"\n"
"Input\n"
"num -- problem number (0-91)\n"
"n   -- problem dimension\n"
"\n"
"Output\n"
"data -- dictionary with problem information\n"
"\n"
"The dictionary has the following members:\n"
"x0     -- initial point (NumPy array)\n"
"xmax   -- maximum stepsize for the problem\n"
"\n"
"The fmin value returned by the FORTRAN function tiud06 is incorrect.\n"
"Therefore it is not included in the dictionary.\n"; 

static PyObject *tiud28_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n; 
	integer n, problem, err;
	doublereal fmin=0., *xini, xmax;
	PyObject *dict, *tmpo; 
	PyArrayObject *Mx0;
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "test28: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly two arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "ii", &i_problem, &i_n)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	
	if (problem<1 || problem>92) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=i_n;
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	tiud28_(&n, xini, &fmin, &xmax, &problem, &err);
	
	/* Handle error */
	if (err) {
		Py_XDECREF(Mx0);
		PyErr_SetString(PyExc_Exception, "Call to tiud06 failed.");
		return NULL;
	}
	
	/* Prepare return value */
	dict=PyDict_New();
	
	/* fmin is not valid
	tmpo=PyFloat_FromDouble((double)fmin);
	PyDict_SetItemString(dict, "fmin", tmpo);
	Py_XDECREF(tmpo);
	*/
	
	tmpo=PyFloat_FromDouble((double)xmax);
	PyDict_SetItemString(dict, "xmax", tmpo);
	Py_XDECREF(tmpo);
	
	PyDict_SetItemString(dict, "x0", (PyObject *)Mx0); 
	Py_XDECREF(Mx0);
	
	/* Check reference count 
	printf("%d\n", Mx0->ob_refcnt); 
	*/
	
	return dict; 
}

static char tffu28_doc[]=
"Unconstrained problems - function values.\n"
"Wraps FORTRAN function tffu28.\n"
"\n"
"fi=tffu28(num, x)\n"
"\n"
"Input\n"
"num -- problem number (0-91)\n"
"x   -- function argument vector of length n\n"
"\n"
"Output\n"
"f   -- 1-dimensional array of length 1\n"; 

static PyObject *tffu28_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *x; 
	PyArrayObject *Mx, *Mf; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tffu28: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2) {
		PyErr_SetString(PyExc_Exception, "Function takes two arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iO", &i_problem, &Mx)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	problem=i_problem+1;
	
	if (problem<1 || problem>92) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length n");
		return NULL; 
	}
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Extract n */
	n=PyArray_DIM(Mx, 0);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	tffu28_(&n, x, f, &problem); 
	
	return (PyObject *)Mf; 
}



/* Methods table */
static PyMethodDef _lvu_methods[] = {
	{"tiud28", tiud28_wrap, METH_VARARGS, tiud28_doc},
	{"tffu28", tffu28_wrap, METH_VARARGS, tffu28_doc},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_lvu()  {
	(void) Py_InitModule("_lvu", _lvu_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_lvu_amd64()  {
	(void) Py_InitModule("_lvu_amd64", _lvu_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_lvu_i386()  {
	(void) Py_InitModule("_lvu_i386", _lvu_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
