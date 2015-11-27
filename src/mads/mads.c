/* (c)2013 Arpad Buermen */
/* Interface to the MADS test problems written in c/c++. */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>

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

/* Decrease reference count for newly created tuple members */
PyObject *decRefTuple(PyObject *tuple) {
	Py_ssize_t pos; 
	for(pos=0;pos<PyTuple_Size(tuple);pos++) {
		Py_XDECREF(PyTuple_GetItem(tuple, pos));
	}
	return tuple;
}


/* mdo.cpp */

static char mdo_doc[]=
"MDO problem\n"
"Wraps mdo_itf().\n"
"\n"
"(f,c)=mdo_wrap(x, eps, max_it)\n"
"\n"
"Input\n"
"x      -- function argument vector of length 10\n"
"eps    -- precision\n"
"max_it -- iteration limit\n"
"\n"
"Output\n"
"f -- function value"
"c -- constraint function values (vector of length 10)\n"; 

static PyObject *mdo_wrap(PyObject *self, PyObject *args) {
	int dummy, maxit; 
	npy_double eps, *x, *f, *c;
	PyArrayObject *Mx, *Mf, *Mc; 
	npy_intp dims[1];
	
	if (PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes 3 arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "Odi", &Mx, &eps, &maxit)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1 && PyArray_DIM(Mx, 0)==10)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length 10");
		return NULL; 
	}
	
	if (eps<=0) {
		PyErr_SetString(PyExc_Exception, "eps must be greater than zero");
		return NULL; 
	}
	
	if (maxit<=0) {
		PyErr_SetString(PyExc_Exception, "maxit must be greater than zero");
		return NULL; 
	}
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
		
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Allocate vector for c */
	dims[0]=10;
	Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	c=(npy_double *)PyArray_DATA(Mc);
	
	/* Call */
	mdo_itf(eps, maxit, x, f, c, &dummy);
	
	return decRefTuple(PyTuple_Pack(2, Mf, Mc)); 
}


/* sty.a */

static char sty_doc[]=
"MDO problem\n"
"Wraps mdo_itf().\n"
"\n"
"(f,c)=mdo_wrap(x)\n"
"\n"
"Input\n"
"x      -- function argument vector of length 8\n"
"\n"
"Output\n"
"f -- function value"
"c -- constraint function values (vector of length 11)\n"; 

static PyObject *sty_wrap(PyObject *self, PyObject *args) {
	int dummy; 
	npy_double *x, *f, *c;
	PyArrayObject *Mx, *Mf, *Mc; 
	npy_intp dims[1];
	
	if (PyObject_Length(args)!=1) {
		PyErr_SetString(PyExc_Exception, "Function takes one argument.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "O", &Mx)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1 && PyArray_DIM(Mx, 0)==8)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length 8");
		return NULL; 
	}
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Allocate vector for c */
	dims[0]=11;
	Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	c=(npy_double *)PyArray_DATA(Mc);
	
	/* Call */
	sty_itf(x, f, c);
	
	return decRefTuple(PyTuple_Pack(2, Mf, Mc)); 
}


/* Methods table */
static PyMethodDef _mads_methods[] = {
	{"mdo_wrap", mdo_wrap, METH_VARARGS, mdo_doc},
	{"sty_wrap", sty_wrap, METH_VARARGS, sty_doc},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_mads()  {
	(void) Py_InitModule("_mads", _mads_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_mads_amd64()  {
	(void) Py_InitModule("_mads_amd64", _mads_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_mads_i386()  {
	(void) Py_InitModule("_mads_i386", _mads_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
