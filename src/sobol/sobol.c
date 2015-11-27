/* (c)2013 Arpad Buermen */
/* Sobol sequence generator based on the direction numbers and the algorithm given by Joe and Kuo. */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include "data.h"
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


/* Precompute direction numbers */
static char precompute_doc[]=
"Sobol sequence - direction numbers.\n"
"\n"
"V=precompute(dim, L)\n"
"\n"
"Input\n"
"dim -- dimension (max 21201)\n"
"L   -- sequence length (log2, ceil)\n"
"\n"
"Output\n"
"V   -- array with dim*L members.\n";

static PyObject *precompute(PyObject *self, PyObject *args) {
	int i_dim, i_L; 
	unsigned int s, a, i0, *m; 
	// PyObject *MV; 
	PyArrayObject *MV; 
	npy_uint32 i, j, k, *Vtab, dim, L, *V; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "precompute: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly two arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "ii", &i_dim, &i_L)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	if (i_dim<1 || i_dim>21201) {
		PyErr_SetString(PyExc_Exception, "Bad dimension (must be between 1 and 21201).");
		return NULL; 
	}
	
	if (i_L<1 || i_L>32) {
		PyErr_SetString(PyExc_Exception, "Bad log2 length (must be between 1 and 32).");
		return NULL; 
	}
	
	dim=(npy_uint32)i_dim;
	L=(npy_uint32)i_L;
	
	/* Construct NumPy array */
	/* Allocate vector for V */
	dims[0]=i_dim*i_L;
	// MV=PyArray_SimpleNew(1, dims, PyArray_UINT32);
	MV=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
	V=(npy_uint32 *)PyArray_DATA(MV);
	
	/* First index .. dimension (0..dim-1), second index .. C (0..L-1) */
	Vtab=V;
	for(i=0;i<L;i++)
		Vtab[i]=1<<(32-i-1);
	
	for(j=1;j<dim;j++) {
		Vtab=Vtab+L; 
		
		s=stab[j];
		a=atab[j];
		i0=i0tab[j];
		m=mtab+i0;
		
		/* Compute direction numbers */
		if (L<=s) {
			for(i=0;i<L;i++) {
				Vtab[i] = m[i] << (32-i-1);
			}
		} else {
			for(i=0;i<s;i++) {
				Vtab[i] = m[i] << (32-i-1);
			}
			for(i=s;i<L;i++) {
				Vtab[i] = Vtab[i-s] ^ (Vtab[i-s] >> s);
				for(k=0;k<s-1;k++) {
					Vtab[i] ^= (((a >> (s-1-k-1)) & 1) * Vtab[i-k-1]);
				}
			}
		}
	}
	
	return (PyObject *)MV;
}

/* Compute new point, update index and state (X) */
static char generate_doc[]=
"Sobol sequence - generate one or more members.\n"
"\n"
"out=precompute(dim, L, V, index, X, n, retval)\n"
"\n"
"Input\n"
"dim   -- dimension (max 21201)\n"
"L     -- sequence length (log2, ceil)\n"
"V     -- direction numbers\n"
"index -- position in sequence\n"
"X     -- current value\n"
"n     -- number of iterates to return\n"
"ret   -- 0 if no values should be returned\n"
"\n"
"Output\n"
"out   -- array with n*dim members or None\n";

static PyObject *generate(PyObject *self, PyObject *args) {
	int i_dim, i_L, i_n, i_ret; 
	// PyObject *MV, *Mindex, *MX, *Mout;
	PyArrayObject *MV, *Mindex, *MX, *Mout; 
	npy_uint32 i, j, C, *V, *X, *index; 
	npy_double *out; 
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "generate: checking arguments\n");
#endif
	if (PyObject_Length(args)!=7) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly 7 arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iiOOOii", &i_dim, &i_L, &MV, &Mindex, &MX, &i_n, &i_ret)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	if (i_dim<1 || i_dim>21201) {
		PyErr_SetString(PyExc_Exception, "Bad dimension (must be between 1 and 21201).");
		return NULL; 
	}
	
	if (i_L<1 || i_L>32) {
		PyErr_SetString(PyExc_Exception, "Bad log2 length (must be between 1 and 32).");
		return NULL; 
	}
	
	if (i_n<1) {
		PyErr_SetString(PyExc_Exception, "Bad number of points.");
		return NULL; 
	}
	
	if (!(PyArray_Check(MV) && PyArray_TYPE(MV)==NPY_UINT32 && 
	      PyArray_NDIM(MV)==1 && PyArray_DIMS(MV)[0]==(i_L*i_dim))) {
		PyErr_SetString(PyExc_Exception, "Bad direction numbers.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mindex) && PyArray_TYPE(Mindex)==NPY_UINT32 && 
	      PyArray_NDIM(Mindex)==1 && PyArray_DIMS(Mindex)[0]==1)) {
		PyErr_SetString(PyExc_Exception, "Bad index numbers.");
		return NULL; 
	}
	
	if (!(PyArray_Check(MX) && PyArray_TYPE(MX)==NPY_UINT32 && 
	      PyArray_NDIM(MX)==1 && PyArray_DIMS(MX)[0]==i_dim)) {
		PyErr_SetString(PyExc_Exception, "Bad state.");
		return NULL; 
	}
	
	if (i_ret) {
		/* Allocate vector for V */
		dims[0]=i_n;
		dims[1]=i_dim;
		// Mout=PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
		Mout=(PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		out=(npy_double *)PyArray_DATA(Mout);
	}
	
	V=(npy_uint32 *)PyArray_DATA(MV);
	index=(npy_uint32 *)PyArray_DATA(Mindex);
	X=(npy_uint32 *)PyArray_DATA(MX);
	
	
	for(j=0;j<i_n;j++) {
		/* Return value (double) */
		if (i_ret) {
			for(i=0;i<i_dim;i++) {
				*(out+i)=X[i]/4294967296.0; /* 2^32 */
			}
			out+=i_dim;
		}
		
		/* Count ones in old index */
		i=*index;
		C=0;
		while (i & 1) {
			C++;
			i>>=1;
		}
		
		/* Update index */
		(*index)++;
		
		/* New X */
		for(i=0;i<i_dim;i++) {
			X[i]=X[i]^V[i*i_L+C];
		}
	}
	
	if (i_ret) 
		return (PyObject *)Mout;
	else
		Py_RETURN_NONE;
}

/* Methods table */
static PyMethodDef _sobol_methods[] = {
	{"precompute", precompute, METH_VARARGS, precompute_doc},
	{"generate", generate, METH_VARARGS, generate_doc},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _sobol in compile and link */
__declspec(dllexport) void init_sobol()  {
	(void) Py_InitModule("_sobol", _sobol_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_sobol_amd64()  {
	(void) Py_InitModule("_sobol_amd64", _sobol_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_sobol_i386()  {
	(void) Py_InitModule("_sobol_i386", _sobol_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
