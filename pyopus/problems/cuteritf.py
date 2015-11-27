"""
**C source code of the CUTEr interface**

This module is independent of PyOPUS. It can be taken as is and used as 
a module in some other package. 
"""
__all__ = [ 'itf_c_source' ] 

# Because we dont want backslashes to be interpreted as escape characters 
# the string must be a raw string. 
itf_c_source=r"""
/* CUTEr2 interface to Python and NumPy */
/* (c)2011 Arpad Buermen */
/* Licensed under GPL V3 */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */
 
/* Unused CUTEr tools - sparse finite element matrices and banded matrices
     cdimse
	 ceh
	 csgreh
	 ubandh
	 udimse
	 ueh
     ugreh
	 
   CUTEr tools that are not used because they duplicate functionality or are obsolete
     cnames		... used pbname, varnames, connames
	 udimen		... used cdimen 
	 ufn		... used uofg
	 ugr		... used uofg
	 unames		... used pbname, varnames
     cscfg		... obsolete
	 cscifg		... obsolete
*/

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
 
#include "Python.h"
#include "cuter.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>

/* Debug switch - uncomment to enable debug messages */
/* #undefine PYDEBUG */

/* Debug file */
#define df stdout

#ifndef WIN32
#define __declspec(a) 
#endif

/* Safeguard against C++ symbol mangling */
#ifdef __cplusplus
extern "C" {
#endif

/* NumPy type for CUTEr integer 
   INTEGER type in FORTRAN is by default 4 bytes wide (tested with gfortran 
   on AMD64 Linux). The C prototypes of FORTRAN-based CUTEr toos use 
   integer C type which is typedefed to long int in cuter.h. 
   On IA32 Linux long int is 4 bytes wide. Unfortunately on AMD64 it is 8 
   bytes wide. This causes problems if we are passing arguments to
   FORTRAN functions. A serious problem appears when an input or output 
   argument is an array because the elements of the array are twice the size
   the FORTRAN expects them to be. 
   Until this bug is fixed in CUTEr the solution is to use npy_integer for
   variables and npy_integer_type_num for creating NumPy arrays. To get rid 
   of warnings integer arguments are explicitly converted to (integer *) when 
   they are passed to FORTRAN functions. This is harmless since FORTRAN 
   arguments are passed by reference (as pointers) and the type conversion does 
   not affect the pointer's value. 
   The same goes for logical typedef of CUTEr. This interface uses npy_logical 
   which is actually int, but type casts npy_logical* pointers to logical*.
*/
static int npy_integer_type_num=NPY_INT;
typedef int npy_integer; 
typedef int npy_logical; 


/* Prototypes */
static PyObject *cuter__dims(PyObject *self, PyObject *args);
static PyObject *cuter__setup(PyObject *self, PyObject *args);
static PyObject *cuter__varnames(PyObject *self, PyObject *args);
static PyObject *cuter__connames(PyObject *self, PyObject *args);
static PyObject *cuter_objcons(PyObject *self, PyObject *args);
static PyObject *cuter_obj(PyObject *self, PyObject *args);
static PyObject *cuter_cons(PyObject *self, PyObject *args);
static PyObject *cuter_lagjac(PyObject *self, PyObject *args);
static PyObject *cuter_jprod(PyObject *self, PyObject *args);
static PyObject *cuter_hess(PyObject *self, PyObject *args);
static PyObject *cuter_ihess(PyObject *self, PyObject *args);
static PyObject *cuter_hprod(PyObject *self, PyObject *args);
static PyObject *cuter_gradhess(PyObject *self, PyObject *args);
static PyObject *cuter__scons(PyObject *self, PyObject *args); 
static PyObject *cuter__slagjac(PyObject *self, PyObject *args); 
static PyObject *cuter__sphess(PyObject *self, PyObject *args); 
static PyObject *cuter__isphess(PyObject *self, PyObject *args); 
static PyObject *cuter__gradsphess(PyObject *self, PyObject *args); 
static PyObject *cuter_report(PyObject *self, PyObject *args); 

/* Persistent data */
#define STR_LEN 10
static npy_integer CUTEr_nvar = 0;		/* number of variables */
static npy_integer CUTEr_ncon = 0;		/* number of constraints */
static npy_integer CUTEr_nnzj = 0;		/* nnz in Jacobian */
static npy_integer CUTEr_nnzh = 0;		/* nnz in upper triangular Hessian */
static char CUTEr_probName[STR_LEN+1];	/* problem name */
static char setupCalled = 0;			/* Flag to indicate if setup was called */
static char dataFileOpen = 0;			/* Flag to indicate if OUTSDIf is open */

static npy_integer funit = 42;			/* FORTRAN unit number for OUTSDIF.d */
static npy_integer  iout = 6;			/* FORTRAN unit number for error output */
static char  fName[] = "OUTSDIF.d"; 	/* Data file name */
/* Logical constants for FORTRAN calls */
static logical somethingFalse = FALSE_, somethingTrue = TRUE_;


/* Helper functions */

/* Open data file, return 0 on error. */
int open_datafile(void) {
	npy_integer  ioErr;					/* Exit flag from OPEN and CLOSE */
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: opening data file\n");
#endif
	ioErr = 0;
	if (! dataFileOpen) 
		FORTRAN_OPEN((integer *)&funit, fName, (integer *)&ioErr);
	if (ioErr) {
		PyErr_SetString(PyExc_Exception, "Failed to open data file");
		return 0;
	}
	dataFileOpen = 1;
	return 1;
}

/* Close data file, return 0 on error. */
int close_datafile(void) {
	npy_integer ioErr;					/* Exit flag from OPEN and CLOSE */
	ioErr = 0;
	FORTRAN_CLOSE((integer *)&funit, (integer *)&ioErr);
	if (ioErr) {
		PyErr_SetString(PyExc_Exception, "Error closing data file");
		return 0;
	}
	dataFileOpen = 0;
	return 1; 
}

/* Check if the problem is set up, return 0 if it is not. */
int check_setup(void) {
	if (!setupCalled) {
		PyErr_SetString(PyExc_Exception, "Problem is not set up");
		return 0;
	}
	return 1;
}

/* Trim trailing spaces from a string starting at index n. */
void trim_string(char *s, int n) {
	int i;
	
	for(i=n;i>=0;i--) {
		if (s[i]!=' ')
			break;
	}
	s[i+1]=0;
}

/* Decrese reference counf for newly created dictionary members */
PyObject *decRefDict(PyObject *dict) {
	PyObject *key, *value;
	Py_ssize_t pos; 
	pos=0;
	while (PyDict_Next(dict, &pos, &key, &value)) {
		Py_XDECREF(value);
	}
	return dict;
}

/* Decrease reference count for newly created tuple members */
PyObject *decRefTuple(PyObject *tuple) {
	Py_ssize_t pos; 
	for(pos=0;pos<PyTuple_Size(tuple);pos++) {
		Py_XDECREF(PyTuple_GetItem(tuple, pos));
	}
	return tuple;
}

/* Extract sparse gradient and Jacobian in form of NumPy arrays */
void extract_sparse_gradient_jacobian(npy_integer nnzjplusno, npy_integer *sji, npy_integer *sjfi, npy_double *sjv, 
		PyArrayObject **Mgi, PyArrayObject **Mgv, PyArrayObject **MJi, PyArrayObject **MJfi, PyArrayObject **MJv) {
	npy_double *gv, *Jv;
	npy_integer *gi, *Ji, *Jfi, nnzg, i, jg, jj; 
	npy_intp dims[1]; 
	
	/* Get number of nonzeros in gradient vector */
	nnzg=0;
	for(i=0;i<nnzjplusno;i++) {
		if (sjfi[i]==0)
			nnzg++;
	}
	
	/* Alocate and fill objective/Lagrangian gradient data and Jacobian data, 
	   convert indices from FORTRAN to C. */
	dims[0]=nnzg; 
	*Mgi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
	*Mgv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
	gi=(npy_integer *)PyArray_DATA(*Mgi);
	gv=(npy_double *)PyArray_DATA(*Mgv);
	dims[0]=nnzjplusno-nnzg; 
	*MJi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
	*MJfi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
	*MJv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
	Ji=(npy_integer *)PyArray_DATA(*MJi);
	Jfi=(npy_integer *)PyArray_DATA(*MJfi);
	Jv=(npy_double *)PyArray_DATA(*MJv);
	jg=0;
	jj=0;
	for(i=0;i<nnzjplusno;i++) {
		if (sjfi[i]==0) {
			gi[jg]=sji[i]-1;
			gv[jg]=sjv[i];
			jg++;
		} else {
			Ji[jj]=sji[i]-1;
			Jfi[jj]=sjfi[i]-1;
			Jv[jj]=sjv[i];
			jj++;
		}
	}
}

/* Extract sparse Hessian in form of NumPy arrays 
   from sparse ijv format of Hessian's upper triangle + diagonal. 
   Add elements to lower triangle. */
void extract_sparse_hessian(npy_integer nnzho, npy_integer *si, npy_integer *sj, npy_double *sv, 
					PyArrayObject **MHi, PyArrayObject **MHj, PyArrayObject **MHv) {
	npy_integer *Hi, *Hj, nnzdiag, i, j;
	npy_double *Hv; 
	npy_intp dims[1];
	
	
	/* Get number of nonzeros on the diagonal */
	nnzdiag=0;
	for(i=0;i<nnzho;i++) {
		if (si[i]==sj[i])
			nnzdiag++;
	}
	
	/* Alocate and fill objective/Lagrangian gradient data and Jacobian data, 
	   convert indices from FORTRAN to C, fill lower triangle. */
	dims[0]=2*nnzho-nnzdiag; /* Do not duplicate diagonal elements */
	*MHi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
	*MHj=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
	*MHv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
	Hi=(npy_integer *)PyArray_DATA(*MHi);
	Hj=(npy_integer *)PyArray_DATA(*MHj);
	Hv=(npy_double *)PyArray_DATA(*MHv);
	j=0;
	for(i=0;i<nnzho;i++) {
		Hi[j]=si[i]-1;
		Hj[j]=sj[i]-1;
		Hv[j]=sv[i];
		j++;
		if (si[i]!=sj[i]) {
			Hi[j]=sj[i]-1;
			Hj[j]=si[i]-1;
			Hv[j]=sv[i];
			j++;
		}
	}
}


/* Functions */

static char cuter__dims_doc[]=
"Returns the dimension of the problem and the number of constraints.\n"
"\n"
"(n, m)=_dims()\n"
"\n"
"Output\n"
"n -- number of variables\n"
"m -- number of constraints\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"__init__.py script when the test function interface is loaded.\n"
"If you decide to call it anyway, the working directory at the time of call\n"
"must be the one where the file OUTSIF.d can be found.\n"
"\n"
"CUTEr tools used: CDIMEN\n";

static PyObject *cuter__dims(PyObject *self, PyObject *args) {
	if (PyObject_Length(args)!=0)
		PyErr_SetString(PyExc_Exception, "_dims() takes no arguments");

	if (!open_datafile())
		return NULL;
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Calling CDIMEN\n");
#endif
	CDIMEN((integer *)&funit, (integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon);
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr:   n = %-d, m = %-d\n", CUTEr_nvar, CUTEr_ncon);
#endif

	return decRefTuple(PyTuple_Pack(2, PyInt_FromLong((long)CUTEr_nvar), PyInt_FromLong((long)CUTEr_ncon))); 
}


static char cuter__setup_doc[]=
"Sets up the problem.\n"
"\n"
"data=_setup(efirst, lfirst, nvfirst)\n"
"\n"
"Input\n"
"efirst  -- if True, equation constraints are ordered before inequations.\n"
"           Defaults to False.\n"
"lfirst  -- if True, linear constraints are ordered before nonlinear ones.\n"
"           Defaults to False.\n"
"nvfirst -- if True, nonlinear variables are ordered before linear ones.\n"
"           Defaults to False.\n"
"\n"
"Setting both efirst and lfirst to True results in the following ordering:\n"
"linear equations, followed by linear inequations, nonlinear equations,\n"
"and finally nonlinear inequations.\n"
"\n"
"Output\n"
"data -- dictionary with the summary of test function's properties\n"
"\n"
"The problem data dictionary has the following members:\n"
"name    -- problem name\n"
"n       -- number of variables\n"
"m       -- number of constraints (excluding bounds)\n"
"x       -- initial point (1D array of length n)\n"
"bl      -- vector of lower bounds on variables (1D array of length n)\n"
"bu      -- vector of upper bounds on variables (1D array of length n)\n"
"nnzh    -- number of nonzero elements in the diagonal and upper triangle of\n"
"           sparse Hessian\n"
"vartype -- 1D integer array of length n storing variable type\n"
"           0=real,  1=boolean (0 or 1), 2=integer\n"
"\n"
"For constrained problems the following additional members are available\n"
"nnzj    -- number of nonzero elements in sparse Jacobian of constraints\n"
"v       -- initial value of Lagrange multipliers (1D array of length m)\n"
"cl      -- lower bounds on constraint functions (1D array of length m)\n"
"cu      -- upper bounds on constraint functions (1D array of length m)\n"
"equatn  -- 1D boolean array of length m indicating whether a constraint\n"
"           is an equation constraint\n"
"linear  -- 1D boolean array of length m indicating whether a constraint\n"
"           is a linear constraint\n"
"\n"
"-1e+20 and 1e+20 in bl, bu, cl, and cu stand for -infinity and +infinity.\n"
"\n"
"This function must be called before any other CUTEr function is called.\n"
"The only exception is the _dims() function.\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"__init__.py script when the test function interface is loaded.\n"
"If you decide to call it anyway, the working directory at the time of call\n"
"must be the one where the file OUTSIF.d can be found.\n"
"\n"
"CUTEr tools used: CDIMEN, CSETUP, USETUP, CVARTY, UVARTY, \n"
"                  CDIMSH, UDIMSH, CDIMSJ, PBNAME\n";

static PyObject *cuter__setup(PyObject *self, PyObject *args) {
	npy_logical  efirst = FALSE_, lfirst = FALSE_, nvfrst = FALSE_;
	int eFirst, lFirst, nvFirst;
	PyObject *dict;
	PyArrayObject *Mx, *Mbl, *Mbu, *Mv=NULL, *Mcl=NULL, *Mcu=NULL, *Meq=NULL, *Mlinear=NULL;
	PyArrayObject *Mvt; 
	doublereal *x, *bl, *bu, *v=NULL, *cl=NULL, *cu=NULL;
	npy_integer *vartypes; 
	npy_logical *equatn=NULL, *linear=NULL;
	npy_intp dims[1];
	int i;
	
	if (PyObject_Length(args)!=0 && PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "_setup() takes 0 or 3 arguments");
		return NULL; 
	}
	
	if (PyObject_Length(args)==3) {
		if (!PyArg_ParseTuple(args, "iii", &eFirst, &lFirst, &nvFirst)) {
			return NULL; 
		}

		efirst = eFirst  ? TRUE_ : FALSE_;
		lfirst = lFirst  ? TRUE_ : FALSE_;
		nvfrst = nvFirst ? TRUE_ : FALSE_;
	}
	
	if (!open_datafile())
		return NULL; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Calling CDIMEN\n");
#endif
	CDIMEN((integer *)&funit, (integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon);
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr:   n = %-d, m = %-d\n", CUTEr_nvar, CUTEr_ncon);
#endif

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Allocating space\n");
#endif
	/* Numpy arrays */
	dims[0]=CUTEr_nvar;
	Mx=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	Mbl=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	Mbu=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	Mvt=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT); 
	if (CUTEr_ncon>0) {
		dims[0]=CUTEr_ncon;
		Mv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		Mcl=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		Mcu=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		Meq=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);
		Mlinear=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);
	}
	
	/* Get internal data buffers */
	/* Assume that npy_double is equivalent to double and npy_int is equivalent to integer */
	x = (npy_double *)PyArray_DATA(Mx); 
	bl = (npy_double *)PyArray_DATA(Mbl); 
	bu = (npy_double *)PyArray_DATA(Mbu); 
	if (CUTEr_ncon>0) {
		v = (npy_double *)PyArray_DATA(Mv); 
		cl = (npy_double *)PyArray_DATA(Mcl); 
		cu = (npy_double *)PyArray_DATA(Mcu); 
	
		/* Create temporary CUTEr logical arrays */
		equatn = (npy_logical *)malloc(CUTEr_ncon*sizeof(npy_logical));
		linear = (npy_logical *)malloc(CUTEr_ncon*sizeof(npy_logical));
	}
	vartypes=(npy_integer *)malloc(CUTEr_nvar*sizeof(npy_integer)); 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Calling [UC]SETUP\n");
#endif
	if (CUTEr_ncon > 0)
		CSETUP((integer *)&funit, (integer *)&iout, (integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, bl, bu,
				(integer *)&CUTEr_nvar, (logical *)equatn, (logical *)linear, v, cl, cu, (integer *)&CUTEr_ncon,
				(logical *)&efirst, (logical *)&lfirst, (logical *)&nvfrst);
	else
		USETUP((integer *)&funit, (integer *)&iout, (integer *)&CUTEr_nvar, x, bl, bu, (integer *)&CUTEr_nvar);
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr:   n = %-d, m = %-d\n", CUTEr_nvar, CUTEr_ncon);
#endif

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Calling [UC]VARTY\n");
#endif
	if (CUTEr_ncon > 0)
		CVARTY((integer *)&CUTEr_nvar, (integer *)vartypes);
	else
		UVARTY((integer *)&CUTEr_nvar, (integer *)vartypes);
		
	/* Copy logical values to NumPy bool arrays and free temporary storage */
	if (CUTEr_ncon > 0) {
		for(i=0; i<CUTEr_ncon; i++) {
			*((npy_bool*)(PyArray_GETPTR1(Meq, i)))=(equatn[i]==TRUE_ ? NPY_TRUE : NPY_FALSE);
			*((npy_bool*)(PyArray_GETPTR1(Mlinear, i)))=(linear[i]==TRUE_ ? NPY_TRUE : NPY_FALSE);
		}
		free(equatn);
		free(linear);
	} 
	
	/* Copy variable types to NumPy integer arrays and free temporary storage */
	for(i=0; i<CUTEr_nvar; i++) {
		*((npy_int*)PyArray_GETPTR1(Mvt, i))=vartypes[i];
	}
	free(vartypes);
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Calling [CU]DIMSH\n");
#endif
	if (CUTEr_ncon>0)
		CDIMSH((integer *)&CUTEr_nnzh);
	else 
		UDIMSH((integer *)&CUTEr_nnzh);
		
	if (CUTEr_ncon > 0) {
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: Calling CDIMSJ\n");
#endif
		CDIMSJ((integer *)&CUTEr_nnzj);
		CUTEr_nnzj -= CUTEr_nvar;
	} 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr:   nnzh = %-d, nnzj = %-d\n", CUTEr_nnzh, CUTEr_nnzj);
	fprintf(df, "PyCUTEr: Finding out problem name\n");
#endif
	PBNAME((integer *)&CUTEr_nvar, CUTEr_probName);
	trim_string(CUTEr_probName, STR_LEN-1);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr:   %-s\n", CUTEr_probName);
	fprintf(df, "PyCUTEr: Closing data file\n");
#endif
	close_datafile();

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: Building structure\n");
#endif
	dict=PyDict_New();
	PyDict_SetItemString(dict, "n", PyInt_FromLong((long)CUTEr_nvar)); 
	PyDict_SetItemString(dict, "m", PyInt_FromLong((long)CUTEr_ncon)); 
	PyDict_SetItemString(dict, "nnzh", PyInt_FromLong((long)CUTEr_nnzh)); 
	PyDict_SetItemString(dict, "x", (PyObject *)Mx); 
	PyDict_SetItemString(dict, "bl", (PyObject *)Mbl); 
	PyDict_SetItemString(dict, "bu", (PyObject *)Mbu); 
	PyDict_SetItemString(dict, "name", PyString_FromString(CUTEr_probName)); 
	PyDict_SetItemString(dict, "vartype", (PyObject *)Mvt); 
	if (CUTEr_ncon > 0) {
		PyDict_SetItemString(dict, "nnzj", PyInt_FromLong((long)CUTEr_nnzj)); 
		PyDict_SetItemString(dict, "v", (PyObject*)Mv); 
		PyDict_SetItemString(dict, "cl", (PyObject*)Mcl); 
		PyDict_SetItemString(dict, "cu", (PyObject*)Mcu); 
		PyDict_SetItemString(dict, "equatn", (PyObject*)Meq); 
		PyDict_SetItemString(dict, "linear", (PyObject*)Mlinear); 
	}
	
	setupCalled = 1;
	
	return decRefDict(dict); 
}


static char cuter__varnames_doc[]=
"Returns the names of variables in the problem.\n"
"\n"
"namelist=_varnames()\n"
"\n"
"Output\n"
"namelist -- list of length n holding strings holding names of variables\n"
"\n"
"The list reflects the ordering imposed by the nvfirst argument to _setup().\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"__init__.py script when the test function interface is loaded.\n"
"\n"
"CUTEr tools used: VARNAMES\n";

static PyObject *cuter__varnames(PyObject *self, PyObject *args) {
	char *Fvnames, Fvname[STR_LEN+1], *ptr;
	PyObject *list; 
	int i, j; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
		
	if (PyObject_Length(args)!=0) {
		PyErr_SetString(PyExc_Exception, "_varnames() takes 0 arguments");
		return NULL; 
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: allocating space\n");
#endif
	Fvnames=(char *)malloc(CUTEr_nvar*STR_LEN*sizeof(char));
	list=PyList_New(0);
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling VARNAMES\n");
#endif
	VARNAMES((integer *)&CUTEr_nvar, Fvnames);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: building results\n");
#endif
	for(i=0;i<CUTEr_nvar;i++) {
		ptr=Fvnames+i*STR_LEN; 
		for(j=0;j<STR_LEN;j++) {
			Fvname[j]=*ptr;
			ptr++;
		}
		trim_string(Fvname, STR_LEN-1);
		PyList_Append(list, PyString_FromString(Fvname)); 
	}
	
	free(Fvnames); 
	
	return list;
}


static char cuter__connames_doc[]=
"Returns the names of constraints in the problem.\n"
"\n"
"namelist=_connames()\n"
"\n"
"Output\n"
"namelist -- list of length m holding strings holding names of constraints\n"
"\n"
"The list is ordered in the way specified by efirst and lfirst arguments to\n"
"_setup().\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"__init__.py script when the test function interface is loaded.\n"
"\n"
"CUTEr tools used: CONNAMES\n";

static PyObject *cuter__connames(PyObject *self, PyObject *args) {
	char *Fcnames, Fcname[STR_LEN+1], *ptr;
	PyObject *list; 
	int i, j; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
		
	if (PyObject_Length(args)!=0) {
		PyErr_SetString(PyExc_Exception, "_connames() takes 0 arguments");
		return NULL; 
	}
	
	list=PyList_New(0);
	
	if (CUTEr_ncon>0) {

#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: allocating space\n");
#endif
		Fcnames=(char *)malloc(CUTEr_ncon*STR_LEN*sizeof(char));
	
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: calling CONNAMES\n");
#endif
		CONNAMES((integer *)&CUTEr_ncon, Fcnames);

#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: building results\n");
#endif
		for(i=0;i<CUTEr_ncon;i++) {
			ptr=Fcnames+i*STR_LEN; 
			for(j=0;j<STR_LEN;j++) {
				Fcname[j]=*ptr;
				ptr++;
			}
			trim_string(Fcname, STR_LEN-1);
			PyList_Append(list, PyString_FromString(Fcname)); 
		}
	
		free(Fcnames); 
	}
	
	return list;
}


static char cuter_objcons_doc[]=
"Returns the value of objective and constraints at x.\n"
"\n"
"(f, c)=objcons(x)\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"\n"
"Output\n"
"f -- 1D array of length 1 holding the value of the function at x\n"
"c -- 1D array of length m holding the values of constraints at x\n"
"\n"
"CUTEr tools used: CFN\n";

static PyObject *cuter_objcons(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *Mf, *Mc; 
	doublereal *x, *f, *c; 
	npy_intp dims[1]; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	if (!PyArg_ParseTuple(args, "O", &arg1)) 
		return NULL; 
	
	/* Check if x is double and of correct length and shape */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1)&& PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	dims[0]=CUTEr_ncon;
	Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	c=(npy_double *)PyArray_DATA(Mc);
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CFN\n");
#endif
	CFN((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, f, (integer *)&CUTEr_ncon, c);
	
	return decRefTuple(PyTuple_Pack(2, Mf, Mc)); 
}


static char cuter_obj_doc[]=
"Returns the value of objective and its gradient at x.\n"
"\n"
"f=obj(x)\n"
"(f, g)=obj(x, gradFlag)\n"
"\n"
"Input\n"
"x        -- 1D array of length n with the values of variables\n"
"gradFlag -- if given the function returns f and g; can be anything\n"
"\n"
"Output\n"
"f -- 1D array of length 1 holding the value of the function at x\n"
"g -- 1D array of length n holding the value of the gradient of f at x\n"
"\n"
"CUTEr tools used: UOFG, COFG\n";

static PyObject *cuter_obj(PyObject *self, PyObject *args) {
	PyArrayObject *arg1; 
	PyObject *arg2;
	PyArrayObject *Mf, *Mg=NULL; 
	doublereal *x, *f, *g=NULL; 
	npy_intp dims[1];

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2)) 
		return NULL; 
	
	/* Check if x is double and of correct length and shape */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	if (PyObject_Length(args)>1) {
		dims[0]=CUTEr_nvar;
		Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		g=(npy_double *)PyArray_DATA(Mg);
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling [UC]OFG\n");
#endif
	if (CUTEr_ncon == 0) {
		if (PyObject_Length(args)==1) {
			UOFG((integer *)&CUTEr_nvar, x, f, NULL, &somethingFalse);
			return (PyObject *)Mf; 
		} else {
			UOFG((integer *)&CUTEr_nvar, x, f, g, &somethingTrue);
			return decRefTuple(PyTuple_Pack(2, Mf, Mg)); 
		}
	} else {
		if (PyObject_Length(args)==1) {
			COFG((integer *)&CUTEr_nvar, x, f, NULL, &somethingFalse);
			return (PyObject *)Mf; 
		} else {
			COFG((integer *)&CUTEr_nvar, x, f, g, &somethingTrue);
			return decRefTuple(PyTuple_Pack(2, Mf, Mg)); 
		}
	}
}


static char cuter_cons_doc[]=
"Returns the value of constraints and the Jacobian of constraints at x.\n"
"\n"
"c=cons(x)                 -- constraints\n"
"ci=cons(x, False, i)      -- i-th constraint\n"
"(c, J)=cons(x, True)      -- Jacobian of constraints\n"
"(ci, Ji)=cons(x, True, i) -- i-th constraint and its gradient\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"i -- integer index of constraint (between 0 and m-1)\n"
"\n"
"Output\n"
"c  -- 1D array of length m holding the values of constraints at x\n"
"ci -- 1D array of length 1 holding the value of i-th constraint at x\n"
"J  -- 2D array with m rows of n columns holding Jacobian of constraints at x\n"
"Ji -- 1D array of length n holding the gradient of i-th constraintn"
"      (also equal to the i-th row of Jacobian)\n"
"\n"
"CUTEr tools used: CCFG, CCIFG\n";

static PyObject *cuter_cons(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *Mc, *MJ; 
	PyObject *arg2;
	doublereal *x, *c, *J; 
	int derivs, index, wantSingle;
	npy_integer icon; 
	npy_integer zero = 0;
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	if (!PyArg_ParseTuple(args, "O|Oi", &arg1, &arg2, &index)) 
		return NULL; 
	
	/* Check if x is double and of correct length and shape */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Do we want derivatives */
	if (arg2!=NULL && arg2==Py_True)
		derivs=1;
	else
		derivs=0;
	
	/* Do we want a particular derivative */
	if (PyObject_Length(args)==3) {
		/* Check index */
		if (index<0 || index>=CUTEr_ncon) {
			PyErr_SetString(PyExc_Exception, "Argument 3 must be an integer between 0 and ncon-1");
			return NULL; 
		}
		icon=index+1;
		wantSingle=1;
	} else {
		wantSingle=0;
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (!wantSingle) {
		dims[0]=CUTEr_ncon;
		Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		c=(npy_double *)PyArray_DATA(Mc);
		if (derivs) {
			dims[0]=CUTEr_ncon;
			dims[1]=CUTEr_nvar;
			/* Create a FORTRAN style array (first index stride is 1) */
			MJ=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
			J=(npy_double *)PyArray_DATA(MJ);
		}
	} else {
		dims[0]=1;
		Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		c=(npy_double *)PyArray_DATA(Mc);
		if (derivs) {
			dims[0]=CUTEr_nvar;
			MJ=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
			J=(npy_double *)PyArray_DATA(MJ);
		}
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CCFG/CCIFG\n");
#endif
	if (!wantSingle) {
		if (!derivs) {
			CCFG((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (integer *)&CUTEr_ncon, c,
					&somethingFalse, (integer *)&zero, (integer *)&zero, NULL, &somethingFalse);
			return (PyObject *)Mc;
		} else {
			CCFG((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (integer *)&CUTEr_ncon, c,
					&somethingFalse, (integer *)&CUTEr_ncon, (integer *)&CUTEr_nvar, J,
					&somethingTrue);
			return decRefTuple(PyTuple_Pack(2, Mc, MJ)); 
		}
	} else {
		if (!derivs) {
			CCIFG((integer *)&CUTEr_nvar, (integer *)&icon, x, c, NULL, &somethingFalse);
			return (PyObject *)Mc; 
		} else {
			CCIFG((integer *)&CUTEr_nvar, (integer *)&icon, x, c, J, &somethingTrue);
			return decRefTuple(PyTuple_Pack(2, Mc, MJ)); 
		}
	}
}


static char cuter_lagjac_doc[]=
"Returns the gradient of the objective or Lagrangian, and the Jacobian of\n"
"constraints at x. The gradient is the gradient with respect to the problem's\n"
"variables (has n components).\n"
"\n"
"(g, J)=lagjac(x)    -- objective gradient and the Jacobian of constraints\n"
"(g, J)=lagjac(x, v) -- Lagrangian gradient and the Jacobian of constraints\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"v -- 1D array of length m with the Lagrange multipliers\n"
"\n"
"Output\n"
"g  -- 1D array of length n holding the gradient at x\n"
"J  -- 2D array with m rows of n columns holding Jacobian of constraints at x\n"
"\n"
"CUTEr tools used: CGR\n";

static PyObject *cuter_lagjac(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *Mg, *MJ; 
	doublereal *x, *v=NULL, *g, *J; 
	int lagrangian; 
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2)) 
		return NULL; 
	
	/* Check if x is double and of correct length and shape */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Check if v is double and of correct length and shape. */
	if (arg2!=NULL) {
		if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
			return NULL; 
		}
		lagrangian=1;
	} else {
		lagrangian=0;
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (lagrangian) 
		v=(npy_double *)PyArray_DATA(arg2);
	dims[0]=CUTEr_nvar;
	Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	g=(npy_double *)PyArray_DATA(Mg);
	dims[0]=CUTEr_ncon;
	dims[1]=CUTEr_nvar;
	/* Create a FORTRAN style array (first index stride is 1) */
	MJ=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
	J=(npy_double *)PyArray_DATA(MJ);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CGR\n");
#endif
	if (!lagrangian) {
		CGR((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, &somethingFalse, (integer *)&CUTEr_ncon,
			NULL, g, &somethingFalse, (integer *)&CUTEr_ncon, (integer *)&CUTEr_nvar, J);
	} else {
		CGR((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, &somethingTrue, (integer *)&CUTEr_ncon,
			v, g, &somethingFalse, (integer *)&CUTEr_ncon, (integer *)&CUTEr_nvar, J);
	}
	
	return decRefTuple(PyTuple_Pack(2, Mg, MJ)); 
}


static char cuter_jprod_doc[]=
"Returns the product of constraints Jacobian at x with vector p\n"
"\n"
"r=jprod(transpose, p, x) -- computes Jacobian at x before product calculation\n"
"r=jprod(transpose, p)    -- uses last computed Jacobian\n"
"\n"
"Input\n"
"transpose -- boolean flag indicating that the Jacobian should be trasposed\n"
"             before the product is calculated\n"
"p         -- the vector that will be multiplied with the Jacobian\n"
"             1D array of length n (m) if transpose if False (True)\n"
"x         -- 1D array of length n holding the values of variables used in the\n"
"             evaluation of the constraints Jacobian\n"
"\n"
"Output\n"
"r  -- 1D array of length m if transpose=False (or n if transpose=True)\n"
"      with the result\n"
"\n"
"CUTEr tools used: CJPROD\n";

static PyObject *cuter_jprod(PyObject *self, PyObject *args) {
	PyArrayObject *arg2, *arg3, *Mr; 
	PyObject *arg1;
	doublereal *p, *x, *r; 
	int transpose; 
	npy_intp dims[1];

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg3=NULL;
	if (!PyArg_ParseTuple(args, "OO|O", &arg1, &arg2, &arg3)) 
		return NULL; 
	
	/* Check if arg1 is True */
	if (arg1==Py_True) 
		transpose=1;
	else
		transpose=0; 
	
	/* Check if p is double and of correct dimension */
	if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array");
		return NULL; 
	}
	
	/* Check length of p when J is not transposed */
	if (!transpose && !(PyArray_DIM(arg2, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be of length nvar (J is not transposed)");
		return NULL;
	}
	
	/* Check length of p when J is transposed */
	if (transpose && !(PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be of length ncon (J is transposed)");
		return NULL; 
	}
	
	/* Check if x is double and of correct length and shape. */
	if (arg3!=NULL) {
		if (!(arg3!=NULL && PyArray_Check(arg3) && PyArray_ISFLOAT(arg3) && PyArray_TYPE(arg3)==NPY_DOUBLE && PyArray_NDIM(arg3)==1 && PyArray_DIM(arg3, 0)==CUTEr_nvar)) {
			PyErr_SetString(PyExc_Exception, "Argument 3 must be a 1D double array of length nvar");
			return NULL; 
		}
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	p=(npy_double *)PyArray_DATA(arg2);
	if (arg3!=NULL) 
		x=(npy_double *)PyArray_DATA(arg3);
	else
		x=NULL;
	if (!transpose) {
		dims[0]=CUTEr_ncon;
	} else {
		dims[0]=CUTEr_nvar;
	}
	Mr=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	r=(npy_double *)PyArray_DATA(Mr);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CJPROD\n");
#endif
	if (!transpose) {
		if (arg3==NULL) {
			CJPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingTrue,
					&somethingFalse, NULL, p, (integer *)&CUTEr_nvar, r, (integer *)&CUTEr_ncon);
		} else {
			CJPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingFalse,
					&somethingFalse, x, p, (integer *)&CUTEr_nvar, r, (integer *)&CUTEr_ncon);
		}
	} else {
		if (arg3==NULL) {
			CJPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingTrue,
					&somethingTrue, NULL, p, (integer *)&CUTEr_ncon, r, (integer *)&CUTEr_nvar);
		} else {
			CJPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingFalse,
					&somethingTrue, x, p, (integer *)&CUTEr_ncon, r, (integer *)&CUTEr_nvar);
		}
	}
	
	return (PyObject *)Mr;
}


static char cuter_hess_doc[]=
"Returns the Hessian of the objective (for unconstrained problems) or the\n"
"Hessian of the Lagrangian (for constrained problems) at x.\n" 
"\n"
"H=hess(x)    -- Hessian of objective at x for unconstrained problems\n"
"H=hess(x, v) -- Hessian of Lagrangian at (x, v) for constrained problems\n"
"\n"
"The first form can only be used for unconstrained problems. The second one\n"
"can only be used for constrained problems. For obtaining the Hessian of the\n"
"objective in case of a constrained problem use ihess().\n"
"\n"
"The Hessian is meant with respect to problem variables (has dimension n).\n"
"\n"
"Input\n"
"x         -- 1D array of length n holding the values of variables\n"
"v         -- 1D array of length m holding the values of Lagrange multipliers\n"
"\n"
"Output\n"
"H  -- 2D array with n rows of n columns holding the Hessian at x (or (x, v))\n"
"\n"
"CUTEr tools used: CDH, UDH\n";

static PyObject *cuter_hess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *MH; 
	doublereal *x, *v=NULL, *H; 
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2)) 
		return NULL; 
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	if (CUTEr_ncon>0) {
		/* Check if v is double and of correct dimension */
		if (arg2!=NULL) {
			if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
				PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
				return NULL; 
			}
		} else {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be specified for constrained problems. Use ihess().");
			return NULL; 
		}
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (CUTEr_ncon>0) 
		v=(npy_double *)PyArray_DATA(arg2);
	dims[0]=CUTEr_nvar;
	dims[1]=CUTEr_nvar;
	/* Create a FORTRAN style array (first index stride is 1) */
	MH=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
	H=(npy_double *)PyArray_DATA(MH);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling [CU]DH\n");
#endif
	if (CUTEr_ncon>0) {
		CDH((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (integer *)&CUTEr_ncon, v, (integer *)&CUTEr_nvar, H);
	} else {
		UDH((integer *)&CUTEr_nvar, x, (integer *)&CUTEr_nvar, H);
	}
	
	return (PyObject *)MH; 
}


static char cuter_ihess_doc[]=
"Returns the Hessian of the objective or the Hessian of i-th constraint at x.\n"
"\n"
"H=ihess(x)    -- Hessian of the objective\n"
"H=ihess(x, i) -- Hessian of i-th constraint\n"
"\n"
"The first form can only be used for unconstrained problems. The second one\n"
"can only be used for constrained problems. For obtaining the Hessian of the\n"
"objective in case of a constrained problem use ihess().\n"
"\n"
"The Hessian is meant with respect to problem variables (has dimension n).\n"
"\n"
"Input\n"
"x -- 1D array of length n holding the values of variables\n"
"i -- integer holding the index of the constraint (between 0 and m-1)\n"
"\n"
"Output\n"
"H  -- 2D array with n rows of n columns holding the Hessian at x\n"
"\n"
"CUTEr tools used: CIDH, UDH\n";

static PyObject *cuter_ihess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *MH; 
	doublereal *x, *H; 
	npy_intp dims[2];
	int i; 
	npy_integer icon; 

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	if (!PyArg_ParseTuple(args, "O|i", &arg1, &i)) 
		return NULL; 
	
	if (PyObject_Length(args)>1) {
		icon=i+1;
		if (i<0 || i>=CUTEr_ncon) {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be between 0 and ncon-1");
			return NULL; 
		}
	} else {
		icon=0; 
	}
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	dims[0]=CUTEr_nvar;
	dims[1]=CUTEr_nvar;
	/* Create a FORTRAN style array (first index stride is 1) */
	MH=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
	H=(npy_double *)PyArray_DATA(MH);	
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CIDH/UDH\n");
#endif
	if (CUTEr_ncon>0) {
		CIDH((integer *)&CUTEr_nvar, x, (integer *)&icon, (integer *)&CUTEr_nvar, H);
	} else {
		UDH((integer *)&CUTEr_nvar, x, (integer *)&CUTEr_nvar, H);
	}
	
	return (PyObject *)MH;
}


static char cuter_hprod_doc[]=
"Returns the product of Hessian at x and vector p.\n"
"The Hessian is either the Hessian of objective or the Hessian of Lagrangian.\n" 
"\n"
"r=hprod(p, x, v) -- use Hessian of Lagrangian at x (constrained problem)\n"
"r=hprod(p, x)    -- use Hessian of objective at x (unconstrained problem)\n"
"r=hprod(p)       -- use last computed Hessian\n"
"\n"
"The first form can only be used for constrained problems. The second one\n"
"can only be used for unconstrained problems.\n"
"\n"
"The Hessian is meant with respect to problem variables (has dimension n).\n"
"\n"
"Input\n"
"p -- 1D array of length n holding the components of the vector\n"
"x -- 1D array of length n holding the values of variables\n"
"v -- 1D array of length m holding the values of Lagrange multipliers\n"
"\n"
"Output\n"
"r  -- 1D array of length n holding the result\n"
"\n"
"CUTEr tools used: CPROD, UPROD\n";

static PyObject *cuter_hprod(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *arg3, *Mr; 
	doublereal *p, *x=NULL, *v=NULL, *r; 
	npy_intp dims[1];

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=arg3=NULL; 
	if (!PyArg_ParseTuple(args, "O|OO", &arg1, &arg2, &arg3)) 
		return NULL; 
	
	if (CUTEr_ncon>0) {
		if (PyObject_Length(args)==2) {
			PyErr_SetString(PyExc_Exception, "Need 1 or 3 arguments for constrained problems");
			return NULL; 
		}
	} else {
		if (PyObject_Length(args)==3) {
			PyErr_SetString(PyExc_Exception, "Need 1 or 2 arguments for unconstrained problems");
			return NULL; 
		}
	}
	
	/* Check if p is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Check if x is double and of correct dimension */
	if (arg2!=NULL && !(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Check if v is double and of correct dimension */
	if (arg3!=NULL && !(PyArray_Check(arg3) && PyArray_ISFLOAT(arg3) && PyArray_TYPE(arg3)==NPY_DOUBLE && PyArray_NDIM(arg3)==1 && PyArray_DIM(arg3, 0)==CUTEr_ncon)) {
		PyErr_SetString(PyExc_Exception, "Argument 3 must be a 1D double array of length ncon");
		return NULL; 
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	p=(npy_double *)PyArray_DATA(arg1);
	if (arg2!=NULL)
		x=(npy_double *)PyArray_DATA(arg2);
	if (arg3!=NULL)
		v=(npy_double *)PyArray_DATA(arg3);
	dims[0]=CUTEr_nvar;
	Mr=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
	r=(npy_double *)PyArray_DATA(Mr);	

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling [CU]PROD\n");
#endif
	if (CUTEr_ncon>0) {
		if (arg2==NULL)
			CPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingTrue, NULL, (integer *)&CUTEr_ncon, NULL, p, r);
		else 
			CPROD((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingFalse, x, (integer *)&CUTEr_ncon, v, p, r);
	} else {
		if (arg2==NULL)
			UPROD((integer *)&CUTEr_nvar, &somethingTrue, NULL, p, r);
		else 
			UPROD((integer *)&CUTEr_nvar, &somethingFalse, x, p, r);
	}
	
	return (PyObject *)Mr;
}


static char cuter_gradhess_doc[]=
"Returns the Hessian of the Lagrangian, the Jacobian of constraints, and the\n"
"gradient of the objective or the gradient of the Lagrangian at.\n" 
"\n"
"(g, H)=gradhess(x)       -- for unconstrained problems\n"
"(g, J, H)=gradhess(x, v, gradl) -- for constrained problems\n"
"\n"
"The first form can only be used for unconstrained problems. The second one\n"
"can only be used for constrained problems.\n"
"\n"
"The Hessian is meant with respect to problem variables (has dimension n).\n"
"\n"
"Input\n"
"x     -- 1D array of length n holding the values of variables\n"
"v     -- 1D array of length m holding the values of Lagrange multipliers\n"
"gradl -- boolean flag. If False the gradient of the objective is returned, \n"
"         if True the gradient of the Lagrangian is returned.\n"
"         Default is False.\n"
"\n"
"Output\n"
"g  -- 1D array of length n holding\n"
"      the gradient of objective at x (for unconstrained problems) or\n"
"      the gradient of Lagrangian at (x, v) (for constrained problems)\n"
"J  -- 2D array with m rows and n columns holding the Jacobian of constraints\n"
"H  -- 2D array with n rows and n columns holding\n"
"      the Hessian of the objective (for unconstrained problems) or\n"
"      the Hessian of the Lagrangian (for constrained problems)\n"
"\n"
"CUTEr tools used: CGRDH, UGRDH\n";

static PyObject *cuter_gradhess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *Mg, *MH, *MJ; 
	PyObject *arg3; 
	doublereal *x, *v=NULL, *g, *H, *J; 
	npy_logical grlagf; 
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	arg3=NULL; 
	if (!PyArg_ParseTuple(args, "O|OO", &arg1, &arg2, &arg3)) 
		return NULL; 
	
	if (CUTEr_ncon>0) {
		if (PyObject_Length(args)<2) {
			PyErr_SetString(PyExc_Exception, "Need at least 2 arguments for constrained problems");
			return NULL; 
		}
	} else {
		if (PyObject_Length(args)!=1) {
			PyErr_SetString(PyExc_Exception, "Need 1 argument for unconstrained problems");
			return NULL; 
		}
	}
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Check if v is double and of correct dimension */
	if (arg2!=NULL && !(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
		return NULL; 
	}
	
	/* Are we computing the gradient of the Lagrangian */
	if (arg3!=NULL && arg3==Py_True) {
		grlagf=TRUE_;
	} else {
		grlagf=FALSE_; 
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (arg2!=NULL)
		v=(npy_double *)PyArray_DATA(arg2);
	dims[0]=CUTEr_nvar;
	Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
	g=(npy_double *)PyArray_DATA(Mg);	
	dims[0]=CUTEr_nvar;
	dims[1]=CUTEr_nvar;
	/* Create a FORTRAN style array (first index stride is 1) */
	MH=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
	H=(npy_double *)PyArray_DATA(MH);
	dims[0]=CUTEr_ncon;
	dims[1]=CUTEr_nvar;
	/* Create a FORTRAN style array (first index stride is 1) */
	MJ=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
	J=(npy_double *)PyArray_DATA(MJ);

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling [CU]GRDH\n");
#endif
	if (CUTEr_ncon>0) {
		CGRDH((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (logical *)&grlagf, (integer *)&CUTEr_ncon, v,
				g, &somethingFalse, (integer *)&CUTEr_ncon, (integer *)&CUTEr_nvar, J, (integer *)&CUTEr_nvar, H);
		return decRefTuple(PyTuple_Pack(3, Mg, MJ, MH)); 
	} else {
		UGRDH((integer *)&CUTEr_nvar, x, g, (integer *)&CUTEr_nvar, H);
		return decRefTuple(PyTuple_Pack(2, Mg, MH)); 
	}
}


static char cuter__scons_doc[]=
"Returns the value of constraints and the sparse Jacobian of constraints at x.\n"
"\n"
"(c, Jvi, Jfi, Jv)=_scons(x) -- Jacobian of constraints\n"
"(ci, gi, gv)=_scons(x, i)   -- i-th constraint and its gradient\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"i -- integer index of constraint (between 0 and m-1)\n"
"\n"
"Output\n"
"c   -- 1D array of length m holding the values of constraints at x\n"
"Jvi -- 1D array of integers holding the column indices (0 .. n-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jfi -- 1D array of integers holding the row indices (0 .. m-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jv  -- 1D array holding the values of nonzero elements in the sparse Jacobian\n"
"       of constraints at x. Has the same length as Jvi and Jfi.\n"
"ci  -- 1D array of length 1 with the value of i-th constraint at x\n"
"gi  -- 1D array of integers holding the indices (0 .. n-1) of nonzero\n"
"       elements in the sparse gradient vector of i-th constraint\n"
"gv  -- 1D array holding the values of nonzero elements in the sparse gradient\n"
"       vector representing the gradient of i-th constraint at x.\n"
"       Has the same length as gi. gi and gv corespond to the i-th row of\n"
"       constraints Jacobian at x.\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"wrapper function scons().\n"
"\n"
"CUTEr tools used: CCFSG, CCIFSG\n";

static PyObject *cuter__scons(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *Mc, *MJi, *MJfi, *MJv, *Mgi, *Mgv; 
	doublereal *c, *Jv, *gv, *x, *sv; 
	npy_integer *Ji, *Jfi, *gi, *si;  
	npy_integer index, nnzsgc; 
	int i; 
	npy_intp dims[1]; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	if (!PyArg_ParseTuple(args, "O|i", &arg1, &i)) 
		return NULL; 
		
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	if (PyObject_Length(args)==2) {
		if (i<0 || i>=CUTEr_ncon) {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be an integer between 0 and ncon-1");
			return NULL; 
		}
		index=i+1;
	}
	
	if (PyObject_Length(args)==1) {
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
		x=(npy_double *)PyArray_DATA(arg1);
		dims[0]=CUTEr_nnzj;
		MJi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
		MJfi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
		MJv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
		Ji=(npy_integer *)PyArray_DATA(MJi);	
		Jfi=(npy_integer *)PyArray_DATA(MJfi);	
		Jv=(npy_double *)PyArray_DATA(MJv);	
		dims[0]=CUTEr_ncon; 
		Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
		c=(npy_double *)PyArray_DATA(Mc);	
	
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: calling CCFSG\n");
#endif
		CCFSG((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (integer *)&CUTEr_ncon, c, (integer *)&CUTEr_nnzj,
              (integer *)&CUTEr_nnzj, Jv, (integer *)Ji, (integer *)Jfi, &somethingTrue);
		
		/* Convert FORTRAN indices to C indices */
		for(i=0;i<CUTEr_nnzj;i++) {
			Ji[i]--;
			Jfi[i]--;
		}
		
		return decRefTuple(PyTuple_Pack(4, Mc, MJi, MJfi, MJv));
	} else {
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
		x=(npy_double *)PyArray_DATA(arg1);
		si=(npy_integer *)malloc(CUTEr_nvar*sizeof(npy_integer));
		sv=(npy_double *)malloc(CUTEr_nvar*sizeof(npy_double));
		dims[0]=1;
		Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
		c=(npy_double *)PyArray_DATA(Mc);	

#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: calling CCIFSG\n");
#endif
		CCIFSG((integer *)&CUTEr_nvar, (integer *)&index, x, c, (integer *)&nnzsgc, (integer *)&CUTEr_nvar, sv, (integer *)si, &somethingTrue);
		
		/* Allocate and copy results, convert indices from FORTRAN to C, free storage */
		dims[0]=nnzsgc; 
		Mgi=(PyArrayObject *)PyArray_SimpleNew(1, dims, npy_integer_type_num); 
		Mgv=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
		gi=(npy_integer *)PyArray_DATA(Mgi);
		gv=(npy_double *)PyArray_DATA(Mgv);	
		for (i=0;i<nnzsgc;i++) {
			gi[i]=si[i]-1;
			gv[i]=sv[i];
		}
		free(si);
		free(sv);
		
		return decRefTuple(PyTuple_Pack(3, Mc, Mgi, Mgv)); 
	}
}


static char cuter__slagjac_doc[]=
"Returns the sparse gradient of objective at x or Lagrangian at (x, v), \n"
"and the sparse Jacobian of constraints at x.\n"
"\n"
"(gi, gv, Jvi, Jfi, Jv)=_slagjac(x)    -- objective gradient and Jacobian\n"
"(gi, gv, Jvi, Jfi, Jv)=_slagjac(x, v) -- Lagrangian gradient and Jacobian\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"v -- 1D array of length m with the values of Lagrange multipliers\n"
"\n"
"Output\n"
"gi  -- 1D array of integers holding the indices (0 .. n-1) of nonzero\n"
"       elements in the sparse gradient vector\n"
"gv  -- 1D array holding the values of nonzero elements in the sparse gradient\n"
"       vector. Has the same length as gi.\n"
"Jvi -- 1D array of integers holding the column indices (0 .. n-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jfi -- 1D array of integers holding the row indices (0 .. m-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jv  -- 1D array holding the values of nonzero elements in the sparse Jacobian\n"
"       of constraints at x. Has the same length as Jvi and Jfi.\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"wrapper function slagjac().\n"
"\n"
"CUTEr tools used: CSGR\n";

static PyObject *cuter__slagjac(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *Mgi, *Mgv, *MJi, *MJfi, *MJv; 
	doublereal *x, *v=NULL, *sv; 
	npy_integer *si, *sfi;
	npy_integer nnzjplusn, nnzjplusno; 
	int lagrangian; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2)) 
		return NULL; 
	
	/* Check if x is double and of correct length and shape */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	/* Check if v is double and of correct length and shape. */
	if (arg2!=NULL) {
		if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
			return NULL; 
		}
		lagrangian=1;
	} else {
		lagrangian=0;
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (lagrangian) 
		v=(npy_double *)PyArray_DATA(arg2);
	nnzjplusn=CUTEr_nnzj+CUTEr_nvar; 
	si=(npy_integer *)malloc(nnzjplusn*sizeof(npy_integer));
	sfi=(npy_integer *)malloc(nnzjplusn*sizeof(npy_integer));
	sv=(npy_double *)malloc(nnzjplusn*sizeof(npy_double)); 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CSGR\n");
#endif
	/* Must use different variable for output NNZJ and input LCJAC */
	if (!lagrangian) {
		CSGR((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingFalse, (integer *)&CUTEr_ncon,
				NULL, x, (integer *)&nnzjplusno, (integer *)&nnzjplusn, sv, (integer *)si, (integer *)sfi);
	} else {
		CSGR((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, &somethingTrue, (integer *)&CUTEr_ncon,
				v, x, (integer *)&nnzjplusno, (integer *)&nnzjplusn, sv, (integer *)si, (integer *)sfi);
	}
	
	extract_sparse_gradient_jacobian(nnzjplusno, si, sfi, sv, (PyArrayObject **)&Mgi, (PyArrayObject **)&Mgv, (PyArrayObject **)&MJi, (PyArrayObject **)&MJfi, (PyArrayObject **)&MJv); 
	
	/* Free temporary storage */
	free(si);
	free(sfi);
	free(sv);
	
	return decRefTuple(PyTuple_Pack(5, Mgi, Mgv, MJi, MJfi, MJv)); 
}


static char cuter__sphess_doc[]=
"Returns the sparse Hessian of the objective at x (unconstrained problems) or\n"
"the sparse Hessian of the Lagrangian (constrained problems) at (x, v).\n"
"\n"
"(Hi, Hj, Hv)=_sphess(x)    -- Hessian of objective (unconstrained problems)\n"
"(Hi, Hj, Hv)=_sphess(x, v) -- Hessian of Lagrangian (constrained problems)\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"v -- 1D array of length m with the values of Lagrange multipliers\n"
"\n"
"Output\n"
"Hi -- 1D array of integers holding the row indices (0 .. n-1)\n"
"      of nozero elements in sparse Hessian\n"
"Hj -- 1D array of integers holding the column indices (0 .. n-1)\n"
"      of nozero elements in sparse Hessian\n"
"Hv -- 1D array holding the values of nonzero elements in the sparse Hessian\n"
"      Has the same length as Hi and Hj.\n"
"\n"
"Hi, Hj, and Hv represent the full Hessian and not only the diagonal and the\n"
"upper triangle. To obtain the Hessian of the objective of constrained\n"
"problems use _isphess().\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"wrapper function sphess().\n"
"\n"
"CUTEr tools used: CSH, USH\n";

static PyObject *cuter__sphess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *MHi, *MHj, *MHv; 
	doublereal *x, *v=NULL, *sv; 
	npy_integer *si, *sj, nnzho; 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2)) 
		return NULL; 
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	if (CUTEr_ncon>0) {
		/* Check if v is double and of correct dimension */
		if (arg2!=NULL) {
			if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
				PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
				return NULL; 
			}
		} else {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be specified for constrained problems. Use _isphess().");
			return NULL; 
		}
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	if (CUTEr_ncon>0) 
		v=(npy_double *)PyArray_DATA(arg2);
	si=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sj=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sv=(npy_double *)malloc(CUTEr_nnzh*sizeof(npy_double)); 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling [CU]SH\n");
#endif
	if (CUTEr_ncon>0) {
		CSH((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, (integer *)&CUTEr_ncon, v, (integer *)&nnzho, (integer *)&CUTEr_nnzh, 
			sv, (integer *)si, (integer *)sj);
	} else {
		USH((integer *)&CUTEr_nvar, x, (integer *)&nnzho, (integer *)&CUTEr_nnzh, 
			sv, (integer *)si, (integer *)sj);
	}
	
	extract_sparse_hessian(nnzho, si, sj, sv, (PyArrayObject **)&MHi, (PyArrayObject **)&MHj, (PyArrayObject **)&MHv); 
	
	/* Free temporary storage */
	free(si);
	free(sj);
	free(sv);
	
	return decRefTuple(PyTuple_Pack(3, MHi, MHj, MHv)); 
}


static char cuter__isphess_doc[]=
"Returns the sparse Hessian of the objective or the sparse Hessian of i-th\n"
"constraint at x.\n"
"\n"
"(Hi, Hj, Hv)=_isphess(x)    -- Hessian of objective\n"
"(Hi, Hj, Hv)=_isphess(x, i) -- Hessian of i-th constraint\n"
"\n"
"Input\n"
"x -- 1D array of length n with the values of variables\n"
"i -- integer holding the index of constraint (between 0 and m-1)\n"
"\n"
"Output\n"
"Hi -- 1D array of integers holding the row indices (0 .. n-1)\n"
"      of nozero elements in sparse Hessian\n"
"Hj -- 1D array of integers holding the column indices (0 .. n-1)\n"
"      of nozero elements in sparse Hessian\n"
"Hv -- 1D array holding the values of nonzero elements in the sparse Hessian\n"
"      Has the same length as Hi and Hj.\n"
"\n"
"Hi, Hj, and Hv represent the full Hessian and not only the diagonal and the\n"
"upper triangle.\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"wrapper function isphess().\n"
"\n"
"CUTEr tools used: CISH, USH\n";

static PyObject *cuter__isphess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *MHi, *MHj, *MHv; 
	doublereal *x, *sv; 
	npy_integer *si, *sj, nnzho, i; 
	npy_integer icon; 

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	if (!PyArg_ParseTuple(args, "O|i", &arg1, &i)) 
		return NULL; 
	
	if (PyObject_Length(args)>1) {
		icon=i+1;
		if (i<0 || i>=CUTEr_ncon) {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be between 0 and ncon-1");
			return NULL; 
		}
	} else {
		icon=0; 
	}
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	si=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sj=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sv=(npy_double *)malloc(CUTEr_nnzh*sizeof(npy_double)); 
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: calling CISH/USH\n");
#endif
	if (CUTEr_ncon>0) {
		CISH((integer *)&CUTEr_nvar, x, (integer *)&icon, (integer *)&nnzho, (integer *)&CUTEr_nnzh, sv, (integer *)si, (integer *)sj);
	} else {
		USH((integer *)&CUTEr_nvar, x, (integer *)&nnzho, (integer *)&CUTEr_nnzh, sv, (integer *)si, (integer *)sj);
	}
	
	extract_sparse_hessian(nnzho, si, sj, sv, (PyArrayObject **)&MHi, (PyArrayObject **)&MHj, (PyArrayObject **)&MHv); 
	
	/* Free temporary storage */
	free(si);
	free(sj);
	free(sv);
	
	return decRefTuple(PyTuple_Pack(3, MHi, MHj, MHv)); 
}


static char cuter__gradsphess_doc[]=
"Returns the sparse Hessian of the Lagrangian, the sparse Jacobian of\n"
"constraints, and the gradient of the objective or Lagrangian.\n" 
"\n"
"(g, Hi, Hj, Hv)=_gradsphess(x) -- unconstrained problems\n"
"(gi, gv, Jvi, Jfi, Jv, Hi, Hj, Hv)=_gradsphess(x, v, gradl)\n"
"                               -- constrained problems\n"
"\n"
"Input\n"
"x     -- 1D array of length n with the values of variables\n"
"v     -- 1D array of length m holding the values of Lagrange multipliers\n"
"gradl -- boolean flag. If False the gradient of the objective is returned, \n"
"         if True the gradient of the Lagrangian is returned. Default is False.\n"
"\n"
"Output\n"
"g   -- 1D array of length n with the gradient of objective or Lagrangian\n"
"Hi  -- 1D array of integers holding the row indices (0 .. n-1)\n"
"       of nozero elements in sparse Hessian\n"
"Hj  -- 1D array of integers holding the column indices (0 .. n-1)\n"
"       of nozero elements in sparse Hessian\n"
"Hv  -- 1D array holding the values of nonzero elements in the sparse Hessian\n"
"       Has the same length as Hi and Hj.\n"
"gi  -- 1D array of integers holding the indices (0 .. n-1) of nonzero\n"
"       elements in the sparse gradient vector\n"
"gv  -- 1D array holding the values of nonzero elements in the sparse gradient\n"
"       vector. Has the same length as gi.\n"
"Jvi -- 1D array of integers holding the column indices (0 .. n-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jfi -- 1D array of integers holding the row indices (0 .. m-1)\n"
"       of nozero elements in sparse Jacobian of constraints\n"
"Jv  -- 1D array holding the values of nonzero elements in the sparse Jacobian\n"
"       of constraints at x. Has the same length as Jvi and Jfi.\n"
"\n"
"For constrained problems the gradient is returned in sparse format.\n"
"\n"
"Hi, Hj, and Hv represent the full Hessian and not only the diagonal and the\n"
"upper triangle.\n"
"\n"
"This function is not supposed to be called by the user. It is called by the\n"
"wrapper function gradsphess().\n"
"\n"
"CUTEr tools used: CSGRSH, UGRSH\n";

static PyObject *cuter__gradsphess(PyObject *self, PyObject *args) {
	PyArrayObject *arg1, *arg2, *Mg=NULL, *Mgi, *Mgv, *MJi, *MJfi, *MJv, *MHi, *MHj, *MHv; 
	PyObject *arg3;
	doublereal *x, *v, *g, *sv, *sjv; 
	npy_integer lagrangian;
	npy_integer *si, *sj, *sji, *sjfi, nnzho, nnzjplusn, nnzjplusno; 
	npy_intp dims[1]; 

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
	
	arg2=NULL;
	arg3=NULL;
	if (!PyArg_ParseTuple(args, "O|OO", &arg1, &arg2, &arg3)) 
		return NULL; 	
	
	/* Check bool argument */
	if (arg3!=NULL && arg3==Py_True)
		lagrangian=1;
	else
		lagrangian=0;
	
	/* Check if x is double and of correct dimension */
	if (!(PyArray_Check(arg1) && PyArray_ISFLOAT(arg1) && PyArray_TYPE(arg1)==NPY_DOUBLE && PyArray_NDIM(arg1)==1 && PyArray_DIM(arg1, 0)==CUTEr_nvar)) {
		PyErr_SetString(PyExc_Exception, "Argument 1 must be a 1D double array of length nvar");
		return NULL; 
	}
	
	if (CUTEr_ncon>0) {
		/* Check if v is double and of correct dimension */
		if (arg2!=NULL) {
			/* Check if v is double and of correct dimension */
			if (!(PyArray_Check(arg2) && PyArray_ISFLOAT(arg2) && PyArray_TYPE(arg2)==NPY_DOUBLE && PyArray_NDIM(arg2)==1 && PyArray_DIM(arg2, 0)==CUTEr_ncon)) {
				PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length ncon");
				return NULL; 
			}
		} else {
			PyErr_SetString(PyExc_Exception, "Argument 2 must be specified for constrained problems.");
			return NULL; 
		}
	}

#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: preparing for evaluation\n");
#endif
	x=(npy_double *)PyArray_DATA(arg1);
	si=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sj=(npy_integer *)malloc(CUTEr_nnzh*sizeof(npy_integer));
	sv=(npy_double *)malloc(CUTEr_nnzh*sizeof(npy_double)); 
		
	if (CUTEr_ncon>0) {
		v=(npy_double *)PyArray_DATA(arg2);
		nnzjplusn=CUTEr_nnzj+CUTEr_nvar;
		sji=(npy_integer *)malloc(nnzjplusn*sizeof(npy_integer));
		sjfi=(npy_integer *)malloc(nnzjplusn*sizeof(npy_integer));
		sjv=(npy_double *)malloc(nnzjplusn*sizeof(npy_double)); 
		
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: calling CSGRSH\n");
#endif
		if (lagrangian) {
			CSGRSH((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, &somethingTrue, (integer *)&CUTEr_ncon, v,
					(integer *)&nnzjplusno, (integer *)&nnzjplusn, sjv, (integer *)sji, (integer *)sjfi, 
					(integer *)&nnzho, (integer *)&CUTEr_nnzh, sv, (integer *)si, (integer *)sj);
		} else { 
			CSGRSH((integer *)&CUTEr_nvar, (integer *)&CUTEr_ncon, x, &somethingFalse, (integer *)&CUTEr_ncon, v,
					(integer *)&nnzjplusno, (integer *)&nnzjplusn, sjv, (integer *)sji, (integer *)sjfi, 
					(integer *)&nnzho, (integer *)&CUTEr_nnzh, sv, (integer *)si, (integer *)sj);
		}
		
		extract_sparse_gradient_jacobian(nnzjplusno, sji, sjfi, sjv, (PyArrayObject **)&Mgi, (PyArrayObject **)&Mgv, (PyArrayObject **)&MJi, (PyArrayObject **)&MJfi, (PyArrayObject **)&MJv); 
		
		/* Free temporary storage - Jacobian */
		free(sji);
		free(sjfi);
		free(sjv);
	} else {
		dims[0]=CUTEr_nvar; 
		Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
		g=(npy_double *)PyArray_DATA(Mg);
		
#ifdef PYDEBUG
		fprintf(df, "PyCUTEr: calling UGRSH\n");
#endif
		UGRSH((integer *)&CUTEr_nvar, x, g, (integer *)&nnzho, (integer *)&CUTEr_nnzh, sv, (integer *)si, (integer *)sj);
	}
	
	extract_sparse_hessian(nnzho, si, sj, sv, (PyArrayObject **)&MHi, (PyArrayObject **)&MHj, (PyArrayObject **)&MHv); 
	
	/* Free temporary storage - Hessian */
	free(si);
	free(sj);
	free(sv);
		
	if (CUTEr_ncon>0) {
		return decRefTuple(PyTuple_Pack(8, Mgi, Mgv, MJi, MJfi, MJv, MHi, MHj, MHv)); 
	} else {
		return decRefTuple(PyTuple_Pack(4, Mg, MHi, MHj, MHv)); 
	}
}


static char cuter_report_doc[]=
"Reports usage statistics.\n"
"\n"
"stat=report()\n"
"\n"
"Output\n"
"stat -- dictionary with the usage statistics\n"
"\n"
"The usage statistics dictionary has the following members:\n"
"f      -- number of objective evaluations\n"
"g      -- number of objective gradient evaluations\n"
"H      -- number of objective Hessian evaluations\n"
"Hprod  -- number of Hessian multiplications with a vector\n"
"tsetup -- CPU time used in setup\n"
"trun   -- CPU time used in run\n"
"\n"
"For constrained problems the following additional members are available\n"
"c      -- number of constraint evaluations\n"
"cg     -- number of constraint gradient evaluations\n"
"cH     -- number of constraint Hessian evaluations\n"
"\n"
"CUTEr tools used: CREPRT, UREPRT\n";

static PyObject *cuter_report(PyObject *self, PyObject *args) {
	real calls[7], time[2];
	PyObject *dict;
	
#ifdef PYDEBUG
	fprintf(df, "PyCUTEr: checking arguments\n");
#endif
	if (!check_setup())
		return NULL; 
		
	if (PyObject_Length(args)!=0) {
		PyErr_SetString(PyExc_Exception, "report() takes no arguments");
		return NULL; 
	}
	
	if (CUTEr_ncon>0) 
		CREPRT(calls, time); 
	else
		UREPRT(calls, time); 
		
	dict=PyDict_New();
	PyDict_SetItemString(dict, "f", PyInt_FromLong((long)(calls[0]))); 
	PyDict_SetItemString(dict, "g", PyInt_FromLong((long)(calls[1]))); 
	PyDict_SetItemString(dict, "H", PyInt_FromLong((long)(calls[2]))); 
	PyDict_SetItemString(dict, "Hprod", PyInt_FromLong((long)(calls[3]))); 
	if (CUTEr_ncon>0) {
		PyDict_SetItemString(dict, "c", PyInt_FromLong((long)(calls[4]))); 
		PyDict_SetItemString(dict, "cg", PyInt_FromLong((long)(calls[5]))); 
		PyDict_SetItemString(dict, "cH", PyInt_FromLong((long)(calls[6]))); 
	}
	PyDict_SetItemString(dict, "tsetup", PyFloat_FromDouble((long)(time[0]))); 
	PyDict_SetItemString(dict, "trun", PyFloat_FromDouble((long)(time[1]))); 
	
	return decRefDict(dict); 
}

/* Methods table */
static PyMethodDef _methods[] = {
	{"_dims", cuter__dims, METH_VARARGS, cuter__dims_doc},
	{"_setup", cuter__setup, METH_VARARGS, cuter__setup_doc},
	{"_varnames", cuter__varnames, METH_VARARGS, cuter__varnames_doc},
	{"_connames", cuter__connames, METH_VARARGS, cuter__connames_doc},
	{"objcons", cuter_objcons, METH_VARARGS, cuter_objcons_doc},
	{"obj", cuter_obj, METH_VARARGS, cuter_obj_doc},
	{"cons", cuter_cons, METH_VARARGS, cuter_cons_doc},
	{"lagjac", cuter_lagjac, METH_VARARGS, cuter_lagjac_doc},
	{"jprod", cuter_jprod, METH_VARARGS, cuter_jprod_doc},
	{"hess", cuter_hess, METH_VARARGS, cuter_hess_doc},
	{"ihess", cuter_ihess, METH_VARARGS, cuter_ihess_doc},
	{"hprod", cuter_hprod, METH_VARARGS, cuter_hprod_doc},
	{"gradhess", cuter_gradhess, METH_VARARGS, cuter_gradhess_doc},
	{"_scons", cuter__scons, METH_VARARGS, cuter__scons_doc},
	{"_slagjac", cuter__slagjac, METH_VARARGS, cuter__slagjac_doc},
	{"_sphess", cuter__sphess, METH_VARARGS, cuter__sphess_doc},
	{"_isphess", cuter__isphess, METH_VARARGS, cuter__isphess_doc},
	{"_gradsphess", cuter__gradsphess, METH_VARARGS, cuter__gradsphess_doc},
	{"report", cuter_report, METH_VARARGS, cuter_report_doc}, 
	{NULL, NULL}     /* Marks the end of this structure */
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_pycuteritf(void)  {
	(void) Py_InitModule("_pycuteritf", _methods);
	import_array();  /* Must be present for NumPy.  Called first after above line. */
}

#ifdef __cplusplus
}
#endif
"""
"CUTEr problem binary iterface source code."