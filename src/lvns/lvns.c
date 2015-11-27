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

/* lvumm.c */
/* Number of parameters and number of partial functions for the problems */
int n_umm[]={2,2,2,3,4,4, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 9,7,10,20,10,11,20,11};
int m_umm[]={3,3,2,6,4,4,21,15,11,20,21,21,61,21,30,51,11,41,5, 9,18, 2,10,31,65};

static char tiud06_doc[]=
"Unconstrained nonlinear minimax problems - problem information.\n"
"Wraps FORTRAN function tiud06.\n"
"\n"
"data=tiud06(num)\n"
"\n"
"Input\n"
"num -- problem number (0-24)\n"
"\n"
"Output\n"
"data -- dictionary with problem information\n"
"\n"
"The dictionary has the following members:\n"
"n      -- number of parameters\n"
"m      -- number of partial functions\n"
"x0     -- initial point (NumPy array)\n"
"type   -- type of function\n"
"          <0 .. maximum of partial functions\n"
"           0 .. maximum of absolute values\n"
"xmax   -- maximum stepsize for the problem\n"
"\n"
"The fmin value returned by the FORTRAN function tiud06 is incorrect.\n"
"Therefore it is not included in the dictionary.\n"; 

static PyObject *tiud06_wrap(PyObject *self, PyObject *args) {
	int i_problem; 
	integer n, m, problem, objtype, err;
	doublereal fmin=0., *xini, xmax;
	PyObject *dict, *tmpo; 
	PyArrayObject *Mx0; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tiud06: checking arguments\n");
#endif
	if (PyObject_Length(args)!=1) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly one argument.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "i", &i_problem)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	
	if (problem<1 || problem>25) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n,m */
	n=n_umm[i_problem];
	m=m_umm[i_problem];
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	tiud06_(&n, &m, xini, &fmin, &xmax, &problem, &objtype, &err);
	
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
	
	tmpo=PyInt_FromLong((long)objtype);
	PyDict_SetItemString(dict, "type", tmpo); 
	Py_XDECREF(tmpo);
	
	tmpo=PyInt_FromLong((long)n);
	PyDict_SetItemString(dict, "n", tmpo);
	Py_XDECREF(tmpo);
	
	tmpo=PyInt_FromLong((long)m);
	PyDict_SetItemString(dict, "m", tmpo); 
	Py_XDECREF(tmpo);
	
	PyDict_SetItemString(dict, "x0", (PyObject *)Mx0); 
	Py_XDECREF(Mx0);
	
	/* Check reference count 
	printf("%d\n", Mx0->ob_refcnt); 
	*/
	
	return dict; 
}

static char tafu06_doc[]=
"Unconstrained nonlinear minimax problems - function values.\n"
"Wraps FORTRAN function tafu06.\n"
"\n"
"fi=tafu06(num, x, i)\n"
"\n"
"Input\n"
"num -- problem number (0-24)\n"
"x   -- function argument vector of length n\n"
"i   -- optional partial function index (0..m-1)\n"
"       All partial functions are returned if i is not provided.\n"
"\n"
"Output\n"
"fi -- 1-dimensional array of length m or 1 (if i is given)\n"; 

static PyObject *tafu06_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *x; 
	PyArrayObject *Mx, *Mf; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tafu06: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2 && PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes two or three arguments.");
		return NULL; 
	}
	
	if (PyObject_Length(args)==2) {
		if (!PyArg_ParseTuple(args, "iO", &i_problem, &Mx)) {
			PyErr_SetString(PyExc_Exception, "Bad input arguments.");
			return NULL; 
		}
	} else {
		if (!PyArg_ParseTuple(args, "iOi", &i_problem, &Mx, &i_function)) {
			PyErr_SetString(PyExc_Exception, "Bad input arguments.");
			return NULL; 
		}
		function=i_function+1;
		i_function_given=1;
		
		if (function<1 || function>m) {
			PyErr_SetString(PyExc_Exception, "Bad partial function number.");
			return NULL; 
		}
	}
	problem=i_problem+1;
	
	n=n_umm[i_problem]; 
	m=m_umm[i_problem]; 
	
	if (problem<1 || problem>25) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1 && PyArray_DIM(Mx, 0)==n)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length n");
		return NULL; 
	}
	
	/* Allocate vector for f */
	if (!i_function_given)
		dims[0]=m;
	else
		dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	if (i_function_given) {
		tafu06_(&n, &function, x, f, &problem);
	} else {
		for(i=0;i<m;i++) {
			function=i+1;
			tafu06_(&n, &function, x, &f[i], &problem);
		}
	}
	
	return (PyObject *)Mf; 
}


/* lvuns.c */
/* Number of parameters for the problems */
int n_uns[]={2,2,2,2,2,2,2,2,2,2,4,5,5,5,6,10,10,12,20,20,48,50,50,50,15};

static char tiud19_doc[]=
"Unconstrained nonsmooth problems - problem information.\n"
"Wraps FORTRAN function tiud19.\n"
"\n"
"data=tiud19(num)\n"
"\n"
"Input\n"
"num -- problem number (0-24)\n"
"\n"
"Output\n"
"data -- dictionary with problem information\n"
"\n"
"The dictionary has the following members:\n"
"n      -- number of parameters\n"
"x0     -- initial point (NumPy array)\n"
"xmax   -- maximum stepsize for the problem\n"
"\n"
"The fmin value returned by the FORTRAN function tiud19 is incorrect.\n"
"Therefore it is not included in the dictionary.\n"; 

static PyObject *tiud19_wrap(PyObject *self, PyObject *args) {
	int i_problem; 
	integer n, problem, err;
	doublereal fmin=0., *xini, xmax;
	PyObject *dict, *tmpo; 
	PyArrayObject *Mx0; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tiud19: checking arguments\n");
#endif
	if (PyObject_Length(args)!=1) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly one argument.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "i", &i_problem)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	
	if (problem<1 || problem>25) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=n_uns[i_problem];
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	tiud19_(&n, xini, &fmin, &xmax, &problem, &err);
	
	/* Handle error */
	if (err) {
		Py_XDECREF(Mx0);
		PyErr_SetString(PyExc_Exception, "Call to tiud19 failed.");
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
	
	tmpo=PyInt_FromLong((long)n);
	PyDict_SetItemString(dict, "n", tmpo);
	Py_XDECREF(tmpo);
	
	PyDict_SetItemString(dict, "x0", (PyObject *)Mx0); 
	Py_XDECREF(Mx0);
	
	return dict; 
}

static char tffu19_doc[]=
"Unconstrained nonsmooth problems - function values.\n"
"Wraps FORTRAN function tffu19.\n"
"\n"
"f=tffu19(num, x)\n"
"\n"
"Input\n"
"num -- problem number (0-24)\n"
"x   -- function argument vector of length n\n"
"\n"
"Output\n"
"data -- 1-dimensional array of length 1\n"; 

static PyObject *tffu19_wrap(PyObject *self, PyObject *args) {
	int i_problem; 
	integer n, problem;
	doublereal *f, *x; 
	PyArrayObject *Mx, *Mf; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tffu19: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2) {
		PyErr_SetString(PyExc_Exception, "Function takes two arguments.");
		return NULL; 
	}
	
	if (PyObject_Length(args)==2) {
		if (!PyArg_ParseTuple(args, "iO", &i_problem, &Mx)) {
			PyErr_SetString(PyExc_Exception, "Bad input arguments.");
			return NULL; 
		}
	} 
	
	problem=i_problem+1;
	
	n=n_uns[i_problem]; 

	if (problem<1 || problem>25) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1 && PyArray_DIM(Mx, 0)==n)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length n");
		return NULL; 
	}
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	tffu19_(&n, x, f, &problem);
	
	return (PyObject *)Mf; 
}


/* lvlcmm.c */
/* Number of parameters for the problems */
int n_lcmm[]  ={2,2,2,2, 6,  7,8,10,20,20,  9,10, 7,8,16};
int m_lcmm[]  ={3,3,3,3, 3,163,8, 6,14,38,124, 9,13,4,19};
int m1_lcmm[] ={3,3,3,3, 3,163,8, 6,14,38,124, 9,15,7,20};
int nc_lcmm[] ={1,1,1,1,15,  7,1, 3, 4, 0,  4, 5, 2,3, 1};

static char eild22_doc[]=
"Linearly constrained minimax problems - problem information.\n"
"Wraps FORTRAN function eild22.\n"
"\n"
"data=eild22(num)\n"
"\n"
"Input\n"
"num -- problem number (0-14)\n"
"\n"
"Output\n"
"data -- dictionary with problem information\n"
"\n"
"The dictionary has the following members:\n"
"n      -- number of parameters\n"
"m      -- number of partial functions\n"
"nb     -- number of box constrants (0 or n)\n"
"nc     -- number of constraint functions\n"
"x0     -- initial point (NumPy array)\n"
"bt     -- bound types\n"
"          0-no bound, 1-lower bound, 2-upper bound, 3-both bounds,\n"
"          5-equality bound\n"
"xl     -- lower bound on variables\n"
"xh     -- upper bound on variables\n"
"ct     -- constraint types, same codes as for bt\n"
"          there are no problems with ct=0 in the test set\n"
"cl     -- lower bound on constraint functions\n"
"ch     -- upper bound on constraint functions\n"
"Jc     -- Jacobian of the constraint functions\n"
"xmax   -- maximum stepsize for the problem\n"
"type   -- type of function\n"
"          <0 .. maximum of partial functions\n"
"           0 .. maximum of absolute values\n"
"\n"
"The fmin value returned by the FORTRAN function eild22 is incorrect.\n"
"Therefore it is not included in the dictionary.\n"; 

static PyObject *eild22_wrap(PyObject *self, PyObject *args) {
	int i_problem, i; 
	integer n, m, m1, nc, nb, problem, objtype, err, *i_bt, *i_ct;
	doublereal fmin=0., *xini, xmax, *xl, *xh, *cl, *ch, *J;
	PyObject *dict, *value; 
	PyArrayObject *Mx0, *Mxl, *Mxh, *Mcl, *Mch, *MJ, *Mbt, *Mct, *key; 
	Py_ssize_t pos;
	npy_intp dims[2];
	npy_int *bt, *ct; 
	
#ifdef PYDEBUG
	fprintf(df, "eild22: checking arguments\n");
#endif
	if (PyObject_Length(args)!=1) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly one argument.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "i", &i_problem)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	
	if (problem<1 || problem>15) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n, m, nc */
	n=n_lcmm[i_problem];
	m=m_lcmm[i_problem];
	m1=m1_lcmm[i_problem];
	nc=nc_lcmm[i_problem];
	
	/* Allocate vector for x0, xl, xh, bt */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	Mxl=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xl=(npy_double *)PyArray_DATA(Mxl);
	Mxh=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xh=(npy_double *)PyArray_DATA(Mxh);
	i_bt=(integer *)malloc(n*sizeof(integer));
	
	/* Allocate vector for cl, ch, ct */
	dims[0]=nc;
	Mcl=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	cl=(npy_double *)PyArray_DATA(Mcl);
	Mch=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	ch=(npy_double *)PyArray_DATA(Mch);
	i_ct=(integer *)malloc(nc*sizeof(integer));
	
	/* Allocate vector for Jc */
	dims[0]=nc;
	dims[1]=n;
	MJ=(PyArrayObject *)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL, NULL, 0, /* NPY_F_CONTIGUOUS */ 0, NULL);
	J=(npy_double *)PyArray_DATA(MJ);
	
	/* Call */
	/* Must lie about m for problems 12-14. Pass m1 instead of m. */
	eild22_(&n, &m1, &nb, &nc, xini, i_bt, xl, xh, i_ct, cl, ch, J, &fmin, &xmax, &problem, &objtype, &err);
	
	/* Handle error */
	if (err) {
		Py_XDECREF(Mx0);
		Py_XDECREF(Mxl);
		Py_XDECREF(Mxh);
		Py_XDECREF(Mcl);
		Py_XDECREF(Mch);
		Py_XDECREF(MJ);
		free(i_bt);
		free(i_ct);
		PyErr_SetString(PyExc_Exception, "Call to eild22 failed.");
		return NULL;
	}
	
	/* Prepare storage for bound and constraint types */
	dims[0]=nb;
	Mbt=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
	bt=(int *)PyArray_DATA(Mbt);
	dims[0]=nc;
	Mct=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
	ct=(int *)PyArray_DATA(Mct);
	
	/* Copy i_bt */
	for(i=0;i<(int)nb;i++) {
		bt[i]=i_bt[i];
	}
	
	/* Copy i_ct */
	for(i=0;i<(int)nc;i++) {
		ct[i]=i_ct[i];
	}
	
	/* Free i_bt and i_ct */
	free(i_bt);
	free(i_ct);
	
	/* Prepare return value */
	dict=PyDict_New();
	
	/* fmin is not valid
	tmpo=PyFloat_FromDouble((double)fmin);
	PyDict_SetItemString(dict, "fmin", tmpo);
	Py_XDECREF(tmpo);
	*/
	
	PyDict_SetItemString(dict, "xmax", PyFloat_FromDouble((double)xmax));
	PyDict_SetItemString(dict, "type", PyInt_FromLong((long)objtype));
	PyDict_SetItemString(dict, "n", PyInt_FromLong((long)n));
	PyDict_SetItemString(dict, "m", PyInt_FromLong((long)m));
	PyDict_SetItemString(dict, "nc", PyInt_FromLong((long)nc));
	PyDict_SetItemString(dict, "x0", (PyObject *)Mx0); 
	PyDict_SetItemString(dict, "bt", (PyObject *)Mbt); 
	PyDict_SetItemString(dict, "xl", (PyObject *)Mxl); 
	PyDict_SetItemString(dict, "xh", (PyObject *)Mxh); 
	PyDict_SetItemString(dict, "ct", (PyObject *)Mct); 
	PyDict_SetItemString(dict, "cl", (PyObject *)Mcl); 
	PyDict_SetItemString(dict, "ch", (PyObject *)Mch); 
	PyDict_SetItemString(dict, "Jc", (PyObject *)MJ); 
	
	/* Decrease reference count for members */
	pos=0;
	while (PyDict_Next(dict, &pos, (PyObject **)&key, &value)) {
		Py_XDECREF(value);
	}
	
	return dict; 
}

static char tafu22_doc[]=
"Linearly constrained minimax problems - function values.\n"
"Wraps FORTRAN function tafu22.\n"
"\n"
"fi=tafu22(num, x, i)\n"
"\n"
"Input\n"
"num -- problem number (0-14)\n"
"x   -- function argument vector of length n\n"
"i   -- optional partial function index (0..m-1)\n"
"       All partial functions are returned if i is not provided.\n"
"\n"
"Output\n"
"fi -- 1-dimensional array of length m or 1 (if i is given)\n"; 

static PyObject *tafu22_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *x; 
	PyArrayObject *Mx, *Mf; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "tafu06: checking arguments\n");
#endif
	if (PyObject_Length(args)!=2 && PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes two or three arguments.");
		return NULL; 
	}
	
	if (PyObject_Length(args)==2) {
		if (!PyArg_ParseTuple(args, "iO", &i_problem, &Mx)) {
			PyErr_SetString(PyExc_Exception, "Bad input arguments.");
			return NULL; 
		}
	} else {
		if (!PyArg_ParseTuple(args, "iOi", &i_problem, &Mx, &i_function)) {
			PyErr_SetString(PyExc_Exception, "Bad input arguments.");
			return NULL; 
		}
		function=i_function+1;
		i_function_given=1;
		
		if (function<1 || function>m) {
			PyErr_SetString(PyExc_Exception, "Bad partial function number.");
			return NULL; 
		}
	}
	problem=i_problem+1;
	
	n=n_lcmm[i_problem]; 
	m=m_lcmm[i_problem]; 
	
	if (problem<1 || problem>25) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1 && PyArray_DIM(Mx, 0)==n)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array of length n");
		return NULL; 
	}
	
	/* Allocate vector for f */
	if (!i_function_given)
		dims[0]=m;
	else
		dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	if (i_function_given) {
		tafu22_(&n, &function, x, f, &problem);
	} else {
		for(i=0;i<m;i++) {
			function=i+1;
			tafu22_(&n, &function, x, &f[i], &problem);
		}
	}
	
	return (PyObject *)Mf; 
}



/* Methods table */
static PyMethodDef _lvns_methods[] = {
	{"tiud06", tiud06_wrap, METH_VARARGS, tiud06_doc},
	{"tafu06", tafu06_wrap, METH_VARARGS, tafu06_doc},
	{"tiud19", tiud19_wrap, METH_VARARGS, tiud19_doc},
	{"tffu19", tffu19_wrap, METH_VARARGS, tffu19_doc},
	{"eild22", eild22_wrap, METH_VARARGS, eild22_doc},
	{"tafu22", tafu22_wrap, METH_VARARGS, tafu22_doc},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_lvns()  {
	(void) Py_InitModule("_lvns", _lvns_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_lvns_amd64()  {
	(void) Py_InitModule("_lvns_amd64", _lvns_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_lvns_i386()  {
	(void) Py_InitModule("_lvns_i386", _lvns_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
