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

/* L1HILB and WATSON */
int extra_u_x0(double *x, int n, int num) {
	int i;
	if (num==11) {
		/* L1HILB */
		for(i=0;i<n;i++) {
			x[i]=1.0;
		}
	} else if (num==12) {
		/* WATSON */
		if (n<2 || n>31)
			return 0;
			
		for(i=0;i<n;i++) {
			x[i]=0.0;
		}
	} else {
		return 0; 
	}
	return 1;
}

int extra_u_fg(double *x, int n, double *f, double *g, int num) {
	int i, j, k, imax;
	double tmp, fi, fii, fisign, subsum;
	
	if (num==11) {
		/* L1HILB */
		*f=0.0;
		for(i=0;i<n;i++) {
			g[i]=0.0;
		}
		for(i=0;i<n;i++) {
			tmp=0.0;
			for(j=0;j<n;j++) {
				tmp+=x[j]/(i+1+j+1-1);
			}
			(*f)+=fabs(tmp);
			for(k=0;k<n;k++) {
				if (tmp>=0.0) {
					g[k]+=1.0/(i+1+k+1-1);
				} else {
					g[k]-=1.0/(i+1+k+1-1);
				}
			}
		}
	} else if (num==12) {
		/* WATSON */
		*f=0.0;
		for(i=0;i<n;i++) {
			g[i]=0.0;
		}
		imax=1;
		for(i=1;i<=31;i++) {
			if (i==1) {
				fi=x[0];
			} else if (i==2) {
				fi=x[1]-x[0]*x[0]-1.0;
			} else {
				fi=0.0;
				for(j=2;j<=n;j++) {
					fi+=(j-1)*x[j-1]*pow((i-2)*1.0/29, j-2);
				}
				fii=0.0;
				for(j=1;j<=n;j++) {
					fii+=x[j-1]*pow((i-2)*1.0/29, j-1);
				}
				fi-=fii*fii;
			}
			tmp=fabs(fi);
			if (tmp>=*f) {
				*f=tmp;
				imax=i;
				if (fi>=0) {
					fisign=1.0;
				} else {
					fisign=-1.0;
				}
				subsum=fii;
			} 
		}
		if (imax==1) {
			g[0]=fisign;
		} else if (imax==2) {
			g[1]=fisign;
			g[0]=-fisign*2*x[0];
		} else {
			for(k=1;k<=n;k++) {
				g[k-1]=fisign * ((k-1)*pow((imax-2)*1.0/29, k-2)-2*subsum*pow((imax-2)*1.0/29, k-1));
			}
		}
	} else {
		return 0;
	}
	return 1;
}


/* Decrease reference count for newly created tuple members */
PyObject *decRefTuple(PyObject *tuple) {
	Py_ssize_t pos; 
	for(pos=0;pos<PyTuple_Size(tuple);pos++) {
		Py_XDECREF(PyTuple_GetItem(tuple, pos));
	}
	return tuple;
}

/* tnsunc.c */

static char startxu_doc[]=
"Large scale unconstrained nonsmooth problems - initial point.\n"
"Wraps FORTRAN function startxu.\n"
"\n"
"x0=startxu(num, n)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"n   -- problem dimension\n"
"\n"
"Output\n"
"x0  -- initial point (NumPy array)\n";

static PyObject *startxu_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n; 
	integer n, problem;
	doublereal *xini;
	PyArrayObject *Mx0; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "startxu: checking arguments\n");
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
	
	if (problem<1 || problem>12) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=i_n;
	
	if (
		n<1 || 
		(problem==1 && n%2!=0) 
	) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	if (problem<=10) {
		startxu_(&n, xini, &problem);
		
		/* Handle error */
		if (problem<0) {
			Py_XDECREF(Mx0);
			PyErr_SetString(PyExc_Exception, "Call to startxu failed.");
			return NULL;
		}
	} else {
		if (extra_u_x0(xini, n, problem)==0) {
			Py_XDECREF(Mx0);
			PyErr_SetString(PyExc_Exception, "Call to startxu failed.");
			return NULL;
		}
	}
	
	
	return (PyObject *)Mx0; 
}

static char funcu_doc[]=
"Large scale unconstrained nonsmooth problems - function and subgradient.\n"
"Wraps FORTRAN function funcu.\n"
"\n"
"(f,g)=funcu(num, x)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"x   -- function argument vector of length n\n"
"\n"
"Output (tuple)\n"
"f -- function value - 1-dimensional array of length 1\n"
"g -- subgradient - 1-dimensional array with same length as x\n"; 

static PyObject *funcu_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *g, *x; 
	PyArrayObject *Mx, *Mf, *Mg; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "funcu: checking arguments\n");
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
	
	if (problem<1 || problem>12) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array");
		return NULL; 
	}
	
	n=PyArray_DIM(Mx, 0);
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Allocate vector for g */
	dims[0]=n;
	Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	g=(npy_double *)PyArray_DATA(Mg);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	if (problem<=10) {
		funcu_(&n, x, f, g, &problem);
		
		/* Error */
		if (problem<0) {
			Py_XDECREF(Mf);
			Py_XDECREF(Mg);
			PyErr_SetString(PyExc_Exception, "Error in function evaluation.");
			return NULL;
		}
	} else {
		if (extra_u_fg(x, n, f, g, problem)==0) {
			Py_XDECREF(Mf);
			Py_XDECREF(Mg);
			PyErr_SetString(PyExc_Exception, "Error in function evaluation.");
			return NULL;
		}
	}
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(2, Mf, Mg));
}


/* tnsboc.c */

static char startxb_doc[]=
"Large scale bound unconstrained nonsmooth problems - initial point.\n"
"Wraps FORTRAN function startxb.\n"
"\n"
"x0=startxb(num, n)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"n   -- problem dimension\n"
"\n"
"Output\n"
"x0  -- initial point (NumPy array)\n";

static PyObject *startxb_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n; 
	integer n, problem;
	doublereal *xini;
	PyArrayObject *Mx0; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "startxb: checking arguments\n");
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
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=i_n;
	
	if (
		n<1 || 
		(problem==1 && n%2!=0) 
	) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	startxb_(&n, xini, &problem);
	
	/* Handle error */
	if (problem<0) {
		Py_XDECREF(Mx0);
		PyErr_SetString(PyExc_Exception, "Call to startxb failed.");
		return NULL;
	}
	
	return (PyObject *)Mx0; 
}

static char bounds_doc[]=
"Large scale bound unconstrained nonsmooth problems - bounds.\n"
"Wraps FORTRAN function bounds.\n"
"\n"
"(type, xl, xu)=bounds(num, n)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"n   -- problem dimension\n"
"\n"
"Output\n"
"type -- type of bound (NumPy array)\n"
"        0=unbounded, 1=lower, 2=both, 3=upper\n"
"xl   -- lower bound (NumPy array)\n"
"xh   -- upper bound (NumPy array)\n"; 

static PyObject *bounds_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_n, i; 
	integer n, problem;
	doublereal *xl, *xh;
	integer *xini, *type; 
	PyArrayObject *Mxl, *Mxh, *Mtype; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "startxb: checking arguments\n");
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
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	/* Set n */
	n=i_n;
	
	if (
		n<1 || 
		(problem==1 && n%2!=0) 
	) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	/* Allocate vector for xl, xh */
	dims[0]=n;
	Mxl=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xl=(npy_double *)PyArray_DATA(Mxl);
	
	dims[0]=n;
	Mxh=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xh=(npy_double *)PyArray_DATA(Mxh);
	
	/* Allocate dummy integer array */
	type=(integer*)malloc(n*sizeof(integer)); 
	
	/* Call */
	bounds_(&n, type, xl, xh, &problem);
	
	/* Handle error */
	if (problem<0) {
		Py_XDECREF(Mxl);
		Py_XDECREF(Mxh);
		free(type);
		PyErr_SetString(PyExc_Exception, "Call to startxb failed.");
		return NULL;
	}
	
	/* Prepare integer array */
	Mtype=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);
	for(i=0;i<n;i++) {
		*((npy_int*)PyArray_GETPTR1(Mtype, i))=type[i];
	}
	
	/* Free dummy integer array */
	free(type);
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(3, Mtype, Mxl, Mxh));
}

static char funcb_doc[]=
"Large scale bound constrained nonsmooth problems - function and subgradient.\n"
"Wraps FORTRAN function funcb.\n"
"\n"
"(f,g)=funcb(num, x)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"x   -- function argument vector of length n\n"
"\n"
"Output (tuple)\n"
"f -- function value - 1-dimensional array of length 1\n"
"g -- subgradient - 1-dimensional array with same length as x\n"; 

static PyObject *funcb_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *g, *x; 
	PyArrayObject *Mx, *Mf, *Mg; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "funcb: checking arguments\n");
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
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array");
		return NULL; 
	}
	
	n=PyArray_DIM(Mx, 0);
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Allocate vector for g */
	dims[0]=n;
	Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	g=(npy_double *)PyArray_DATA(Mg);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	funcb_(&n, x, f, g, &problem);
	
	/* Error */
	if (problem<0) {
		Py_XDECREF(Mf);
		Py_XDECREF(Mg);
		PyErr_SetString(PyExc_Exception, "Error in function evaluation.");
		return NULL;
	}
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(2, Mf, Mg));
}


/* tnsiec.c */

int get_m(int n, int nf, int nc) {
	switch (nc) {
		case 0:
			return n-2;
		case 2:
			return 2;
		case 3: 
			return 4;
		case 5:
			return n-1;
		case 6:
			return n-2;
		default:
			return 1;
	}
}

static char xinit3_doc[]=
"Large scale inequality unconstrained nonsmooth problems - initial point.\n"
"Wraps FORTRAN function xinit3.\n"
"\n"
"(x0, m)=xinit3(num, cnum, n, m)\n"
"\n"
"Input\n"
"num  -- problem number (0-9)\n"
"cnum -- constraint number (0-7)\n"
"n    -- problem dimension\n"
"\n"
"Output\n"
"x0  -- initial point (NumPy array)\n"
"m   -- number of constraints\n"; 

static PyObject *xinit3_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_cons, i_n, i_m; 
	integer n, m, problem, cons;
	doublereal *xini;
	PyArrayObject *Mx0, *Mm; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "xinit3: checking arguments\n");
#endif
	if (PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes exactly three arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iii", &i_problem, &i_cons, &i_n)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}

	problem=i_problem+1;
	cons=i_cons+1; 
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (cons<1 || cons>8) {
		PyErr_SetString(PyExc_Exception, "Bad constraint number.");
		return NULL; 
	}
	
	/* Set n, m */
	n=i_n;
	m=get_m(i_n, i_problem, i_cons); 
	
	if (
		n<1 || 
		(problem==1 && n%2!=0) 
	) {
		PyErr_SetString(PyExc_Exception, "Bad problem dimension.");
		return NULL; 
	}
	
	/* Allocate vector for x0 */
	dims[0]=n;
	Mx0=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	xini=(npy_double *)PyArray_DATA(Mx0);
	
	/* Call */
	xinit3_(&n, &m, xini, &problem, &cons);
	
	/* Handle error */
	if (problem<0) {
		Py_XDECREF(Mx0);
		PyErr_SetString(PyExc_Exception, "Call to startxb failed.");
		return NULL;
	}
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(2, Mx0, PyInt_FromLong((long)m)));
}

static char funci_doc[]=
"Large scale bound constrained nonsmooth problems - function and subgradient.\n"
"Wraps FORTRAN function funci.\n"
"\n"
"(f,g)=funci(num, x)\n"
"\n"
"Input\n"
"num -- problem number (0-9)\n"
"x   -- function argument vector of length n\n"
"\n"
"Output (tuple)\n"
"f -- function value - 1-dimensional array of length 1\n"
"g -- subgradient - 1-dimensional array with same length as x\n"; 

static PyObject *funci_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_function, i_function_given=0, i; 
	integer n, m, problem, function;
	doublereal *f, *g, *x; 
	PyArrayObject *Mx, *Mf, *Mg; 
	npy_intp dims[1];
	
#ifdef PYDEBUG
	fprintf(df, "funci: checking arguments\n");
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
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array");
		return NULL; 
	}
	
	n=PyArray_DIM(Mx, 0);
	
	/* Allocate vector for f */
	dims[0]=1;
	Mf=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	f=(npy_double *)PyArray_DATA(Mf);
	
	/* Allocate vector for g */
	dims[0]=n;
	Mg=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	g=(npy_double *)PyArray_DATA(Mg);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	funci_(&n, x, f, g, &problem);
	
	/* Error */
	if (problem<0) {
		Py_XDECREF(Mf);
		Py_XDECREF(Mg);
		PyErr_SetString(PyExc_Exception, "Error in function evaluation.");
		return NULL;
	}
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(2, Mf, Mg));
}

static char cineq_doc[]=
"Large scale bound constrained nonsmooth problems - function and subgradient.\n"
"Wraps FORTRAN function cineq.\n"
"\n"
"(c,J)=cineq(num, cnum, x, m)\n"
"\n"
"Input\n"
"num  -- problem number (0-9)\n"
"cnum -- constraint function number (0-7)\n"
"x    -- function argument vector of length n\n"
"\n"
"Output (tuple)\n"
"c -- function value - 1-dimensional array of length 1\n"
"J -- subgradient - 1-dimensional array with same length as x\n"; 

static PyObject *cineq_wrap(PyObject *self, PyObject *args) {
	int i_problem, i_cons, i_m, i; 
	integer n, m, problem, cons;
	doublereal *c, *J, *x; 
	PyArrayObject *Mc, *MJ, *Mx; 
	npy_intp dims[2];
	
#ifdef PYDEBUG
	fprintf(df, "cineq: checking arguments\n");
#endif
	if (PyObject_Length(args)!=3) {
		PyErr_SetString(PyExc_Exception, "Function takes three arguments.");
		return NULL; 
	}
	
	if (!PyArg_ParseTuple(args, "iiO", &i_problem, &i_cons, &Mx)) {
		PyErr_SetString(PyExc_Exception, "Bad input arguments.");
		return NULL; 
	}
	
	problem=i_problem+1;
	cons=i_cons+1;
	
	if (problem<1 || problem>10) {
		PyErr_SetString(PyExc_Exception, "Bad problem number.");
		return NULL; 
	}
	
	if (cons<1 || cons>8) {
		PyErr_SetString(PyExc_Exception, "Bad constraint number.");
		return NULL; 
	}
	
	if (!(PyArray_Check(Mx) && PyArray_ISFLOAT(Mx)&& PyArray_TYPE(Mx)==NPY_DOUBLE && PyArray_NDIM(Mx)==1)) {
		PyErr_SetString(PyExc_Exception, "Argument 2 must be a 1D double array");
		return NULL; 
	}
	
	n=PyArray_DIM(Mx, 0);
	m=get_m(n, i_problem, i_cons); 
	
	/* Allocate vector for c */
	dims[0]=m;
	Mc=(PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	c=(npy_double *)PyArray_DATA(Mc);
	
	/* Allocate vector for J */
	dims[0]=m;
	dims[1]=n;
	MJ=(PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	J=(npy_double *)PyArray_DATA(MJ);
	
	/* Extract x */
	x=(npy_double *)PyArray_DATA(Mx);
	
	/* Call */
	cineq_(&n, &m, x, c, J, &problem, &cons);
	
	/* Error */
	if (problem<0) {
		Py_XDECREF(Mc);
		Py_XDECREF(MJ);
		PyErr_SetString(PyExc_Exception, "Error in function evaluation.");
		return NULL;
	}
	
	/* Return tuple */
	return decRefTuple(PyTuple_Pack(2, Mc, MJ));
}


/* Methods table */
static PyMethodDef _karmitsa_methods[] = {
	{"startxu", startxu_wrap, METH_VARARGS, startxu_doc},
	{"funcu", funcu_wrap, METH_VARARGS, funcu_doc},
	{"startxb", startxb_wrap, METH_VARARGS, startxb_doc},
	{"bounds", bounds_wrap, METH_VARARGS, bounds_doc},
	{"funcb", funcb_wrap, METH_VARARGS, funcb_doc},
	{"xinit3", xinit3_wrap, METH_VARARGS, xinit3_doc},
	{"funci", funci_wrap, METH_VARARGS, funci_doc}, 
	{"cineq", cineq_wrap, METH_VARARGS, cineq_doc}, 
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_karmitsa()  {
	(void) Py_InitModule("_karmitsa", _karmitsa_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_karmitsa_amd64()  {
	(void) Py_InitModule("_karmits_amd64", _karmitsa_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_karmitsa_i386()  {
	(void) Py_InitModule("_karmitsa_i386", _karmitsa_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

#ifdef __cplusplus
}
#endif
