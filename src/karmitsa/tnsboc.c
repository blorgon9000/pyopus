/* tnsboc.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Table of constant values */

static integer c__9 = 9;
static integer c__1 = 1;
static doublereal c_b51 = 1.;
static doublereal c_b66 = 3.5;

/* *********************************************************************** */


/*     Test problems for NonSmooth BOund Constrained minimization */


/*     TNSBOC includes the following subroutines */

/*     S   STARTX          Initiation of variables (not necessarily */
/*                           feasible). */
/*     S   BOUNDS          Bound constraints. */
/*     S   FEASIX          Projection of starting points to feasible */
/*                           region. */
/*     S   FUNC            Computation of the value and the subgradient */
/*                           of the objective function. */


/*     Napsu Karmitsa (2003, bound constrained version 2006-2007) */

/*     Haarala M., Miettinen K. and Mäkelä M.M.: New Limited Memory */
/*     Bundle Method for Large-Scale Nonsmooth Optimization, Optimization */
/*     Methods and Software, Vol. 19, No. 6, 2004, 673-692. */

/*     Karmitsa N.: Test Problems for Large-Scale Nonsmooth Minimization, */
/*     Reports of the Department of Mathematical Information Technology, */
/*     Series B, Scientific Computing, B 4/2007, University of Jyväskylä, */
/*     Jyväskylä, 2007. */


/* *********************************************************************** */

/*     * SUBROUTINE STARTX * */


/*     * Purpose * */


/*     Initiation of X. */


/*     * Calling sequence * */

/*     CALL STARTX(N,X,NEXT) */


/*     * Parameters * */

/*     II  N          Number of variables. */
/*     RO  X(N)       Vector of variables. */
/*     RI  NEXT       Problem number. */


/*     * Problems * */

/*     1.  Generalization of MAXQ (convex). */
/*     2.  Generalization of MXHILB (convex). */
/*     3.  Chained LQ (convex). */
/*     4.  Chained CB3 (convex). */
/*     5.  Chained CB3 2 (convex). */
/*     6.  Number of active faces (nonconvex). */
/*     7.  Nonsmooth generalization of Brown function 2 (nonconvex). */
/*     8.  Chained Mifflin 2 (nonconvex). */
/*     9.  Chained crescent (nonconvex). */
/*     10. Chained crescent 2 (nonconvex). */


/*     Napsu Haarala (2003) */

/*     Haarala M., Miettinen K. and Mäkelä M.M.: New Limited Memory */
/*     Bundle Method for Large-Scale Nonsmooth Optimization, Optimization */
/*     Methods and Software, Vol. 19, No. 6, 2004, 673-692.. */

/* Subroutine */ int startxb_(integer *n, doublereal *x, integer *next)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);

    /* Local variables */
    static integer i__;

    /* Fortran I/O blocks */
    static cilist io___1 = { 0, 6, 0, 0, 0 };


/*     Scalar Arguments */
/*     Array Arguments */
/*     Local Arguments */
    /* Parameter adjustments */
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L40;
	case 5:  goto L40;
	case 6:  goto L60;
	case 7:  goto L70;
	case 8:  goto L80;
	case 9:  goto L90;
	case 10:  goto L90;
    }
    s_wsle(&io___1);
    do_lio(&c__9, &c__1, "Error: Not such a problem.", (ftnlen)26);
    e_wsle();
    *next = -1;
    return 0;

/*     Generalization of MAXQ (convex) */

L10:
    i__1 = *n / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = (doublereal) i__;
/* L11: */
    }
    i__1 = *n;
    for (i__ = *n / 2 + 1; i__ <= i__1; ++i__) {
	x[i__] = -((doublereal) i__);
/* L12: */
    }
    return 0;

/*     Generalization of MXHILB (convex) */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L21: */
    }
    return 0;

/*     Chained LQ (convex) */

L30:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -.5;
/* L31: */
    }
    return 0;

/*     Chained CB3 1 and 2 (convex) */

L40:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L41: */
    }
    return 0;

/*     Number of active faces (nonconvex) */

L60:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L61: */
    }
    return 0;

/*     Nonsmooth generalization of Brown function 2 (nonconvex) */

L70:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -1.;
	} else {
	    x[i__] = 1.;
	}
/* L71: */
    }
    return 0;

/*     Chained Mifflin 2 (nonconvex) */

L80:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L81: */
    }
    return 0;

/*     Chained crescent (nonconvex) */

L90:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -1.5;
	} else {
	    x[i__] = 2.;
	}
/* L91: */
    }
    return 0;
} /* startxb_ */

/* *********************************************************************** */

/*     * SUBROUTINE BOUNDS * */


/*     * Purpose * */


/*     Defination of bound constraint. */


/*     * Calling sequence * */

/*     CALL BOUNDS(N,IB,XL,XU,NEXT) */


/*     * Parameters * */

/*     II  N          Number of variables. */
/*     IO  IB(N)      Type of bound constraints: */
/*                      0  - X(I) is unbounded, */
/*                      1  - X(I) has only a lower bound, */
/*                      2  - X(I) has both lower and upper bounds, */
/*                      3  - X(I) has only an upper bound. */
/*     RO  XL(N)      Lower bounds for variables. */
/*     RO  XU(N)      Upper bounds for variables. */
/*     RI  NEXT       Problem number. */


/*     * Problems * */

/*     1.  Generalization of MAXQ (convex). */
/*     2.  Generalization of MXHILB (convex). */
/*     3.  Chained LQ (convex). */
/*     4.  Chained CB3 (convex). */
/*     5.  Chained CB3 2 (convex). */
/*     6.  Number of active faces (nonconvex). */
/*     7.  Nonsmooth generalization of Brown function 2 (nonconvex). */
/*     8.  Chained Mifflin 2 (nonconvex). */
/*     9.  Chained crescent (nonconvex). */
/*     10. Chained crescent 2 (nonconvex). */


/*     Napsu Haarala (2007) */

/* Subroutine */ int bounds_(integer *n, integer *ib, doublereal *xl, 
	doublereal *xu, integer *next)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    double sqrt(doublereal);

    /* Local variables */
    static integer i__;

    /* Fortran I/O blocks */
    static cilist io___4 = { 0, 6, 0, 0, 0 };


/*     Scalar Arguments */
/*     Array Arguments */
/*     Local Arguments */
    /* Parameter adjustments */
    --xu;
    --xl;
    --ib;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ib[i__] = 0;
	xl[i__] = 0.;
	xu[i__] = 0.;
    }
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L10;
	case 3:  goto L30;
	case 4:  goto L20;
	case 5:  goto L20;
	case 6:  goto L10;
	case 7:  goto L10;
	case 8:  goto L30;
	case 9:  goto L10;
	case 10:  goto L10;
    }
    s_wsle(&io___4);
    do_lio(&c__9, &c__1, "Error: Not such a problem.", (ftnlen)26);
    e_wsle();
    *next = -1;
    return 0;
L10:
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	ib[i__] = 2;
	xl[i__] = .1;
	xu[i__] = 1.1;
/* L11: */
    }
    return 0;
L20:
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	ib[i__] = 2;
	xl[i__] = 1.1;
	xu[i__] = 2.1;
/* L21: */
    }
    return 0;
L30:
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	ib[i__] = 2;
	xl[i__] = 1. / sqrt(2.) + .1;
	xu[i__] = 1. / sqrt(2.) + 1.1;
/* L31: */
    }
    if (*next == 8) {
	xl[2] = .68;
	xu[2] = 1.68;
	xl[*n] = .1;
	xu[*n] = 1.1;
    }
    return 0;
} /* bounds_ */

/* *********************************************************************** */

/*     * SUBROUTINE FEASIX * */


/*     * Purpose * */

/*     Projection of the initial X to the feasible region. */


/*     * Calling sequence * */

/*     CALL FEASIX(N,X,XL,XU,IB,ITYPE) */


/*     * Parameters * */


/*     II  N               Number of variables. */
/*     RU  X(N)            Vector of variables. */
/*     RI  XL(N)           Lower bounds for variables. */
/*     RI  XU(N)           Upper bounds for variables. */
/*     II  IB(N)           Type of bound constraints: */
/*                             0  - X(I) is unbounded, */
/*                             1  - X(I) has only a lower bound, */
/*                             2  - X(I) has both lower and upper bounds, */
/*                             3  - X(I) has only an upper bound. */
/*     II  ITYPE           Type of starting point needed: */
/*                             0  - feasible, */
/*                             1  - strictly feasible. */


/*     Napsu Karmitsa (2007) */

/* Subroutine */ int feasix_(integer *n, doublereal *x, doublereal *xl, 
	doublereal *xu, integer *ib, integer *itype)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;
    static doublereal feas;

/*     Scalar Arguments */
/*     Array Arguments */
/*     Local Scalars */
    /* Parameter adjustments */
    --ib;
    --xu;
    --xl;
    --x;

    /* Function Body */
    feas = 1e-4;
/*     Project the initial X to the feasible set. */
    if (*itype == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (ib[i__] > 0) {
		if (ib[i__] <= 2) {
		    if (x[i__] < xl[i__]) {
			x[i__] = xl[i__];
		    }
		}
		if (ib[i__] >= 2) {
		    if (x[i__] > xu[i__]) {
			x[i__] = xu[i__];
		    }
		}
	    }
/* L10: */
	}
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (ib[i__] > 0) {
		if (ib[i__] <= 2) {
		    if (x[i__] <= xl[i__]) {
			x[i__] = xl[i__] + feas;
		    }
		}
		if (ib[i__] >= 2) {
		    if (x[i__] >= xu[i__]) {
			x[i__] = xu[i__] - feas;
		    }
		}
	    }
/* L20: */
	}
    }
    return 0;
} /* feasix_ */

/* *********************************************************************** */

/*     * SUBROUTINE FUNC * */


/*     * Purpose * */


/*     Computation of the value and the subgradient of the objective */
/*     function. */


/*     * Calling sequence * */

/*     CALL FUNC(N,X,F,G,NEXT) */


/*     * Parameters * */

/*     II  N          Number of variables. */
/*     RI  X(N)       Vector of variables. */
/*     RI  NEXT       Problem number. */
/*     RO  F          Value of the objective function. */
/*     RO  G(N)       Subgradient of the objective function. */


/*     * Problems * */

/*     1.  Generalization of MAXQ (convex). */
/*     2.  Generalization of MXHILB (convex). */
/*     3.  Chained LQ (convex). */
/*     4.  Chained CB3 (convex). */
/*     5.  Chained CB3 2 (convex). */
/*     6.  Number of active faces (nonconvex). */
/*     7.  Nonsmooth generalization of Brown function 2 (nonconvex). */
/*     8.  Chained Mifflin 2 (nonconvex). */
/*     9.  Chained crescent (nonconvex). */
/*     10. Chained crescent 2 (nonconvex). */


/*     Napsu Haarala (2003) */

/*     Haarala M., Miettinen K. and Mäkelä M.M.: New Limited Memory */
/*     Bundle Method for Large-Scale Nonsmooth Optimization, Optimization */
/*     Methods and Software, Vol. 19, No. 6, 2004, 673-692.. */


/* Subroutine */ int funcb_(integer *n, doublereal *x, doublereal *f, 
	doublereal *g, integer *next)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    double d_sign(doublereal *, doublereal *), exp(doublereal), log(
	    doublereal), pow_dd(doublereal *, doublereal *);

    /* Local variables */
    static doublereal a, b, c__, d__;
    static integer i__, j;
    static doublereal p, q, y;
    static integer hit;
    static doublereal temp2, temp3;

    /* Fortran I/O blocks */
    static cilist io___7 = { 0, 6, 0, 0, 0 };


/*     Scalar Arguments */
/*     Array Arguments */
/*     Local Arguments */
/*     Intrinsic Functions */
    /* Parameter adjustments */
    --g;
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L40;
	case 5:  goto L50;
	case 6:  goto L60;
	case 7:  goto L70;
	case 8:  goto L80;
	case 9:  goto L90;
	case 10:  goto L100;
    }
    s_wsle(&io___7);
    do_lio(&c__9, &c__1, "Error: Not such a problem.", (ftnlen)26);
    e_wsle();
    *next = -1;
    return 0;

/*     Generalization of MAXQ (convex) */

L10:
    *f = x[1] * x[1];
    g[1] = 0.;
    hit = 1;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	y = x[i__] * x[i__];
	if (y > *f) {
	    *f = y;
	    hit = i__;
	}
	g[i__] = 0.;
/* L11: */
    }
    g[hit] = x[hit] * 2;
    return 0;

/*     Generalization of MXHILB (convex) */

L20:
    *f = 0.;
    hit = 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	*f += x[j] / (doublereal) j;
/* L21: */
    }
    g[1] = d_sign(&c_b51, f);
    *f = abs(*f);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	temp2 = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    temp2 += x[j] / (doublereal) (i__ + j - 1);
/* L23: */
	}
	g[i__] = d_sign(&c_b51, &temp2);
	temp2 = abs(temp2);
	if (temp2 > *f) {
	    *f = temp2;
	    hit = i__;
	}
/* L22: */
    }
    temp3 = g[hit];
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	g[j] = temp3 / (doublereal) (hit + j - 1);
/* L24: */
    }
    return 0;

/*     Chained LQ (convex) */

L30:
    *f = 0.;
    g[1] = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__ + 1] = 0.;
	a = -x[i__] - x[i__ + 1];
	b = -x[i__] - x[i__ + 1] + (x[i__] * x[i__] + x[i__ + 1] * x[i__ + 1] 
		- 1.);
	if (a >= b) {
	    *f += a;
	    g[i__] += -1.;
	    g[i__ + 1] = -1.;
	} else {
	    *f += b;
	    g[i__] = g[i__] - 1. + x[i__] * 2.;
	    g[i__ + 1] = x[i__ + 1] * 2. - 1.;
	}
/* L31: */
    }
    return 0;

/*     Chained CB3 (convex) */

L40:
    *f = 0.;
    g[1] = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__ + 1] = 0.;
	a = x[i__] * x[i__] * x[i__] * x[i__] + x[i__ + 1] * x[i__ + 1];
	b = (2. - x[i__]) * (2. - x[i__]) + (2. - x[i__ + 1]) * (2. - x[i__ + 
		1]);
	c__ = exp(-x[i__] + x[i__ + 1]) * 2.;
	y = max(a,b);
	y = max(y,c__);
	if (y == a) {
	    g[i__] += x[i__] * 4. * x[i__] * x[i__];
	    g[i__ + 1] = x[i__ + 1] * 2.;
	} else if (y == b) {
	    g[i__] = g[i__] + x[i__] * 2. - 4.;
	    g[i__ + 1] = x[i__ + 1] * 2. - 4.;
	} else {
	    g[i__] -= c__;
	    g[i__ + 1] = c__;
	}
	*f += y;
/* L41: */
    }
    return 0;

/*     Chained CB3 2 (convex) */

L50:
    *f = 0.;
    g[1] = 0.;
    a = 0.;
    b = 0.;
    c__ = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__ + 1] = 0.;
	a = a + x[i__] * x[i__] * x[i__] * x[i__] + x[i__ + 1] * x[i__ + 1];
	b = b + (2. - x[i__]) * (2. - x[i__]) + (2. - x[i__ + 1]) * (2. - x[
		i__ + 1]);
	c__ += exp(-x[i__] + x[i__ + 1]) * 2.;
/* L51: */
    }
    *f = max(a,b);
    *f = max(*f,c__);
    if (*f == a) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] += x[i__] * 4. * x[i__] * x[i__];
	    g[i__ + 1] = x[i__ + 1] * 2.;
/* L53: */
	}
    } else if (*f == b) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] = g[i__] + x[i__] * 2. - 4.;
	    g[i__ + 1] = x[i__ + 1] * 2. - 4.;
/* L54: */
	}
    } else {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] -= exp(-x[i__] + x[i__ + 1]) * 2.;
	    g[i__ + 1] = exp(-x[i__] + x[i__ + 1]) * 2.;
/* L55: */
	}
    }
    return 0;

/*     Number of active faces (nonconvex) */

L60:
    temp3 = 1.;
    y = -x[1];
    g[1] = 0.;
    *f = log(abs(x[1]) + 1.);
    hit = 1;
    temp2 = *f;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	y -= x[i__];
	g[i__] = 0.;
/* Computing MAX */
	d__2 = *f, d__3 = log((d__1 = x[i__], abs(d__1)) + 1.);
	*f = max(d__2,d__3);
	if (*f > temp2) {
	    hit = i__;
	    temp2 = *f;
	}
/* L62: */
    }
/* Computing MAX */
    d__1 = *f, d__2 = log(abs(y) + 1.);
    *f = max(d__1,d__2);
    if (*f > temp2) {
	if (y >= 0.) {
	    temp3 = -1.;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] = temp3 * (1. / (abs(y) + 1.));
/* L63: */
	}
    } else {
	if (x[hit] < 0.) {
	    temp3 = -1.;
	}
	g[hit] = temp3 * (1. / ((d__1 = x[hit], abs(d__1)) + 1.));
    }
    return 0;

/*     Nonsmooth generalization of Brown function 2 (nonconvex) */

L70:
    *f = 0.;
    g[1] = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	a = (d__1 = x[i__], abs(d__1));
	b = (d__1 = x[i__ + 1], abs(d__1));
	c__ = x[i__] * x[i__] + 1.;
	d__ = x[i__ + 1] * x[i__ + 1] + 1.;
	*f = *f + pow_dd(&b, &c__) + pow_dd(&a, &d__);
	p = 0.;
	q = 0.;
	if (x[i__] < 0.) {
	    if (b > p) {
		p = log(b);
	    }
	    d__1 = d__ - 1.;
	    g[i__] = g[i__] - d__ * pow_dd(&a, &d__1) + x[i__] * 2. * p * 
		    pow_dd(&b, &c__);
	} else {
	    if (b > p) {
		p = log(b);
	    }
	    d__1 = d__ - 1.;
	    g[i__] = g[i__] + d__ * pow_dd(&a, &d__1) + x[i__] * 2. * p * 
		    pow_dd(&b, &c__);
	}
	if (x[i__ + 1] == 0.) {
	    g[i__ + 1] = 0.;
	} else if (x[i__ + 1] < 0.) {
	    if (a > q) {
		q = log(a);
	    }
	    d__1 = c__ - 1.;
	    g[i__ + 1] = -c__ * pow_dd(&b, &d__1) + x[i__ + 1] * 2. * q * 
		    pow_dd(&a, &d__);
	} else {
	    if (a > q) {
		q = log(a);
	    }
	    d__1 = c__ - 1.;
	    g[i__ + 1] = c__ * pow_dd(&b, &d__1) + x[i__ + 1] * 2. * q * 
		    pow_dd(&a, &d__);
	}
/* L71: */
    }
    return 0;

/*     Chained mifflin 2 (nonconvex) */

L80:
    *f = 0.;
    g[1] = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	y = x[i__] * x[i__] + x[i__ + 1] * x[i__ + 1] - 1.;
	*f = *f - x[i__] + y * 2. + abs(y) * 1.75;
	y = d_sign(&c_b66, &y) + 4.;
	g[i__] = g[i__] + y * x[i__] - 1.;
	g[i__ + 1] = y * x[i__ + 1];
/* L81: */
    }
    return 0;

/*     Chained crescent (nonconvex) */

L90:
    temp2 = 0.;
    temp3 = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp2 = temp2 + x[i__] * x[i__] + (x[i__ + 1] - 1.) * (x[i__ + 1] - 
		1.) + x[i__ + 1] - 1.;
	temp3 = temp3 - x[i__] * x[i__] - (x[i__ + 1] - 1.) * (x[i__ + 1] - 
		1.) + x[i__ + 1] + 1.;
/* L91: */
    }
    *f = max(temp2,temp3);
    g[1] = 0.;
    if (temp2 >= temp3) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] += x[i__] * 2.;
	    g[i__ + 1] = (x[i__ + 1] - 1.) * 2. + 1.;
/* L92: */
	}
    } else {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    g[i__] -= x[i__] * 2.;
	    g[i__ + 1] = (x[i__ + 1] - 1.) * -2. + 1.;
/* L93: */
	}
    }
    return 0;

/*     Chained crescent 2 (nonconvex) */

L100:
    *f = 0.;
    g[1] = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp2 = x[i__] * x[i__] + (x[i__ + 1] - 1.) * (x[i__ + 1] - 1.) + x[
		i__ + 1] - 1.;
	temp3 = -x[i__] * x[i__] - (x[i__ + 1] - 1.) * (x[i__ + 1] - 1.) + x[
		i__ + 1] + 1.;
	if (temp2 >= temp3) {
	    *f += temp2;
	    g[i__] += x[i__] * 2.;
	    g[i__ + 1] = (x[i__ + 1] - 1.) * 2. + 1.;
	} else {
	    *f += temp3;
	    g[i__] -= x[i__] * 2.;
	    g[i__ + 1] = (x[i__ + 1] - 1.) * -2. + 1.;
	}
/* L101: */
    }
    return 0;
} /* funcb_ */

