/* lvuns.f -- translated by f2c (version 20100827).
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

/* Common Block Declarations */

struct {
    doublereal y[2700];
} empr19_;

#define empr19_1 empr19_

/* Table of constant values */

static integer c__5 = 5;
static integer c__1 = 1;
static doublereal c_b147 = 2.;
static doublereal c_b148 = 3.5;
static doublereal c_b149 = 16.;
static doublereal c_b164 = 1.;
static doublereal c_b198 = 6.;
static doublereal c_b232 = -1.5;
static doublereal c_b262 = 1.5;
static doublereal c_b272 = 12.;

/* SUBROUTINE TIUD19                ALL SYSTEMS                99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 VL : ORIGINAL VERSION */

/* PURPOSE : */
/*  INITIATION OF VARIABLES FOR NONSMOOTH OPTIMIZATION. */
/*  UNCONSTRAINED DENSE VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RO  X(N)  VECTOR OF VARIABLES. */
/*  RO  FMIN  LOWER BOUND FOR VALUE OF THE OBJECTIVE FUNCTION. */
/*  RO  XMAX  MAXIMUM STEPSIZE. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */
/*  IO  IERR  ERROR INDICATOR. */

/* Subroutine */ int tiud19_(integer *n, doublereal *x, doublereal *fmin, 
	doublereal *xmax, integer *next, integer *ierr)
{
    /* Initialized data */

    static shortint aa[59] = { 0,0,0,0,0,2,1,1,1,3,1,2,1,1,2,1,4,1,2,2,3,2,1,
	    0,1,0,2,1,0,1,1,1,1,1,1,1,0,1,2,1,0,0,2,1,0,1,1,2,0,0,1,5,10,2,4,
	    3,6,6,6 };
    static shortint pp[23] = { 0,2,3,4,5,6,2,3,-1,2,2,2,2,1,1,5,1,1,1,1,2,3,2 
	    };
    static shortint cc[95] = { -16,0,0,0,0,2,-1,-1,1,1,2,-2,0,-2,-9,0,-1,-2,2,
	    1,0,0,2,0,-2,-4,-1,-3,3,1,1,4,0,-4,1,0,-1,-2,4,1,0,2,0,-1,0,0,-1,
	    -1,5,1,-40,-2,0,-4,-4,-1,-40,-60,5,1,30,-20,-10,32,-10,-20,39,-6,
	    -31,32,-10,-6,10,-6,-10,32,-31,-6,39,-20,-10,32,-10,-20,30,4,8,10,
	    6,2,-15,-27,-36,-18,-12 };

    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublereal d__1;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    double exp(doublereal), cos(doublereal), sin(doublereal);
    integer f_open(olist *), s_rsle(cilist *), do_lio(integer *, integer *, 
	    char *, ftnlen), e_rsle(void), f_clos(cllist *);

    /* Local variables */
    static integer i__, j, k, l;
    static doublereal ai, aj, ak;
    static integer kk;

    /* Fortran I/O blocks */
    static cilist io___12 = { 0, 50, 0, 0, 0 };


    /* Parameter adjustments */
    --x;

    /* Function Body */
    *fmin = 0.;
    *xmax = 1e3;
    *ierr = 0;
    if (*n < 2) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L1: */
    }
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
	case 10:  goto L180;
	case 11:  goto L100;
	case 12:  goto L110;
	case 13:  goto L230;
	case 14:  goto L240;
	case 15:  goto L170;
	case 16:  goto L120;
	case 17:  goto L210;
	case 18:  goto L220;
	case 19:  goto L130;
	case 20:  goto L130;
	case 21:  goto L150;
	case 22:  goto L160;
	case 23:  goto L190;
	case 24:  goto L190;
	case 25:  goto L250;
    }
L10:
    *n = 2;
    x[1] = -1.2;
    x[2] = 1.;
    return 0;
L20:
    *n = 2;
    x[1] = -1.5;
    x[2] = 2.;
    return 0;
L30:
    *n = 2;
    x[1] = 1.;
    x[2] = -.1;
    return 0;
L40:
    *n = 2;
    x[1] = 2.;
    x[2] = 2.;
    return 0;
L50:
    *n = 2;
L51:
    *fmin = -1e60;
L52:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L53: */
    }
    return 0;
L60:
    *n = 2;
    x[1] = -1.;
    x[2] = 5.;
    return 0;
L70:
    *n = 2;
    x[1] = -.5;
    x[2] = -.5;
L777:
    *fmin = -1e60;
    return 0;
L80:
    *n = 2;
    x[1] = .8;
    x[2] = .6;
    goto L777;
L90:
    *n = 2;
    x[1] = -1.;
    x[2] = -1.;
    goto L777;
L180:
    *n = 2;
    x[1] = 3.;
    x[2] = 2.;
    goto L777;
L100:
    if (*n < 4) {
	goto L999;
    }
    *n = 4;
    goto L777;
L110:
    if (*n < 5) {
	goto L999;
    }
    *n = 5;
    x[5] = 1.;
    for (i__ = 1; i__ <= 59; ++i__) {
	empr19_1.y[i__ - 1] = (doublereal) aa[i__ - 1];
/* L111: */
    }
    empr19_1.y[56] = 1.7;
    empr19_1.y[57] = 2.5;
    empr19_1.y[59] = 3.5;
    return 0;
L230:
    if (*n < 5) {
	goto L999;
    }
    *n = 5;
    x[5] = 1.;
    *fmin = -1e60;
L232:
    for (i__ = 1; i__ <= 95; ++i__) {
	empr19_1.y[i__ - 1] = (doublereal) cc[i__ - 1];
/* L233: */
    }
    empr19_1.y[2] = -3.5;
    empr19_1.y[44] = -2.8;
    empr19_1.y[52] = -.25;
    return 0;
L240:
    if (*n < 5) {
	goto L999;
    }
    *n = 5;
    x[1] = -2.;
    x[2] = 1.5;
    x[3] = 2.;
    x[4] = -1.;
    x[5] = -1.;
    *fmin = -1e60;
    *xmax = 1.;
    return 0;
L170:
    if (*n < 6) {
	goto L999;
    }
    *n = 6;
    *xmax = 2.;
    x[1] = 2.;
    x[2] = 2.;
    x[3] = 7.;
    x[5] = -2.;
    x[6] = 1.;
    return 0;
L120:
    if (*n < 10) {
	goto L999;
    }
    *n = 10;
    kk = 0;
    for (k = 1; k <= 5; ++k) {
	ak = (doublereal) k;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ai = (doublereal) i__;
	    i__2 = *n;
	    for (j = i__; j <= i__2; ++j) {
		aj = (doublereal) j;
		empr19_1.y[kk + (i__ - 1) * *n + j - 1] = exp(ai / aj) * cos(
			ai * aj) * sin(ak);
		empr19_1.y[kk + (j - 1) * *n + i__ - 1] = empr19_1.y[kk + (
			i__ - 1) * *n + j - 1];
/* L124: */
	    }
/* L123: */
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ai = (doublereal) i__;
	    empr19_1.y[kk + 100 + i__ - 1] = exp(ai / ak) * sin(ai * ak);
	    l = kk + (i__ - 1) * *n + i__;
	    empr19_1.y[l - 1] = (d__1 = sin(ak), abs(d__1)) * ai / (
		    doublereal) (*n);
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		if (j != i__) {
		    empr19_1.y[l - 1] += (d__1 = empr19_1.y[kk + (i__ - 1) * *
			    n + j - 1], abs(d__1));
		}
/* L126: */
	    }
	}
	kk += 110;
/* L122: */
    }
    goto L51;
L210:
    if (*n < 10) {
	goto L999;
    }
    *n = 10;
    *xmax = 10.;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	x[i__] = -.1;
/* L211: */
    }
    return 0;
L220:
    if (*n < 12) {
	goto L999;
    }
    *n = 12;
    for (i__ = 1; i__ <= 23; ++i__) {
	empr19_1.y[i__ - 1] = (doublereal) pp[i__ - 1];
/* L221: */
    }
    empr19_1.y[9] = -.5;
    x[1] = .66666666666666663;
    x[7] = 1.6666666666666667;
    for (i__ = 2; i__ <= 5; ++i__) {
	x[i__] = (x[i__ - 1] + empr19_1.y[i__ - 1] + empr19_1.y[i__]) / 3.;
	x[i__ + 6] = (x[i__ + 5] + empr19_1.y[i__ + 5] + empr19_1.y[i__ + 6]) 
		/ 3.;
/* L222: */
    }
    x[6] = (x[5] + 11.5) / 3.;
    x[12] = (x[11] + 1.) / 3.;
    return 0;
L130:
    if (*n < 20) {
	goto L999;
    }
    *n = 20;
    if (*next == 19) {
	*xmax = 5.;
    }
    if (*next == 20) {
	*xmax = 10.;
    }
    for (i__ = 1; i__ <= 10; ++i__) {
	x[i__] = (doublereal) i__;
	x[i__ + 10] = (doublereal) (-i__ - 10);
/* L131: */
    }
    return 0;
L150:
    if (*n < 48) {
	goto L999;
    }
    *n = 48;
    o__1.oerr = 0;
    o__1.ounit = 50;
    o__1.ofnmlen = 10;
    o__1.ofnm = "TEST19.DAT";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_rsle(&io___12);
    i__2 = *n;
    for (i__ = 2; i__ <= i__2; ++i__) {
	i__1 = *n;
	for (j = i__; j <= i__1; ++j) {
	    do_lio(&c__5, &c__1, (char *)&empr19_1.y[*n * (i__ - 2) + j - 1], 
		    (ftnlen)sizeof(doublereal));
	}
    }
    i__3 = *n;
    for (i__ = 1; i__ <= i__3; ++i__) {
	do_lio(&c__5, &c__1, (char *)&empr19_1.y[*n * *n + i__ - 1], (ftnlen)
		sizeof(doublereal));
    }
    i__4 = *n;
    for (i__ = 1; i__ <= i__4; ++i__) {
	do_lio(&c__5, &c__1, (char *)&empr19_1.y[*n * *n + *n + i__ - 1], (
		ftnlen)sizeof(doublereal));
    }
    e_rsle();
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	empr19_1.y[*n * (i__ - 1) + i__ - 1] = 1e5;
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    empr19_1.y[*n * (i__ - 1) + j - 1] = empr19_1.y[*n * (j - 1) + 
		    i__ - 1];
/* L151: */
	}
    }
    cl__1.cerr = 0;
    cl__1.cunit = 50;
    cl__1.csta = 0;
    f_clos(&cl__1);
    goto L777;
L160:
    if (*n < 50) {
	goto L999;
    }
    *n = 50;
    *xmax = 10.;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	x[i__] = (doublereal) i__ - 25.5;
/* L161: */
    }
    return 0;
L190:
    if (*n < 30) {
	goto L999;
    }
    if (*n >= 50) {
	*n = 50;
    }
    if (*n < 50) {
	*n = 30;
    }
    if (*next == 23) {
	*xmax = 5.;
    }
    goto L52;
L250:
    if (*n < 15) {
	goto L999;
    }
    *n = 15;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	x[i__] = 1e-4;
/* L251: */
    }
    x[12] = 60.;
    goto L232;
L999:
    *ierr = 1;
    return 0;
} /* tiud19_ */

/* SUBROUTINE TFFU19                ALL SYSTEMS                99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 RA : ORIGINAL VERSION */

/* PURPOSE : */
/*  VALUE OF THE NONSMOOTH OBJECTIVE FUNCTION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  F  VALUE OF THE OBJECTIVE FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tffu19_(integer *n, doublereal *x, doublereal *f, 
	integer *next)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */
    double exp(doublereal), sqrt(doublereal), sin(doublereal), cos(doublereal)
	    , pow_di(doublereal *, integer *);

    /* Local variables */
    static integer i__, j, k;
    static doublereal t, z__, f1, f2, f3, f4, ai, aj;
    static integer kk, ll;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    *f = 0.;
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
	case 10:  goto L180;
	case 11:  goto L100;
	case 12:  goto L110;
	case 13:  goto L230;
	case 14:  goto L240;
	case 15:  goto L170;
	case 16:  goto L120;
	case 17:  goto L210;
	case 18:  goto L220;
	case 19:  goto L130;
	case 20:  goto L140;
	case 21:  goto L150;
	case 22:  goto L160;
	case 23:  goto L190;
	case 24:  goto L200;
	case 25:  goto L250;
    }
L10:
/* Computing 2nd power */
    d__2 = x[1];
/* Computing 2nd power */
    d__1 = x[2] - d__2 * d__2;
/* Computing 2nd power */
    d__3 = 1. - x[1];
    *f = d__1 * d__1 * 100. + d__3 * d__3;
    return 0;
L20:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - 1.;
    t = d__1 * d__1 + d__2 * d__2 - 1.;
    *f = x[2] + abs(t);
    return 0;
L30:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 4th power */
    d__2 = x[2], d__2 *= d__2;
    f1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = 2. - x[1];
/* Computing 2nd power */
    d__2 = 2. - x[2];
    f2 = d__1 * d__1 + d__2 * d__2;
    f3 = exp(-x[1] + x[2]) * 2.;
/* Computing MAX */
    d__1 = max(f1,f2);
    *f = max(d__1,f3);
    return 0;
L40:
/* Computing 4th power */
    d__1 = x[1], d__1 *= d__1;
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = 2. - x[1];
/* Computing 2nd power */
    d__2 = 2. - x[2];
    f2 = d__1 * d__1 + d__2 * d__2;
    f3 = exp(-x[1] + x[2]) * 2.;
/* Computing MAX */
    d__1 = max(f1,f2);
    *f = max(d__1,f3);
    return 0;
L50:
    f1 = x[1] * 5. + x[2];
    f2 = x[1] * -5. + x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f3 = d__1 * d__1 + d__2 * d__2 + x[2] * 4.;
/* Computing MAX */
    d__1 = max(f1,f2);
    *f = max(d__1,f3);
    return 0;
L60:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2;
    f2 = f1 + (x[1] * -4. - x[2] + 4.) * 10.;
    f3 = f1 + (-x[1] - x[2] * 2. + 6.) * 10.;
/* Computing MAX */
    d__1 = max(f1,f2);
    *f = max(d__1,f3);
    return 0;
L70:
    f1 = -x[1] - x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f2 = f1 + (d__1 * d__1 + d__2 * d__2 - 1.);
    *f = max(f1,f2);
    return 0;
L80:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2 - 1.;
    f2 = 0.;
    *f = max(f1,f2);
    *f = *f * 20. - x[1];
    return 0;
L90:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2 - 1.;
    if (f1 < 0.) {
	f1 = -f1;
    }
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    *f = -x[1] + (d__1 * d__1 + d__2 * d__2 - 1.) * 2. + f1 * 1.75;
    return 0;
L180:
    if (x[1] > abs(x[2])) {
/* Computing 2nd power */
	d__1 = x[1];
/* Computing 2nd power */
	d__2 = x[2];
	*f = sqrt(d__1 * d__1 * 9. + d__2 * d__2 * 16.) * 5.;
    } else if (x[1] > 0. && x[1] <= abs(x[2])) {
	*f = x[1] * 9. + abs(x[2]) * 16.;
    } else if (x[1] <= 0.) {
/* Computing 9th power */
	d__1 = x[1], d__2 = d__1, d__1 *= d__1, d__1 *= d__1;
	*f = x[1] * 9. + abs(x[2]) * 16. - d__2 * (d__1 * d__1);
    }
    return 0;
L100:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
    *f = d__1 * d__1 + d__2 * d__2 + d__3 * d__3;
/* Computing 2nd power */
    d__1 = x[3];
/* Computing 2nd power */
    d__2 = x[4];
    f1 = *f + d__1 * d__1 + d__2 * d__2 - (x[1] + x[2]) * 5. - x[3] * 21. + x[
	    4] * 7.;
/* Computing 2nd power */
    d__1 = x[4];
    f2 = *f + d__1 * d__1 + x[1] - x[2] + x[3] - x[4] - 8.;
/* Computing 2nd power */
    d__1 = x[2];
/* Computing 2nd power */
    d__2 = x[4];
    f3 = *f + d__1 * d__1 + d__2 * d__2 * 2. - x[1] - x[4] - 10.;
    f4 = *f + x[1] * 2. - x[2] - x[4] - 5.;
/* Computing MAX */
    d__1 = max(0.,f2), d__1 = max(d__1,f3);
    *f = f1 + max(d__1,f4) * 10.;
    return 0;
L110:
    *f = 0.;
    for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	d__1 = x[j] - empr19_1.y[j - 1];
	*f += d__1 * d__1;
/* L111: */
    }
    *f *= empr19_1.y[50];
    for (i__ = 2; i__ <= 10; ++i__) {
	f1 = 0.;
	for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	    d__1 = x[j] - empr19_1.y[(i__ - 1) * 5 + j - 1];
	    f1 += d__1 * d__1;
/* L113: */
	}
	f1 *= empr19_1.y[i__ + 49];
	*f = max(*f,f1);
/* L112: */
    }
    return 0;
L230:
    *f = 0.;
    for (i__ = 1; i__ <= 10; ++i__) {
	t = empr19_1.y[i__ + 49];
	for (j = 1; j <= 5; ++j) {
	    t -= empr19_1.y[i__ + j * 10 - 11] * x[j];
/* L232: */
	}
	*f = max(*f,t);
/* L231: */
    }
    *f *= 50.;
    for (j = 1; j <= 5; ++j) {
	t = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    t += empr19_1.y[i__ + 55 + j * 5 - 1] * x[i__];
/* L233: */
	}
/* Computing 3rd power */
	d__1 = x[j];
	*f = *f + empr19_1.y[j + 84] * (d__1 * (d__1 * d__1)) + empr19_1.y[j 
		+ 89] * x[j] + t * x[j];
/* L234: */
    }
    return 0;
L240:
    f1 = x[1] * x[2] * x[3] * x[4] * x[5];
/* Computing 2nd power */
    d__2 = x[1];
/* Computing 2nd power */
    d__3 = x[2];
/* Computing 2nd power */
    d__4 = x[3];
/* Computing 2nd power */
    d__5 = x[4];
/* Computing 2nd power */
    d__6 = x[5];
    f2 = (d__1 = d__2 * d__2 + d__3 * d__3 + d__4 * d__4 + d__5 * d__5 + d__6 
	    * d__6 - 10., abs(d__1));
    f2 += (d__1 = x[2] * x[3] - x[4] * 5. * x[5], abs(d__1));
/* Computing 3rd power */
    d__2 = x[1];
/* Computing 3rd power */
    d__3 = x[2];
    f2 += (d__1 = d__2 * (d__2 * d__2) + d__3 * (d__3 * d__3) + 1., abs(d__1))
	    ;
    *f = f1 + f2 * 10.;
    return 0;
L170:
    *f = 0.;
    for (i__ = 1; i__ <= 51; ++i__) {
	t = (doublereal) (i__ - 1) / 10.;
	z__ = exp(-t) * .5 - exp(-t * 2.) + exp(-t * 3.) * .5 + exp(-t * 1.5) 
		* 1.5 * sin(t * 7.) + exp(-t * 2.5) * sin(t * 5.);
	f1 = exp(-x[2] * t);
	f2 = exp(-x[6] * t);
	f3 = cos(x[3] * t + x[4]);
	*f += (d__1 = x[1] * f1 * f3 + x[5] * f2 - z__, abs(d__1));
/* L171: */
    }
    return 0;
L120:
    *f = 0.;
    k = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	*f -= empr19_1.y[i__ + 99] * x[i__];
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    *f += empr19_1.y[(i__ - 1) * *n + j - 1] * x[i__] * x[j];
/* L122: */
	}
/* L121: */
    }
    kk = 110;
    for (k = 2; k <= 5; ++k) {
	f1 = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    f1 -= empr19_1.y[kk + 100 + i__ - 1] * x[i__];
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		f1 += empr19_1.y[kk + (i__ - 1) * *n + j - 1] * x[i__] * x[j];
/* L125: */
	    }
/* L124: */
	}
	*f = max(*f,f1);
	kk += 110;
/* L123: */
    }
    return 0;
L210:
    f1 = 0.;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 += d__1 * d__1;
/* L211: */
    }
    f4 = f1 - .25;
    f1 = f4 * .001 * f4;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__] - 1.;
	f1 += d__1 * d__1;
/* L212: */
    }
    f2 = 0.;
    for (i__ = 2; i__ <= 30; ++i__) {
	ai = (doublereal) (i__ - 1) / 29.;
	*f = 0.;
	for (j = 1; j <= 10; ++j) {
	    i__1 = j - 1;
	    *f += x[j] * pow_di(&ai, &i__1);
/* L213: */
	}
	*f = -(*f) * *f - 1.;
	for (j = 2; j <= 10; ++j) {
	    aj = (doublereal) (j - 1);
	    i__1 = j - 2;
	    *f += x[j] * aj * pow_di(&ai, &i__1);
/* L214: */
	}
	f2 += *f * *f;
/* L215: */
    }
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__3 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - d__3 * d__3 - 1.;
    f2 = f2 + d__1 * d__1 + d__2 * d__2;
    f3 = 0.;
    for (i__ = 2; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__2 = x[i__ - 1];
/* Computing 2nd power */
	d__1 = x[i__] - d__2 * d__2;
/* Computing 2nd power */
	d__3 = 1. - x[i__];
	f3 = f3 + d__1 * d__1 * 100. + d__3 * d__3;
/* L216: */
    }
/* Computing MAX */
    d__1 = max(f1,f2);
    *f = max(d__1,f3);
    return 0;
L220:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[7];
/* Computing 2nd power */
    d__3 = 5.5 - x[6];
/* Computing 2nd power */
    d__4 = x[12] + 1.;
    *f = sqrt(d__1 * d__1 + d__2 * d__2) + sqrt(d__3 * d__3 + d__4 * d__4);
    for (j = 1; j <= 6; ++j) {
/* Computing 2nd power */
	d__1 = empr19_1.y[j - 1] - x[j];
/* Computing 2nd power */
	d__2 = empr19_1.y[j + 5] - x[j + 6];
	*f += empr19_1.y[j + 11] * sqrt(d__1 * d__1 + d__2 * d__2);
/* L221: */
    }
    for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	d__1 = x[j] - x[j + 1];
/* Computing 2nd power */
	d__2 = x[j + 6] - x[j + 7];
	*f += empr19_1.y[j + 17] * sqrt(d__1 * d__1 + d__2 * d__2);
/* L222: */
    }
    return 0;
L130:
/* Computing 2nd power */
    d__1 = x[1];
    *f = d__1 * d__1;
    for (i__ = 2; i__ <= 20; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 = d__1 * d__1;
	*f = max(*f,f1);
/* L131: */
    }
    return 0;
L140:
    *f = abs(x[1]);
    for (i__ = 2; i__ <= 20; ++i__) {
	f1 = (d__1 = x[i__], abs(d__1));
	*f = max(*f,f1);
/* L141: */
    }
    return 0;
L150:
    *n = 48;
    *f = 0.;
    kk = *n * *n;
    ll = *n * *n + *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	*f += empr19_1.y[ll + i__ - 1] * x[i__];
/* L151: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	z__ = 1e60;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = empr19_1.y[(i__ - 1) * *n + j - 1] - x[i__];
	    if (t >= z__) {
		goto L153;
	    }
	    z__ = t;
	    k = i__;
L153:
	    ;
	}
	*f += empr19_1.y[kk + j - 1] * z__;
/* L152: */
    }
    *f = -(*f);
    return 0;
L160:
    f1 = 0.;
    *f = -1e60;
    for (i__ = 1; i__ <= 50; ++i__) {
	f2 = x[i__];
	f1 += f2;
	*f = max(*f,f2);
/* L161: */
    }
    *f = *f * 50. - f1;
    return 0;
L190:
    *f = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	*f += x[j] / (doublereal) j;
/* L191: */
    }
    *f = abs(*f);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	f1 = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    f1 += x[j] / (doublereal) (i__ + j - 1);
/* L193: */
	}
	f1 = abs(f1);
	*f = max(f1,*f);
/* L192: */
    }
    return 0;
L200:
    *f = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	f1 = 0.;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    f1 += x[i__] / (doublereal) (i__ + j - 1);
/* L203: */
	}
	f1 = abs(f1);
	*f += f1;
/* L202: */
    }
    return 0;
L250:
    f1 = 0.;
    for (j = 1; j <= 5; ++j) {
/* Computing 3rd power */
	d__1 = x[j];
	f1 += empr19_1.y[j + 84] * (d__1 * (d__1 * d__1));
/* L251: */
    }
    *f = (d__1 = f1 + f1, abs(d__1));
    for (j = 1; j <= 5; ++j) {
	t = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    t += empr19_1.y[i__ + 55 + j * 5 - 1] * x[i__];
/* L252: */
	}
	*f += t * x[j];
/* L253: */
    }
    for (j = 6; j <= 15; ++j) {
	*f -= empr19_1.y[j + 44] * x[j];
/* L254: */
    }
    for (j = 1; j <= 5; ++j) {
	t = empr19_1.y[j + 84] * -3. * x[j] * x[j] - empr19_1.y[j + 89];
	for (i__ = 1; i__ <= 15; ++i__) {
	    if (i__ <= 5) {
		t -= empr19_1.y[i__ + 55 + j * 5 - 1] * 2. * x[i__];
	    }
	    if (i__ > 5) {
		t += empr19_1.y[i__ + j * 10 - 16] * x[i__];
	    }
/* L255: */
	}
	if (t > 0.) {
	    *f += t * 100.;
	}
/* L257: */
    }
    for (i__ = 1; i__ <= 15; ++i__) {
	if (x[i__] < 0.) {
	    *f -= x[i__] * 100.;
	}
/* L258: */
    }
    return 0;
} /* tffu19_ */

/* SUBROUTINE TFGU19                ALL SYSTEMS                99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 VL : ORIGINAL VERSION */

/* PURPOSE : */
/*  GRADIENT OF THE NONSMOOTH OBJECTIVE FUNCTION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  G(N)  GRADIENT OF THE OBJECTIVE FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tfgu19_(integer *n, doublereal *x, doublereal *g, 
	integer *next)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    double d_sign(doublereal *, doublereal *), exp(doublereal), sqrt(
	    doublereal), sin(doublereal), cos(doublereal), pow_di(doublereal *
	    , integer *);

    /* Local variables */
    static doublereal f;
    static integer i__, j, k, l;
    static doublereal t, f1, f2, f3, f4, ai, aj;
    static integer kk;

    /* Parameter adjustments */
    --g;
    --x;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__] = 0.;
/* L1: */
    }
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L30;
	case 5:  goto L50;
	case 6:  goto L60;
	case 7:  goto L70;
	case 8:  goto L80;
	case 9:  goto L90;
	case 10:  goto L180;
	case 11:  goto L100;
	case 12:  goto L110;
	case 13:  goto L230;
	case 14:  goto L240;
	case 15:  goto L170;
	case 16:  goto L120;
	case 17:  goto L210;
	case 18:  goto L220;
	case 19:  goto L130;
	case 20:  goto L140;
	case 21:  goto L150;
	case 22:  goto L160;
	case 23:  goto L190;
	case 24:  goto L200;
	case 25:  goto L250;
    }
L10:
/* Computing 2nd power */
    d__1 = x[1];
    g[2] = (x[2] - d__1 * d__1) * 200.;
    g[1] = x[1] * 2. * (1. - g[2]) - 2.;
    return 0;
L20:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - 1.;
    t = d__1 * d__1 + d__2 * d__2 - 1.;
    f = d_sign(&c_b147, &t);
    g[1] = f * x[1];
    g[2] = f * (x[2] - 1.) + 1.;
    return 0;
L30:
    i__ = *next - 2;
    j = 5 - *next;
/* Computing 2nd power */
    d__1 = x[i__];
/* Computing 4th power */
    d__2 = x[j], d__2 *= d__2;
    f1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = 2. - x[1];
/* Computing 2nd power */
    d__2 = 2. - x[2];
    f2 = d__1 * d__1 + d__2 * d__2;
    f3 = exp(-x[1] + x[2]) * 2.;
    if (f1 >= f2 && f1 >= f3) {
	g[i__] = x[i__] * 2.;
/* Computing 3rd power */
	d__1 = x[j];
	g[j] = d__1 * (d__1 * d__1) * 4.;
    } else if (f2 >= f1 && f2 >= f3) {
	g[1] = (2. - x[1]) * -2.;
	g[2] = (2. - x[2]) * -2.;
    } else {
	g[2] = exp(-x[1] + x[2]) * 2.;
	g[1] = -g[2];
    }
    return 0;
L50:
    f1 = x[1] * 5. + x[2];
    f2 = x[1] * -5. + x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f3 = d__1 * d__1 + d__2 * d__2 + x[2] * 4.;
    g[2] = 1.;
    if (f1 >= f2 && f1 >= f3) {
	g[1] = 5.;
    } else if (f2 >= f1 && f2 >= f3) {
	g[1] = -5.;
    } else {
	g[1] = x[1] * 2.;
	g[2] = x[2] * 2. + 4.;
    }
    return 0;
L60:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2;
    f2 = f1 + (x[1] * -4. - x[2] + 4.) * 10.;
    f3 = f1 + (-x[1] - x[2] * 2. + 6.) * 10.;
    g[1] = x[1] * 2.;
    g[2] = x[2] * 2.;
    if (f1 >= f2 && f1 >= f3) {
    } else if (f2 >= f1 && f2 >= f3) {
	g[1] += -40.;
	g[2] += -10.;
    } else {
	g[1] += -10.;
	g[2] += -20.;
    }
    return 0;
L70:
    f1 = -x[1] - x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f2 = f1 + (d__1 * d__1 + d__2 * d__2 - 1.);
    if (f1 >= f2) {
	g[1] = -1.;
	g[2] = -1.;
    } else {
	g[1] = x[1] * 2. - 1.;
	g[2] = x[2] * 2. - 1.;
    }
    return 0;
L80:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2 - 1.;
    g[1] = -1.;
    if (f1 >= 0.) {
	g[1] = x[1] * 40. - 1.;
	g[2] = x[2] * 40.;
    }
    return 0;
L90:
/* Computing 2nd power */
    d__2 = x[1];
/* Computing 2nd power */
    d__3 = x[2];
    d__1 = d__2 * d__2 + d__3 * d__3 - 1.;
    f1 = d_sign(&c_b148, &d__1) + 4.;
    g[1] = f1 * x[1] - 1.;
    g[2] = f1 * x[2];
    return 0;
L180:
    if (x[1] > abs(x[2])) {
/* Computing 2nd power */
	d__1 = x[1];
/* Computing 2nd power */
	d__2 = x[2];
	g[1] = x[1] * 45. / sqrt(d__1 * d__1 * 9. + d__2 * d__2 * 16.);
/* Computing 2nd power */
	d__1 = x[1];
/* Computing 2nd power */
	d__2 = x[2];
	g[2] = x[2] * 80. / sqrt(d__1 * d__1 * 9. + d__2 * d__2 * 16.);
    } else {
	g[1] = 9.;
	if (x[1] < 0.) {
/* Computing 8th power */
	    d__1 = x[1], d__1 *= d__1, d__1 *= d__1;
	    g[1] = 9. - d__1 * d__1 * 9.;
	}
	g[2] = d_sign(&c_b149, &x[2]);
    }
    return 0;
L100:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
    f = d__1 * d__1 + d__2 * d__2 + d__3 * d__3;
/* Computing 2nd power */
    d__1 = x[4];
    f2 = f + d__1 * d__1 + x[1] - x[2] + x[3] - x[4] - 8.;
/* Computing 2nd power */
    d__1 = x[2];
/* Computing 2nd power */
    d__2 = x[4];
    f3 = f + d__1 * d__1 + d__2 * d__2 * 2. - x[1] - x[4] - 10.;
    f4 = f + x[1] * 2. - x[2] - x[4] - 5.;
    l = 1;
    if (f2 > 0.) {
	l = 2;
    }
    if (f3 > max(f2,0.)) {
	l = 3;
    }
/* Computing MAX */
    d__1 = max(f2,f3);
    if (f4 > max(d__1,0.)) {
	l = 4;
    }
    switch (l) {
	case 1:  goto L101;
	case 2:  goto L102;
	case 3:  goto L103;
	case 4:  goto L104;
    }
L101:
    g[1] = x[1] * 2. - 5.;
    g[2] = x[2] * 2. - 5.;
    g[3] = x[3] * 4. - 21.;
    g[4] = x[4] * 2. + 7.;
    return 0;
L102:
    g[1] = x[1] * 22. + 5.;
    g[2] = x[2] * 22. - 15.;
    g[3] = x[3] * 24. - 11.;
    g[4] = x[4] * 22. - 3.;
    return 0;
L103:
    g[1] = x[1] * 22. - 15.;
    g[2] = x[2] * 42. - 5.;
    g[4] = x[4] * 42. - 3.;
    goto L105;
L104:
    g[1] = x[1] * 22. + 15.;
    g[2] = x[2] * 22. - 15.;
    g[4] = x[4] * 2. - 3.;
L105:
    g[3] = x[3] * 24. - 21.;
    return 0;
L110:
    f = -1e60;
    for (i__ = 1; i__ <= 10; ++i__) {
	f1 = 0.;
	for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	    d__1 = x[j] - empr19_1.y[(i__ - 1) * 5 + j - 1];
	    f1 += d__1 * d__1;
/* L113: */
	}
	f1 *= empr19_1.y[i__ + 49];
	if (f < f1) {
	    k = i__;
	}
	f = max(f,f1);
/* L112: */
    }
    for (j = 1; j <= 5; ++j) {
	g[j] = empr19_1.y[k + 49] * 2. * (x[j] - empr19_1.y[(k - 1) * 5 + j - 
		1]);
/* L114: */
    }
    return 0;
L230:
    f1 = 0.;
    for (i__ = 1; i__ <= 10; ++i__) {
	t = empr19_1.y[i__ + 49];
	for (j = 1; j <= 5; ++j) {
	    t -= empr19_1.y[i__ + j * 10 - 11] * x[j];
/* L232: */
	}
	if (t > f1) {
	    k = i__;
	}
	f1 = max(f1,t);
/* L231: */
    }
    for (j = 1; j <= 5; ++j) {
	t = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    t += empr19_1.y[i__ + 55 + j * 5 - 1] * x[i__];
/* L233: */
	}
	g[j] = empr19_1.y[j + 84] * 3. * x[j] * x[j] + t + t + empr19_1.y[j + 
		89];
	if (f1 > 0.) {
	    g[j] -= empr19_1.y[k + j * 10 - 11] * 50.;
	}
/* L234: */
    }
    return 0;
L240:
    g[1] = x[2] * x[3] * x[4] * x[5];
    g[2] = x[1] * x[3] * x[4] * x[5];
    g[3] = x[1] * x[2] * x[4] * x[5];
    g[4] = x[1] * x[2] * x[3] * x[5];
    g[5] = x[1] * x[2] * x[3] * x[4];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
/* Computing 2nd power */
    d__4 = x[4];
/* Computing 2nd power */
    d__5 = x[5];
    f1 = d__1 * d__1 + d__2 * d__2 + d__3 * d__3 + d__4 * d__4 + d__5 * d__5 
	    - 10.;
    f4 = 1.;
    if (f1 < 0.) {
	f4 = -f4;
    }
    for (i__ = 1; i__ <= 5; ++i__) {
	g[i__] += f4 * 20. * x[i__];
/* L241: */
    }
    f2 = x[2] * x[3] - x[4] * 5. * x[5];
    f4 = 1.;
    if (f2 < 0.) {
	f4 = -f4;
    }
    g[2] += f4 * 10. * x[3];
    g[3] += f4 * 10. * x[2];
    g[4] -= f4 * 50. * x[5];
    g[5] -= f4 * 50. * x[4];
/* Computing 3rd power */
    d__1 = x[1];
/* Computing 3rd power */
    d__2 = x[2];
    f3 = d__1 * (d__1 * d__1) + d__2 * (d__2 * d__2) + 1.;
    f4 = 1.;
    if (f3 < 0.) {
	f4 = -f4;
    }
/* Computing 2nd power */
    d__1 = x[1];
    g[1] += f4 * 30. * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[2];
    g[2] += f4 * 30. * (d__1 * d__1);
    return 0;
L170:
    for (i__ = 1; i__ <= 51; ++i__) {
	t = (doublereal) (i__ - 1) / 10.;
	f = exp(-t) * .5 - exp(-t * 2.) + exp(-t * 3.) * .5 + exp(-t * 1.5) * 
		1.5 * sin(t * 7.) + exp(-t * 2.5) * sin(t * 5.);
	f1 = exp(-x[2] * t);
	f2 = exp(-x[6] * t);
	f3 = cos(x[3] * t + x[4]);
	f4 = sin(x[3] * t + x[4]);
	d__1 = x[1] * f1 * f3 + x[5] * f2 - f;
	ai = d_sign(&c_b164, &d__1);
	g[1] += ai * f1 * f3;
	g[2] -= ai * f1 * f3 * x[1] * t;
	g[3] -= ai * f1 * f4 * x[1] * t;
	g[4] -= ai * f1 * f4 * x[1];
	g[5] += ai * f2;
	g[6] -= ai * f2 * x[5] * t;
/* L172: */
    }
    return 0;
L120:
    f = -1e60;
    l = 1;
    kk = 0;
    for (k = 1; k <= 5; ++k) {
	f1 = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    f1 -= empr19_1.y[kk + 100 + i__ - 1] * x[i__];
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		f1 += empr19_1.y[kk + (i__ - 1) * *n + j - 1] * x[i__] * x[j];
/* L122: */
	    }
	}
	if (f < f1) {
	    l = k;
	}
	f = max(f,f1);
	kk += 110;
/* L123: */
    }
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	g[i__] = -empr19_1.y[(l - 1) * 110 + 100 + i__ - 1];
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    g[i__] += empr19_1.y[(l - 1) * 110 + (i__ - 1) * *n + j - 1] * 2. 
		    * x[j];
/* L126: */
	}
    }
    return 0;
L210:
    f1 = 0.;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 += d__1 * d__1;
/* L211: */
    }
    f4 = f1 - .25;
    f1 = f4 * .001 * f4;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__] - 1.;
	f1 += d__1 * d__1;
/* L212: */
    }
    f2 = 0.;
    for (i__ = 2; i__ <= 30; ++i__) {
	ai = (doublereal) (i__ - 1) / 29.;
	f = 0.;
	for (j = 1; j <= 10; ++j) {
	    i__1 = j - 1;
	    f += x[j] * pow_di(&ai, &i__1);
/* L213: */
	}
	f = -f * f - 1.;
	for (j = 2; j <= 10; ++j) {
	    aj = (doublereal) (j - 1);
	    i__1 = j - 2;
	    f += x[j] * aj * pow_di(&ai, &i__1);
/* L214: */
	}
	f2 += f * f;
/* L215: */
    }
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__3 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - d__3 * d__3 - 1.;
    f2 = f2 + d__1 * d__1 + d__2 * d__2;
    f3 = 0.;
    for (i__ = 2; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__2 = x[i__ - 1];
/* Computing 2nd power */
	d__1 = x[i__] - d__2 * d__2;
/* Computing 2nd power */
	d__3 = 1. - x[i__];
	f3 = f3 + d__1 * d__1 * 100. + d__3 * d__3;
/* L216: */
    }
    if (f1 >= f2 && f1 >= f3) {
	for (i__ = 1; i__ <= 10; ++i__) {
	    g[i__] = x[i__] * 2. - 2. + x[i__] * .004 * f4;
/* L218: */
	}
    } else if (f2 >= f1 && f2 >= f3) {
	for (j = 1; j <= 10; ++j) {
	    for (i__ = 2; i__ <= 30; ++i__) {
		ai = (doublereal) (i__ - 1) / 29.;
		f = 0.;
		for (k = 1; k <= 10; ++k) {
		    i__1 = k - 1;
		    f -= x[k] * pow_di(&ai, &i__1);
/* L2183: */
		}
		i__1 = j - 1;
		t = f * 2. * pow_di(&ai, &i__1);
		if (j >= 2) {
		    i__1 = j - 2;
		    t += (j - 1) * pow_di(&ai, &i__1);
		}
		f = -f * f - 1.;
		for (k = 2; k <= 10; ++k) {
		    i__1 = k - 2;
		    f += x[k] * (k - 1) * pow_di(&ai, &i__1);
/* L2184: */
		}
		g[j] += f * 2. * t;
/* L2185: */
	    }
	}
/* Computing 2nd power */
	d__1 = x[1];
	g[1] = g[1] + x[1] * 2. - x[1] * 4. * (x[2] - d__1 * d__1 - 1.);
/* Computing 2nd power */
	d__1 = x[1];
	g[2] += (x[2] - d__1 * d__1 - 1.) * 2.;
    } else {
	for (i__ = 1; i__ <= 10; ++i__) {
	    g[i__] = 0.;
	    if (i__ >= 2) {
/* Computing 2nd power */
		d__1 = x[i__ - 1];
		g[i__] = g[i__] + (x[i__] - d__1 * d__1) * 200. - (1. - x[i__]
			) * 2.;
	    }
	    if (i__ <= 9) {
/* Computing 2nd power */
		d__1 = x[i__];
		g[i__] -= x[i__] * 400. * (x[i__ + 1] - d__1 * d__1);
	    }
/* L219: */
	}
    }
    return 0;
L220:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[7];
    g[1] = x[1] / sqrt(d__1 * d__1 + d__2 * d__2);
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[7];
    g[7] = x[7] / sqrt(d__1 * d__1 + d__2 * d__2);
/* Computing 2nd power */
    d__1 = 5.5 - x[6];
/* Computing 2nd power */
    d__2 = x[12] + 1.;
    t = sqrt(d__1 * d__1 + d__2 * d__2);
    g[6] = -(5.5 - x[6]) / t;
    g[12] = (x[12] + 1.) / t;
    for (j = 1; j <= 6; ++j) {
/* Computing 2nd power */
	d__1 = empr19_1.y[j - 1] - x[j];
/* Computing 2nd power */
	d__2 = empr19_1.y[j + 5] - x[j + 6];
	t = sqrt(d__1 * d__1 + d__2 * d__2);
	g[j] -= empr19_1.y[j + 11] * (empr19_1.y[j - 1] - x[j]) / t;
	g[j + 6] -= empr19_1.y[j + 11] * (empr19_1.y[j + 5] - x[j + 6]) / t;
/* L223: */
    }
    for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	d__1 = x[j] - x[j + 1];
/* Computing 2nd power */
	d__2 = x[j + 6] - x[j + 7];
	t = sqrt(d__1 * d__1 + d__2 * d__2);
	g[j] += empr19_1.y[j + 17] * (x[j] - x[j + 1]) / t;
	g[j + 1] -= empr19_1.y[j + 17] * (x[j] - x[j + 1]) / t;
	g[j + 6] += empr19_1.y[j + 17] * (x[j + 6] - x[j + 7]) / t;
	g[j + 7] -= empr19_1.y[j + 17] * (x[j + 6] - x[j + 7]) / t;
/* L224: */
    }
    return 0;
L130:
/* Computing 2nd power */
    d__1 = x[1];
    f = d__1 * d__1;
    k = 1;
    for (i__ = 2; i__ <= 20; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 = d__1 * d__1;
	if (f < f1) {
	    k = i__;
	}
	f = max(f,f1);
/* L131: */
    }
    g[k] = x[k] * 2.;
    return 0;
L140:
    f = abs(x[1]);
    k = 1;
    for (i__ = 2; i__ <= 20; ++i__) {
	f1 = (d__1 = x[i__], abs(d__1));
	if (f < f1) {
	    k = i__;
	}
	f = max(f,f1);
/* L141: */
    }
    g[k] = d_sign(&c_b164, &x[k]);
    return 0;
L150:
    kk = *n * *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__] = -empr19_1.y[kk + *n + i__ - 1];
/* L151: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	f = 1e60;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t = empr19_1.y[(i__ - 1) * *n + j - 1] - x[i__];
	    if (t >= f) {
		goto L153;
	    }
	    f = t;
	    k = i__;
L153:
	    ;
	}
	g[k] += empr19_1.y[kk + j - 1];
/* L152: */
    }
    return 0;
L160:
    f = -1e60;
    for (i__ = 1; i__ <= 50; ++i__) {
	f2 = x[i__];
	if (f < f2) {
	    k = i__;
	}
	f = max(f,f2);
	g[i__] = -1.;
/* L162: */
    }
    g[k] += 50.;
    return 0;
L190:
    f1 = -1e60;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	f = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    f += x[j] / (doublereal) (i__ + j - 1);
/* L191: */
	}
	if (f1 >= abs(f)) {
	    goto L192;
	}
	k = i__;
	ai = d_sign(&c_b164, &f);
	f1 = abs(f);
L192:
	;
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	g[j] = ai / (doublereal) (k + j - 1);
/* L194: */
    }
    return 0;
L200:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	f1 = 0.;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    f1 += x[i__] / (doublereal) (i__ + j - 1);
/* L203: */
	}
	aj = d_sign(&c_b164, &f1);
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    g[i__] += aj / (doublereal) (i__ + j - 1);
/* L204: */
	}
    }
    return 0;
L250:
    f1 = 0.;
    for (j = 1; j <= 5; ++j) {
/* Computing 3rd power */
	d__1 = x[j];
	f1 += empr19_1.y[j + 84] * (d__1 * (d__1 * d__1));
/* L251: */
    }
    for (j = 1; j <= 5; ++j) {
	t = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    t += empr19_1.y[i__ + 55 + j * 5 - 1] * x[i__];
/* L252: */
	}
	g[j] = d_sign(&c_b198, &f1) * empr19_1.y[j + 84] * x[j] * x[j] + t + 
		t;
/* L253: */
    }
    for (j = 6; j <= 15; ++j) {
	g[j] = -empr19_1.y[j + 44];
/* L254: */
    }
    for (j = 1; j <= 5; ++j) {
	t = empr19_1.y[j + 84] * -3. * x[j] * x[j] - empr19_1.y[j + 89];
	for (i__ = 1; i__ <= 15; ++i__) {
	    if (i__ <= 5) {
		t -= empr19_1.y[i__ + 55 + j * 5 - 1] * 2. * x[i__];
	    }
	    if (i__ > 5) {
		t += empr19_1.y[i__ + j * 10 - 16] * x[i__];
	    }
/* L255: */
	}
	if (t <= 0.) {
	    goto L257;
	}
	g[j] -= empr19_1.y[j + 84] * 600. * x[j];
	for (i__ = 1; i__ <= 15; ++i__) {
	    if (i__ <= 5) {
		g[i__] -= empr19_1.y[i__ + 55 + j * 5 - 1] * 200.;
	    }
	    if (i__ > 5) {
		g[i__] += empr19_1.y[i__ + j * 10 - 16] * 100.;
	    }
/* L256: */
	}
L257:
	;
    }
    for (i__ = 1; i__ <= 15; ++i__) {
	if (x[i__] < 0.) {
	    g[i__] += -100.;
	}
/* L258: */
    }
    return 0;
} /* tfgu19_ */

/* SUBROUTINE TFHD19                ALL SYSTEMS                99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 95/12/01 VL : ORIGINAL VERSION */

/* PURPOSE : */
/*  HESSIAN MATRIX OF THE NONSMOOTH OBJECTIVE FUNCTION. */
/*  DENSE VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  H(N*(N+1)/2)  HESSIAN MATRIX OF THE OBJECTIVE FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tfhd19_(integer *n, doublereal *x, doublereal *h__, 
	integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    double d_sign(doublereal *, doublereal *), exp(doublereal), pow_dd(
	    doublereal *, doublereal *), sin(doublereal), cos(doublereal), 
	    pow_di(doublereal *, integer *);

    /* Local variables */
    static doublereal f;
    static integer i__, j, k, l;
    static doublereal t, f1, f2, f3, f4, ai, aj;
    static integer kk;

    /* Parameter adjustments */
    --h__;
    --x;

    /* Function Body */
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	h__[i__] = 0.;
/* L1: */
    }
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L30;
	case 5:  goto L50;
	case 6:  goto L60;
	case 7:  goto L70;
	case 8:  goto L80;
	case 9:  goto L90;
	case 10:  goto L180;
	case 11:  goto L100;
	case 12:  goto L110;
	case 13:  goto L230;
	case 14:  goto L240;
	case 15:  goto L170;
	case 16:  goto L120;
	case 17:  goto L210;
	case 18:  goto L220;
	case 19:  goto L130;
	case 20:  goto L140;
	case 21:  goto L150;
	case 22:  goto L160;
	case 23:  goto L190;
	case 24:  goto L200;
	case 25:  goto L250;
    }
L10:
/* Computing 2nd power */
    d__1 = x[1];
    h__[1] = d__1 * d__1 * 1200. - x[2] * 400. + 2.;
    h__[2] = x[1] * -400.;
    h__[3] = 200.;
    return 0;
L20:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - 1.;
    t = d__1 * d__1 + d__2 * d__2 - 1.;
    f = d_sign(&c_b147, &t);
    h__[1] = f;
    h__[3] = f;
    return 0;
L30:
    i__ = *next - 2;
    j = 5 - *next;
/* Computing 2nd power */
    d__1 = x[i__];
/* Computing 4th power */
    d__2 = x[j], d__2 *= d__2;
    f1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = 2. - x[1];
/* Computing 2nd power */
    d__2 = 2. - x[2];
    f2 = d__1 * d__1 + d__2 * d__2;
    f3 = exp(-x[1] + x[2]) * 2.;
    if (f1 >= f2 && f1 >= f3) {
	h__[(*next << 1) - 5] = 2.;
/* Computing 2nd power */
	d__1 = x[j];
	h__[9 - (*next << 1)] = d__1 * d__1 * 12.;
    } else if (f2 >= f1 && f2 >= f3) {
	h__[1] = 2.;
	h__[3] = 2.;
    } else {
	h__[1] = exp(x[2] - x[1]) * 2.;
	h__[2] = -h__[1];
	h__[3] = h__[1];
    }
    return 0;
L50:
    f1 = x[1] * 5. + x[2];
    f2 = x[1] * -5. + x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f3 = d__1 * d__1 + d__2 * d__2 + x[2] * 4.;
    if (f1 >= f2 && f1 >= f3) {
    } else if (f2 >= f1 && f2 >= f3) {
    } else {
	h__[1] = 2.;
	h__[3] = 2.;
    }
    return 0;
L60:
    h__[1] = 2.;
    h__[3] = 2.;
    return 0;
L70:
    f1 = -x[1] - x[2];
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f2 = f1 + (d__1 * d__1 + d__2 * d__2 - 1.);
    if (f1 < f2) {
	h__[1] = 2.;
	h__[3] = 2.;
    }
    return 0;
L80:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    f1 = d__1 * d__1 + d__2 * d__2 - 1.;
    if (f1 >= 0.) {
	h__[1] = 40.;
	h__[3] = 40.;
    }
    return 0;
L90:
/* Computing 2nd power */
    d__2 = x[1];
/* Computing 2nd power */
    d__3 = x[2];
    d__1 = d__2 * d__2 + d__3 * d__3 - 1.;
    f1 = d_sign(&c_b148, &d__1) + 4.;
    h__[1] = f1;
    h__[3] = f1;
    return 0;
L180:
    if (x[1] > abs(x[2])) {
/* Computing 2nd power */
	d__2 = x[1];
/* Computing 2nd power */
	d__3 = x[2];
	d__1 = d__2 * d__2 * 9. + d__3 * d__3 * 16.;
	f1 = pow_dd(&d__1, &c_b232) * 720.;
/* Computing 2nd power */
	d__1 = x[2];
	h__[1] = f1 * (d__1 * d__1);
	h__[2] = -f1 * x[1] * x[2];
/* Computing 2nd power */
	d__1 = x[1];
	h__[3] = f1 * (d__1 * d__1);
    } else {
	if (x[1] < 0.) {
/* Computing 8th power */
	    d__1 = x[1], d__1 *= d__1, d__1 *= d__1;
	    h__[1] = d__1 * d__1 * -72.;
	}
    }
    return 0;
L100:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
    f = d__1 * d__1 + d__2 * d__2 + d__3 * d__3;
/* Computing 2nd power */
    d__1 = x[4];
    f2 = f + d__1 * d__1 + x[1] - x[2] + x[3] - x[4] - 8.;
/* Computing 2nd power */
    d__1 = x[2];
/* Computing 2nd power */
    d__2 = x[4];
    f3 = f + d__1 * d__1 + d__2 * d__2 * 2. - x[1] - x[4] - 10.;
    f4 = f + x[1] * 2. - x[2] - x[4] - 5.;
    l = 1;
    if (f2 > 0.) {
	l = 2;
    }
    if (f3 > max(f2,0.)) {
	l = 3;
    }
/* Computing MAX */
    d__1 = max(f2,f3);
    if (f4 > max(d__1,0.)) {
	l = 4;
    }
    switch (l) {
	case 1:  goto L101;
	case 2:  goto L102;
	case 3:  goto L103;
	case 4:  goto L104;
    }
L101:
    h__[1] = 2.;
    h__[3] = 2.;
    h__[6] = 4.;
    h__[10] = 2.;
    return 0;
L102:
    h__[10] = 22.;
    goto L106;
L103:
    h__[3] = 42.;
    h__[10] = 42.;
    goto L105;
L104:
    h__[10] = 2.;
L106:
    h__[3] = 22.;
L105:
    h__[1] = 22.;
    h__[6] = 24.;
    return 0;
L110:
    f = -1e60;
    for (i__ = 1; i__ <= 10; ++i__) {
	f1 = 0.;
	for (j = 1; j <= 5; ++j) {
/* Computing 2nd power */
	    d__1 = x[j] - empr19_1.y[(i__ - 1) * 5 + j - 1];
	    f1 += d__1 * d__1;
/* L113: */
	}
	f1 *= empr19_1.y[i__ + 49];
	if (f < f1) {
	    k = i__;
	}
	f = max(f,f1);
/* L112: */
    }
    for (j = 1; j <= 5; ++j) {
	h__[j * (j + 1) / 2] = empr19_1.y[k + 49] * 2.;
/* L114: */
    }
    return 0;
L230:
    for (j = 1; j <= 5; ++j) {
	h__[j * (j + 1) / 2] = empr19_1.y[j + 84] * 6. * x[j];
/* L231: */
    }
    k = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	for (j = 1; j <= i__2; ++j) {
	    h__[k] += empr19_1.y[i__ + 55 + j * 5 - 1] * 2.;
	    ++k;
/* L232: */
	}
    }
    return 0;
L240:
    h__[1] = 0.;
    h__[2] = x[3] * x[4] * x[5];
    h__[3] = 0.;
    h__[4] = x[2] * x[4] * x[5];
    h__[5] = x[1] * x[4] * x[5];
    h__[6] = 0.;
    h__[7] = x[2] * x[3] * x[5];
    h__[8] = x[1] * x[3] * x[5];
    h__[9] = x[1] * x[2] * x[5];
    h__[10] = 0.;
    h__[11] = x[2] * x[3] * x[4];
    h__[12] = x[1] * x[3] * x[4];
    h__[13] = x[1] * x[2] * x[4];
    h__[14] = x[1] * x[2] * x[3];
    h__[15] = 0.;
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
/* Computing 2nd power */
    d__4 = x[4];
/* Computing 2nd power */
    d__5 = x[5];
    f1 = d__1 * d__1 + d__2 * d__2 + d__3 * d__3 + d__4 * d__4 + d__5 * d__5 
	    - 10.;
    f4 = 1.;
    if (f1 < 0.) {
	f4 = -f4;
    }
    l = 0;
    for (i__ = 1; i__ <= 5; ++i__) {
	l += i__;
	h__[l] += f4 * 20.;
/* L241: */
    }
    f2 = x[2] * x[3] - x[4] * 5. * x[5];
    f4 = 1.;
    if (f2 < 0.) {
	f4 = -f4;
    }
    h__[5] += f4 * 10.;
    h__[14] -= f4 * 50.;
/* Computing 3rd power */
    d__1 = x[1];
/* Computing 3rd power */
    d__2 = x[2];
    f3 = d__1 * (d__1 * d__1) + d__2 * (d__2 * d__2) + 1.;
    f4 = 1.;
    if (f3 < 0.) {
	f4 = -f4;
    }
    h__[1] += f4 * 60. * x[1];
    h__[3] += f4 * 60. * x[2];
    return 0;
L170:
    for (i__ = 1; i__ <= 51; ++i__) {
	t = (doublereal) (i__ - 1) / 10.;
	f = exp(-t) * .5 - exp(-t * 2.) + exp(-t * 3.) * .5 + exp(-t * 1.5) * 
		1.5 * sin(t * 7.) + exp(-t * 2.5) * sin(t * 5.);
	f1 = exp(-x[2] * t);
	f2 = exp(-x[6] * t);
	f3 = cos(x[3] * t + x[4]);
	f4 = sin(x[3] * t + x[4]);
	d__1 = x[1] * f1 * f3 + x[5] * f2 - f;
	ai = d_sign(&c_b164, &d__1);
	h__[2] -= ai * f1 * f3 * t;
	h__[3] += ai * f1 * f3 * t * t * x[1];
	h__[4] -= ai * f1 * f4 * t;
	h__[5] += ai * f1 * f4 * t * t * x[1];
	h__[6] -= ai * f1 * f3 * t * t * x[1];
	h__[7] -= ai * f1 * f4;
	h__[8] += ai * f1 * f4 * t * x[1];
	h__[9] -= ai * f1 * f3 * t * x[1];
	h__[10] -= ai * f1 * f3 * x[1];
	h__[15] -= ai * f2 * t;
	h__[20] -= ai * f2 * t;
	h__[21] += ai * f2 * t * t * x[5];
/* L172: */
    }
    return 0;
L120:
    f = -1e60;
    l = 1;
    kk = 0;
    for (k = 1; k <= 5; ++k) {
	f1 = 0.;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    f1 -= empr19_1.y[kk + 100 + i__ - 1] * x[i__];
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		f1 += empr19_1.y[kk + (i__ - 1) * *n + j - 1] * x[i__] * x[j];
/* L122: */
	    }
	}
	if (f < f1) {
	    l = k;
	}
	f = max(f,f1);
	kk += 110;
/* L123: */
    }
    k = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	for (j = 1; j <= i__2; ++j) {
	    h__[k] = empr19_1.y[(l - 1) * 110 + (i__ - 1) * *n + j - 1] * 2.;
	    ++k;
/* L124: */
	}
    }
    return 0;
L210:
    f1 = 0.;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 += d__1 * d__1;
/* L211: */
    }
    f4 = f1 - .25;
    f1 = f4 * .001 * f4;
    for (i__ = 1; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__] - 1.;
	f1 += d__1 * d__1;
/* L212: */
    }
    f2 = 0.;
    for (i__ = 2; i__ <= 30; ++i__) {
	ai = (doublereal) (i__ - 1) / 29.;
	f = 0.;
	for (j = 1; j <= 10; ++j) {
	    i__2 = j - 1;
	    f += x[j] * pow_di(&ai, &i__2);
/* L213: */
	}
	f = -f * f - 1.;
	for (j = 2; j <= 10; ++j) {
	    aj = (doublereal) (j - 1);
	    i__2 = j - 2;
	    f += x[j] * aj * pow_di(&ai, &i__2);
/* L214: */
	}
	f2 += f * f;
/* L215: */
    }
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__3 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - d__3 * d__3 - 1.;
    f2 = f2 + d__1 * d__1 + d__2 * d__2;
    f3 = 0.;
    for (i__ = 2; i__ <= 10; ++i__) {
/* Computing 2nd power */
	d__2 = x[i__ - 1];
/* Computing 2nd power */
	d__1 = x[i__] - d__2 * d__2;
/* Computing 2nd power */
	d__3 = 1. - x[i__];
	f3 = f3 + d__1 * d__1 * 100. + d__3 * d__3;
/* L216: */
    }
    if (f1 >= f2 && f1 >= f3) {
	l = 1;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__1 = i__;
	    for (j = 1; j <= i__1; ++j) {
		h__[l] = x[i__] * .008 * x[j];
		if (j == i__) {
		    h__[l] = h__[l] + 2. + f4 * .004;
		}
		++l;
/* L218: */
	    }
	}
    } else if (f2 >= f1 && f2 >= f3) {
	kk = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    for (l = 1; l <= i__2; ++l) {
		for (i__ = 2; i__ <= 30; ++i__) {
		    ai = (doublereal) (i__ - 1) / 29.;
		    f = 0.;
		    for (k = 1; k <= 10; ++k) {
			i__3 = k - 1;
			f -= x[k] * pow_di(&ai, &i__3);
/* L2183: */
		    }
		    i__3 = j - 1;
		    t = f * 2. * pow_di(&ai, &i__3);
		    if (j >= 2) {
			i__3 = j - 2;
			t += (j - 1) * pow_di(&ai, &i__3);
		    }
		    i__3 = l - 1;
		    f4 = f * 2. * pow_di(&ai, &i__3);
		    if (l >= 2) {
			i__3 = l - 2;
			f4 += (l - 1) * pow_di(&ai, &i__3);
		    }
		    f = -f * f - 1.;
		    for (k = 2; k <= 10; ++k) {
			i__3 = k - 2;
			f += x[k] * (k - 1) * pow_di(&ai, &i__3);
/* L2184: */
		    }
		    i__3 = j + l - 2;
		    h__[kk] += (t * f4 - f * 2. * pow_di(&ai, &i__3)) * 2.;
/* L2185: */
		}
		++kk;
/* L2186: */
	    }
	}
/* Computing 2nd power */
	d__1 = x[1];
	h__[1] = h__[1] + d__1 * d__1 * 12. - x[2] * 4. + 6.;
	h__[2] -= x[1] * 4.;
	h__[3] += 2.;
    } else {
	for (i__ = 1; i__ <= 10; ++i__) {
	    j = i__ * (i__ + 1) / 2;
	    if (i__ >= 2) {
		h__[j] = 202.;
	    }
	    if (i__ <= 9) {
/* Computing 2nd power */
		d__1 = x[i__];
		h__[j] += (d__1 * d__1 * 3. - x[i__ + 1]) * 400.;
	    }
	    if (i__ <= 9) {
		h__[j + i__] = x[i__] * -400.;
	    }
/* L219: */
	}
    }
    return 0;
L220:
/* Computing 2nd power */
    d__2 = x[1];
/* Computing 2nd power */
    d__3 = x[7];
    d__1 = d__2 * d__2 + d__3 * d__3;
    t = pow_dd(&d__1, &c_b262);
/* Computing 2nd power */
    d__1 = x[7];
    h__[1] = d__1 * d__1 / t;
    h__[22] = -x[1] * x[7] / t;
/* Computing 2nd power */
    d__1 = x[1];
    h__[28] = d__1 * d__1 / t;
/* Computing 2nd power */
    d__2 = 5.5 - x[6];
/* Computing 2nd power */
    d__3 = x[12] + 1.;
    d__1 = d__2 * d__2 + d__3 * d__3;
    t = pow_dd(&d__1, &c_b262);
/* Computing 2nd power */
    d__1 = x[12] + 1.;
    h__[21] = d__1 * d__1 / t;
    h__[72] = (5.5 - x[6]) * (x[12] + 1.) / t;
/* Computing 2nd power */
    d__1 = 5.5 - x[6];
    h__[78] = d__1 * d__1 / t;
    for (j = 1; j <= 6; ++j) {
/* Computing 2nd power */
	d__2 = empr19_1.y[j - 1] - x[j];
/* Computing 2nd power */
	d__3 = empr19_1.y[j + 5] - x[j + 6];
	d__1 = d__2 * d__2 + d__3 * d__3;
	t = empr19_1.y[j + 11] / pow_dd(&d__1, &c_b262);
/* Computing 2nd power */
	d__1 = empr19_1.y[j + 5] - x[j + 6];
	h__[(j - 1) * j / 2 + j] += d__1 * d__1 * t;
	i__2 = j + 6;
	i__1 = j + 6;
	h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] - (
		empr19_1.y[j + 5] - x[j + 6]) * (empr19_1.y[j - 1] - x[j]) * 
		t;
	i__2 = j + 6;
	i__1 = j + 6;
	i__3 = j + 6;
	i__4 = j + 6;
/* Computing 2nd power */
	d__1 = empr19_1.y[j - 1] - x[j];
	h__[(i__1 - 1) * i__1 / 2 + i__2] = h__[(i__4 - 1) * i__4 / 2 + i__3] 
		+ d__1 * d__1 * t;
/* L223: */
    }
    for (j = 1; j <= 6; ++j) {
	if (j < 6) {
	    f1 = x[j] - x[j + 1];
	    f2 = x[j + 6] - x[j + 7];
	    d__1 = f1 * f1 + f2 * f2;
	    t = empr19_1.y[j + 17] / pow_dd(&d__1, &c_b262);
	    h__[(j - 1) * j / 2 + j] += f2 * f2 * t;
	    i__2 = j + 1;
	    i__1 = j + 1;
	    h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] - 
		    f2 * f2 * t;
	    i__2 = j + 6;
	    i__1 = j + 6;
	    h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] - 
		    f1 * f2 * t;
	    i__2 = j + 7;
	    i__1 = j + 7;
	    h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] + 
		    f1 * f2 * t;
	    i__2 = j + 6;
	    i__1 = j + 6;
	    i__3 = j + 6;
	    i__4 = j + 6;
	    h__[(i__1 - 1) * i__1 / 2 + i__2] = h__[(i__4 - 1) * i__4 / 2 + 
		    i__3] + f1 * f1 * t;
	    i__2 = j + 6;
	    i__1 = j + 7;
	    i__3 = j + 6;
	    i__4 = j + 7;
	    h__[(i__1 - 1) * i__1 / 2 + i__2] = h__[(i__4 - 1) * i__4 / 2 + 
		    i__3] - f1 * f1 * t;
	}
	if (j > 1) {
	    f1 = x[j] - x[j - 1];
	    f2 = x[j + 6] - x[j + 5];
	    d__1 = f1 * f1 + f2 * f2;
	    t = empr19_1.y[j + 16] / pow_dd(&d__1, &c_b262);
	    h__[(j - 1) * j / 2 + j] += f2 * f2 * t;
	    i__2 = j + 5;
	    i__1 = j + 5;
	    h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] + 
		    f1 * f2 * t;
	    i__2 = j + 6;
	    i__1 = j + 6;
	    h__[(i__2 - 1) * i__2 / 2 + j] = h__[(i__1 - 1) * i__1 / 2 + j] - 
		    f1 * f2 * t;
	    i__2 = j + 6;
	    i__1 = j + 6;
	    i__3 = j + 6;
	    i__4 = j + 6;
	    h__[(i__1 - 1) * i__1 / 2 + i__2] = h__[(i__4 - 1) * i__4 / 2 + 
		    i__3] + f1 * f1 * t;
	}
/* L224: */
    }
    return 0;
L130:
/* Computing 2nd power */
    d__1 = x[1];
    f = d__1 * d__1;
    k = 1;
    for (i__ = 2; i__ <= 20; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
	f1 = d__1 * d__1;
	if (f < f1) {
	    k = i__;
	}
	f = max(f,f1);
/* L131: */
    }
    h__[k * (k + 1) / 2] = 2.;
    return 0;
L140:
    return 0;
L150:
    return 0;
L160:
    return 0;
L190:
    return 0;
L200:
    return 0;
L250:
    f1 = 0.;
    for (j = 1; j <= 5; ++j) {
/* Computing 3rd power */
	d__1 = x[j];
	f1 += empr19_1.y[j + 84] * (d__1 * (d__1 * d__1));
/* L251: */
    }
    for (j = 1; j <= 5; ++j) {
	h__[j * (j + 1) / 2] = d_sign(&c_b272, &f1) * empr19_1.y[j + 84] * x[
		j];
/* L252: */
    }
    k = 1;
    for (i__ = 1; i__ <= 5; ++i__) {
	i__2 = i__;
	for (j = 1; j <= i__2; ++j) {
	    h__[k] += empr19_1.y[i__ + 55 + j * 5 - 1] * 2.;
	    ++k;
/* L253: */
	}
    }
    for (j = 1; j <= 5; ++j) {
	t = empr19_1.y[j + 84] * -3. * x[j] * x[j] - empr19_1.y[j + 89];
	for (i__ = 1; i__ <= 15; ++i__) {
	    if (i__ <= 5) {
		t -= empr19_1.y[i__ + 55 + j * 5 - 1] * 2. * x[i__];
	    }
	    if (i__ > 5) {
		t += empr19_1.y[i__ + j * 10 - 16] * x[i__];
	    }
/* L255: */
	}
	if (t <= 0.) {
	    goto L257;
	}
	i__ = j * (j + 1) / 2;
	h__[i__] -= empr19_1.y[j + 84] * 600.;
L257:
	;
    }
    return 0;
} /* tfhd19_ */

