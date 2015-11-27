/* lvlcmm.f -- translated by f2c (version 20100827).
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
    doublereal y[163];
} empr22_;

#define empr22_1 empr22_

/* SUBROUTINE EILD22             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  INITIATION OF VARIABLES FOR NONLINEAR MINIMAX APPROXIMATION. */
/*  LINEARLY CONSTRAINED DENSE VERSION. */

/* PARAMETERS : */
/*  IO  N  NUMBER OF VARIABLES. */
/*  IO  NA  NUMBER OF PARTIAL FUNCTIONS. */
/*  IO  NB  NUMBER OF BOX CONSTRAINTS. */
/*  IO  NC  NUMBER OF GENERAL LINEAR CONSTRAINTS. */
/*  RO  X(N)  VECTOR OF VARIABLES. */
/*  IO  IX(NF)  VECTOR CONTAINING TYPES OF BOUNDS. */
/*  RO  XL(NF)  VECTOR CONTAINING LOWER BOUNDS FOR VARIABLES. */
/*  RO  XU(NF)  VECTOR CONTAINING UPPER BOUNDS FOR VARIABLES. */
/*  IO  IC(NC)  VECTOR CONTAINING TYPES OF CONSTRAINTS. */
/*  RO  CL(NC)  VECTOR CONTAINING LOWER BOUNDS FOR CONSTRAINT FUNCTIONS. */
/*  RO  CU(NC)  VECTOR CONTAINING UPPER BOUNDS FOR CONSTRAINT FUNCTIONS. */
/*  RO  CG(NF*NC) MATRIX WHOSE COLUMNS ARE NORMALS OF THE LINEAR */
/*         CONSTRAINTS. */
/*  RO  FMIN  LOWER BOUND FOR VALUE OF THE OBJECTIVE FUNCTION. */
/*  RO  XMAX  MAXIMUM STEPSIZE. */
/*  IO  NEXT  NUMBER OF THE TEST PROBLEM. */
/*  IO  IEXT  TYPE OF OBJECTIVE FUNCTION. IEXT<0-MAXIMUM OF VALUES. */
/*         IEXT=0-MAXIMUM OF ABSOLUTE VALUES. */
/*  IO  IERR  ERROR INDICATOR. */

/* Subroutine */ int eild22_(integer *n, integer *na, integer *nb, integer *
	nc, doublereal *x, integer *ix, doublereal *xl, doublereal *xu, 
	integer *ic, doublereal *cl, doublereal *cu, doublereal *cg, 
	doublereal *fmin, doublereal *xmax, integer *next, integer *iext, 
	integer *ierr)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double sin(doublereal), cos(doublereal);

    /* Local variables */
    static doublereal a;
    static integer i__, j, k, l;

    /* Parameter adjustments */
    --xu;
    --xl;
    --ix;
    --x;
    --cg;
    --cu;
    --cl;
    --ic;

    /* Function Body */
    *fmin = -1e60;
    *xmax = 1e3;
    *iext = -1;
    *ierr = 0;
    *nb = 0;
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L40;
	case 5:  goto L80;
	case 6:  goto L50;
	case 7:  goto L100;
	case 8:  goto L190;
	case 9:  goto L200;
	case 10:  goto L60;
	case 11:  goto L70;
	case 12:  goto L90;
	case 13:  goto L130;
	case 14:  goto L150;
	case 15:  goto L170;
    }
L10:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	*nc = 1;
	x[1] = 1.;
	x[2] = 2.;
	ic[1] = 1;
	cl[1] = .5;
	cg[1] = 1.;
	cg[2] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L20:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	*nc = 1;
	x[1] = -2.;
	x[2] = -1.;
	ic[1] = 2;
	cu[1] = -2.5;
	cg[1] = 3.;
	cg[2] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L30:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	*nb = *n;
	*nc = 1;
	x[1] = -1.;
	x[2] = .01;
	ix[1] = 0;
	ix[2] = 1;
	xl[2] = .01;
	ic[1] = 1;
	cl[1] = -.5;
	cg[1] = .05;
	cg[2] = -1.;
    } else {
	*ierr = 1;
    }
    return 0;
L40:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	*nb = *n;
	*nc = 1;
	x[1] = -1.;
	x[2] = 3.;
	ix[1] = 0;
	ix[2] = 1;
	xl[2] = .01;
	ic[1] = 1;
	cl[1] = 1.;
	cg[1] = -.9;
	cg[2] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L80:
    if (*n >= 6 && *na >= 3) {
	*n = 6;
	*na = 3;
	x[1] = -1.;
	x[2] = 0.;
	x[3] = 0.;
	x[4] = -1.;
	x[5] = 1.;
	x[6] = 1.;
	*nc = *na * 5;
	i__1 = *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cu[i__] = 1.;
	    ic[i__] = 2;
/* L82: */
	}
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L83: */
	}
	k = 1;
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    l = i__ - 1 << 1;
	    for (j = 1; j <= 5; ++j) {
		cg[k + l] = sin((doublereal) (j - 1) * 6.2831853071795862 / 
			5.);
		cg[k + l + 1] = cos((doublereal) (j - 1) * 6.2831853071795862 
			/ 5.);
		k += *n;
/* L84: */
	    }
/* L85: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L50:
    if (*n >= 7 && *na >= 163) {
	*n = 7;
	*na = 163;
	*nb = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = (doublereal) i__ * .5;
	    ix[i__] = 0;
/* L51: */
	}
	xl[1] = .4;
	ix[1] = 1;
	ix[7] = 5;
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    empr22_1.y[i__ - 1] = sin(((doublereal) i__ * .5 + 8.5) * 
		    3.14159265358979323846 / 180.) * 6.2831853071795862;
/* L52: */
	}
	*nc = 7;
	for (i__ = 1; i__ <= 6; ++i__) {
	    cl[i__] = .4;
	    ic[i__] = 1;
/* L53: */
	}
	cl[7] = 1.;
	cu[7] = 1.;
	ic[7] = 5;
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L54: */
	}
	k = 0;
	for (i__ = 1; i__ <= 6; ++i__) {
	    cg[k + i__] = -1.;
	    cg[k + i__ + 1] = 1.;
	    k += *n;
/* L55: */
	}
	cg[46] = -1.;
	cg[48] = 1.;
	*iext = 0;
	*fmin = 0.;
    } else {
	*ierr = 1;
    }
    return 0;
L100:
    if (*n >= 8 && *na >= 8) {
	*n = 8;
	*na = 8;
	*nb = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = .125;
	    xl[i__] = 1e-8;
	    ix[i__] = 1;
/* L101: */
	}
	for (i__ = 1; i__ <= 40; ++i__) {
	    empr22_1.y[i__ - 1] = 1.;
	    empr22_1.y[i__ + 39] = .1;
/* L102: */
	}
	empr22_1.y[8] = 2.;
	empr22_1.y[9] = .8;
	empr22_1.y[11] = .5;
	empr22_1.y[17] = 1.2;
	empr22_1.y[18] = .8;
	empr22_1.y[19] = 1.2;
	empr22_1.y[20] = 1.6;
	empr22_1.y[21] = 2.;
	empr22_1.y[22] = .6;
	empr22_1.y[23] = .1;
	empr22_1.y[24] = 2.;
	empr22_1.y[25] = .1;
	empr22_1.y[26] = .6;
	empr22_1.y[27] = 2.;
	empr22_1.y[31] = 2.;
	empr22_1.y[32] = 1.2;
	empr22_1.y[33] = 1.2;
	empr22_1.y[34] = .8;
	empr22_1.y[36] = 1.2;
	empr22_1.y[37] = .1;
	empr22_1.y[38] = 3.;
	empr22_1.y[39] = 4.;
	empr22_1.y[40] = 3.;
	empr22_1.y[41] = 1.;
	empr22_1.y[44] = 5.;
	empr22_1.y[47] = 6.;
	empr22_1.y[49] = 10.;
	empr22_1.y[52] = 5.;
	empr22_1.y[57] = 9.;
	empr22_1.y[58] = 10.;
	empr22_1.y[60] = 4.;
	empr22_1.y[62] = 7.;
	empr22_1.y[67] = 10.;
	empr22_1.y[69] = 3.;
	empr22_1.y[79] = 11.;
	empr22_1.y[80] = .5;
	empr22_1.y[81] = 1.2;
	empr22_1.y[82] = .8;
	empr22_1.y[83] = 2.;
	empr22_1.y[84] = 1.5;
	*nc = 1;
	cl[1] = 1.;
	cu[1] = 1.;
	ic[1] = 5;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 1.;
/* L103: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L190:
    if (*n >= 10 && *na >= 6) {
	*n = 10;
	*na = 6;
	x[1] = 2.;
	x[2] = 3.;
	x[3] = 5.;
	x[4] = 5.;
	x[5] = 1.;
	x[6] = 2.;
	x[7] = 7.;
	x[8] = 3.;
	x[9] = 6.;
	x[10] = 10.;
	*nc = 3;
	cu[1] = 105.;
	cu[2] = 0.;
	cu[3] = 12.;
	ic[1] = 2;
	ic[2] = 2;
	ic[3] = 2;
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L191: */
	}
	cg[1] = 4.;
	cg[2] = 5.;
	cg[7] = -3.;
	cg[8] = 9.;
	cg[11] = 10.;
	cg[12] = -8.;
	cg[17] = -17.;
	cg[18] = 2.;
	cg[21] = -8.;
	cg[22] = 2.;
	cg[29] = 5.;
	cg[30] = -2.;
    } else {
	*ierr = 1;
    }
    return 0;
L200:
    if (*n >= 20 && *na >= 14) {
	*n = 20;
	*na = 14;
	x[1] = 2.;
	x[2] = 3.;
	x[3] = 5.;
	x[4] = 5.;
	x[5] = 1.;
	x[6] = 2.;
	x[7] = 7.;
	x[8] = 3.;
	x[9] = 6.;
	x[10] = 10.;
	x[11] = 2.;
	x[12] = 2.;
	x[13] = 6.;
	x[14] = 15.;
	x[15] = 1.;
	x[16] = 2.;
	x[17] = 1.;
	x[18] = 2.;
	x[19] = 1.;
	x[20] = 3.;
	*nc = 4;
	cu[1] = 105.;
	cu[2] = 0.;
	cu[3] = 12.;
	cu[4] = 0.;
	ic[1] = 2;
	ic[2] = 2;
	ic[3] = 2;
	ic[4] = 2;
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L201: */
	}
	cg[1] = 4.;
	cg[2] = 5.;
	cg[7] = -3.;
	cg[8] = 9.;
	cg[21] = 10.;
	cg[22] = -8.;
	cg[27] = -17.;
	cg[28] = 2.;
	cg[41] = -8.;
	cg[42] = 2.;
	cg[49] = 5.;
	cg[50] = -2.;
	cg[61] = 1.;
	cg[62] = 1.;
	cg[71] = 4.;
	cg[72] = -21.;
    } else {
	*ierr = 1;
    }
    return 0;
L60:
    if (*n >= 20 && *na >= 38) {
	*n = 20;
	*na = 38;
	*nb = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 100.;
	    ix[i__] = 0;
	    if (i__ <= 10) {
		ix[i__] = 1;
	    }
	    xl[i__] = .5;
/* L61: */
	}
	*nc = 0;
	*iext = 0;
	*fmin = 0.;
    } else {
	*ierr = 1;
    }
    return 0;
L70:
    if (*n >= 9 && *na >= 124) {
	*n = 9;
	*na = 124;
	*nb = *n;
	k = (*n - 1) / 2;
/*      X(1)=1.8D-2 */
/*      X(2)=1.9D-2 */
/*      X(3)=2.0D-2 */
/*      X(4)=2.1D-2 */
/*      X(5)=0.8D 0 */
/*      X(6)=0.9D 0 */
/*      X(7)=1.0D 0 */
/*      X(8)=1.1D 0 */
/*      X(9)=-1.4D 1 */
	x[1] = .0398;
	x[2] = 9.68e-5;
	x[3] = 1.03e-4;
	x[4] = .0389;
	x[5] = 1.01;
	x[6] = .968;
	x[7] = 1.03;
	x[8] = .988;
	x[9] = -11.6;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    xl[i__] = 0.;
	    ix[i__] = 1;
/* L71: */
	}
	ix[*n] = 0;
	l = (*na - 2) / 2;
	a = .02515400000000001 / (doublereal) (l - 1);
	empr22_1.y[0] = .96732;
	i__1 = l + 1;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    empr22_1.y[i__ - 1] = (doublereal) (i__ - 2) * a + .987423;
	    empr22_1.y[i__ + l - 1] = empr22_1.y[i__ - 1];
/* L72: */
	}
	empr22_1.y[*na - 1] = 1.03268;
	*nc = k;
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L73: */
	}
	l = 0;
	i__1 = *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[l + i__] = 1e4;
	    cg[l + i__ + k] = -1.;
	    cl[i__] = 0.;
	    ic[i__] = 1;
	    l += *n;
/* L74: */
	}
	*iext = 0;
	*fmin = 0.;
    } else {
	*ierr = 1;
    }
    return 0;
L90:
    if (*n >= 10 && *na >= 9) {
	*n = 10;
	*na = 9;
	*nb = *n;
	x[1] = 1745.;
	x[2] = 1.2e4;
	x[3] = 110.;
	x[4] = 3048.;
	x[5] = 1974.;
	x[6] = 89.2;
	x[7] = 92.8;
	x[8] = 8.;
	x[9] = 3.6;
	x[10] = 145.;
	xl[1] = 1e-5;
	xl[2] = 1e-5;
	xl[3] = 1e-5;
	xl[4] = 1e-5;
	xl[5] = 1e-5;
	xl[6] = 85.;
	xl[7] = 90.;
	xl[8] = 3.;
	xl[9] = 1.2;
	xl[10] = 140.;
	xu[1] = 2e3;
	xu[2] = 1.6e4;
	xu[3] = 120.;
	xu[4] = 5e3;
	xu[5] = 2e3;
	xu[6] = 93.;
	xu[7] = 95.;
	xu[8] = 12.;
	xu[9] = 4.;
	xu[10] = 160.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ix[i__] = 3;
/* L91: */
	}
	*nc = 5;
	cu[1] = 35.82;
	cl[2] = 35.82;
	cl[3] = 133.;
	cu[4] = 133.;
	cl[5] = 0.;
	cu[5] = 0.;
	ic[1] = 2;
	ic[2] = 1;
	ic[3] = 1;
	ic[4] = 2;
	ic[5] = 5;
	i__1 = *n * *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L92: */
	}
	cg[9] = .9;
	cg[10] = .222;
	cg[19] = 1.1111111111111112;
	cg[20] = .222;
	cg[27] = 3.;
	cg[30] = -.99;
	cg[37] = 3.;
	cg[40] = -1.0101010101010102;
	cg[41] = -1.;
	cg[44] = 1.22;
	cg[45] = -1.;
    } else {
	*ierr = 1;
    }
    return 0;
L130:
    if (*n >= 7 && *na >= 15) {
	*n = 7;
	*na = 13;
	*nb = *n;
	x[1] = 1745.;
	x[2] = 110.;
	x[3] = 3048.;
	x[4] = 89.;
	x[5] = 92.;
	x[6] = 8.;
	x[7] = 145.;
	xl[1] = 1.;
	xl[2] = 1.;
	xl[3] = 1.;
	xl[4] = 85.;
	xl[5] = 90.;
	xl[6] = 3.;
	xl[7] = 145.;
	xu[1] = 2e3;
	xu[2] = 120.;
	xu[3] = 5e3;
	xu[4] = 93.;
	xu[5] = 95.;
	xu[6] = 12.;
	xu[7] = 162.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ix[i__] = 3;
/* L131: */
	}
	empr22_1.y[0] = 1.715;
	empr22_1.y[1] = .035;
	empr22_1.y[2] = 4.0565;
	empr22_1.y[3] = 10.;
	empr22_1.y[4] = 3e3;
	empr22_1.y[5] = -.063;
	empr22_1.y[6] = .0059553571;
	empr22_1.y[7] = .88392857;
	empr22_1.y[8] = -.1175625;
	empr22_1.y[9] = 1.1088;
	empr22_1.y[10] = .1303533;
	empr22_1.y[11] = -.0066033;
	empr22_1.y[12] = 6.6173269e-4;
	empr22_1.y[13] = .017239878;
	empr22_1.y[14] = -.0056595559;
	empr22_1.y[15] = -.019120592;
	empr22_1.y[16] = 56.85075;
	empr22_1.y[17] = 1.08702;
	empr22_1.y[18] = .32175;
	empr22_1.y[19] = -.03762;
	empr22_1.y[20] = .006198;
	empr22_1.y[21] = 2462.3121;
	empr22_1.y[22] = -25.125634;
	empr22_1.y[23] = 161.18996;
	empr22_1.y[24] = 5e3;
	empr22_1.y[25] = -489510.;
	empr22_1.y[26] = 44.333333;
	empr22_1.y[27] = .33;
	empr22_1.y[28] = .022556;
	empr22_1.y[29] = -.007595;
	empr22_1.y[30] = 6.1e-4;
	empr22_1.y[31] = -5e-4;
	empr22_1.y[32] = .819672;
	empr22_1.y[33] = .819672;
	empr22_1.y[34] = 24500.;
	empr22_1.y[35] = -250.;
	empr22_1.y[36] = .010204082;
	empr22_1.y[37] = 1.2244898e-5;
	empr22_1.y[38] = 6.25e-5;
	empr22_1.y[39] = 6.25e-5;
	empr22_1.y[40] = -7.625e-5;
	empr22_1.y[41] = 1.22;
	empr22_1.y[42] = 1.;
	empr22_1.y[43] = -1.;
	*nc = 2;
	l = 0;
	i__1 = *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cu[i__] = 1.;
	    ic[i__] = 2;
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		cg[l + j] = 0.;
/* L132: */
	    }
	    l += *n;
/* L133: */
	}
	cg[5] = empr22_1.y[28];
	cg[7] = empr22_1.y[29];
	cg[8] = empr22_1.y[31];
	cg[10] = empr22_1.y[30];
    } else {
	*ierr = 1;
    }
    return 0;
L150:
    if (*n >= 8 && *na >= 7) {
	*n = 8;
	*na = 4;
	*nb = *n;
	x[1] = 5e3;
	x[2] = 5e3;
	x[3] = 5e3;
	x[4] = 200.;
	x[5] = 350.;
	x[6] = 150.;
	x[7] = 225.;
	x[8] = 425.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    xl[i__] = 10.;
	    xu[i__] = 1e3;
	    ix[i__] = 3;
/* L151: */
	}
	xl[1] = 100.;
	xl[2] = 1e3;
	xl[3] = 1e3;
	xu[1] = 1e4;
	xu[2] = 1e4;
	xu[3] = 1e4;
	*nc = 3;
	l = 0;
	i__1 = *nc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cu[i__] = 1.;
	    ic[i__] = 2;
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		cg[l + j] = 0.;
/* L152: */
	    }
	    l += *n;
/* L153: */
	}
	cg[4] = .0025;
	cg[6] = .0025;
	cg[12] = -.0025;
	cg[13] = .0025;
	cg[15] = .0025;
	cg[21] = -.01;
	cg[24] = .01;
    } else {
	*ierr = 1;
    }
    return 0;
L170:
    if (*n >= 16 && *na >= 20) {
	*n = 16;
	*na = 19;
	*nb = *n;
	x[1] = .8;
	x[2] = .83;
	x[3] = .85;
	x[4] = .87;
	x[5] = .9;
	x[6] = .1;
	x[7] = .12;
	x[8] = .19;
	x[9] = .25;
	x[10] = .29;
	x[11] = 512.;
	x[12] = 13.1;
	x[13] = 71.8;
	x[14] = 640.;
	x[15] = 650.;
	x[16] = 5.7;
	xl[1] = .1;
	xl[2] = .1;
	xl[3] = .1;
	xl[4] = .1;
	xl[5] = .9;
	xl[6] = 1e-4;
	xl[7] = .1;
	xl[8] = .1;
	xl[9] = .1;
	xl[10] = .1;
	xl[11] = 1.;
	xl[12] = 1e-6;
	xl[13] = 1.;
	xl[14] = 500.;
	xl[15] = 500.;
	xl[16] = 1e-6;
	xu[1] = .9;
	xu[2] = .9;
	xu[3] = .9;
	xu[4] = .9;
	xu[5] = 1.;
	xu[6] = .1;
	xu[7] = .9;
	xu[8] = .9;
	xu[9] = .9;
	xu[10] = .9;
	xu[11] = 1e4;
	xu[12] = 5e3;
	xu[13] = 5e3;
	xu[14] = 1e4;
	xu[15] = 1e4;
	xu[16] = 5e3;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ix[i__] = 3;
/* L171: */
	}
	*nc = 1;
	cu[1] = 1.;
	ic[1] = 2;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cg[i__] = 0.;
/* L172: */
	}
	cg[11] = .002;
	cg[12] = -.002;
    } else {
	*ierr = 1;
    }
    return 0;
} /* eild22_ */

/* SUBROUTINE TAFU22             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  VALUES OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  FA  VALUE OF THE PARTIAL FUNCTION AT THE */
/*          SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tafu22_(integer *n, integer *ka, doublereal *x, 
	doublereal *fa, integer *next)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, d__10, 
	    d__11, d__12, d__13, d__14, d__15, d__16, d__17, d__18, d__19, 
	    d__20;

    /* Builtin functions */
    double sin(doublereal), cos(doublereal), exp(doublereal), sinh(doublereal)
	    , log(doublereal), sqrt(doublereal), pow_dd(doublereal *, 
	    doublereal *);

    /* Local variables */
    static doublereal a, b;
    static integer i__, j, k;
    static doublereal p, s;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L10;
	case 3:  goto L30;
	case 4:  goto L30;
	case 5:  goto L80;
	case 6:  goto L50;
	case 7:  goto L100;
	case 8:  goto L190;
	case 9:  goto L200;
	case 10:  goto L60;
	case 11:  goto L70;
	case 12:  goto L90;
	case 13:  goto L130;
	case 14:  goto L150;
	case 15:  goto L170;
    }
L10:
    switch (*ka) {
	case 1:  goto L11;
	case 2:  goto L12;
	case 3:  goto L13;
    }
L11:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    *fa = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - 1.;
    return 0;
L12:
    *fa = sin(x[1]);
    return 0;
L13:
    *fa = -cos(x[2]);
    return 0;
L30:
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L32;
	case 3:  goto L33;
    }
L31:
    *fa = -exp(x[1] - x[2]);
    return 0;
L32:
    *fa = sinh(x[1] - 1.) - 1.;
    return 0;
L33:
    *fa = -log(x[2]) - 1.;
    return 0;
L80:
    switch (*ka) {
	case 1:  goto L81;
	case 2:  goto L82;
	case 3:  goto L83;
    }
L81:
/* Computing 2nd power */
    d__1 = x[1] - x[3];
/* Computing 2nd power */
    d__2 = x[2] - x[4];
    *fa = -sqrt(d__1 * d__1 + d__2 * d__2);
    return 0;
L82:
/* Computing 2nd power */
    d__1 = x[3] - x[5];
/* Computing 2nd power */
    d__2 = x[4] - x[6];
    *fa = -sqrt(d__1 * d__1 + d__2 * d__2);
    return 0;
L83:
/* Computing 2nd power */
    d__1 = x[5] - x[1];
/* Computing 2nd power */
    d__2 = x[6] - x[2];
    *fa = -sqrt(d__1 * d__1 + d__2 * d__2);
    return 0;
L50:
    a = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	a += cos(empr22_1.y[*ka - 1] * x[i__]);
/* L51: */
    }
    *fa = (a * 2. + 1.) / 15.;
    return 0;
L100:
    *fa = 0.;
    k = 0;
    for (i__ = 1; i__ <= 5; ++i__) {
	a = 0.;
	p = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    d__1 = 1. - empr22_1.y[i__ + 79];
	    a += empr22_1.y[k + j - 1] * pow_dd(&x[j], &d__1);
	    p += empr22_1.y[k + j + 39] * x[j];
/* L101: */
	}
	*fa = *fa + empr22_1.y[k + *ka - 1] * p / (pow_dd(&x[*ka], &
		empr22_1.y[i__ + 79]) * a) - empr22_1.y[k + *ka + 39];
	k += *n;
/* L102: */
    }
    return 0;
L190:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3] - 10.;
/* Computing 2nd power */
    d__4 = x[4] - 5.;
/* Computing 2nd power */
    d__5 = x[5] - 3.;
/* Computing 2nd power */
    d__6 = x[6] - 1.;
/* Computing 2nd power */
    d__7 = x[7];
/* Computing 2nd power */
    d__8 = x[8] - 11.;
/* Computing 2nd power */
    d__9 = x[9] - 10.;
/* Computing 2nd power */
    d__10 = x[10] - 7.;
    *fa = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - x[1] * 14. - x[2] * 16. + 
	    d__3 * d__3 + d__4 * d__4 * 4. + d__5 * d__5 + d__6 * d__6 * 2. + 
	    d__7 * d__7 * 5. + d__8 * d__8 * 7. + d__9 * d__9 * 2. + d__10 * 
	    d__10 + 45.;
    switch (*ka) {
	case 1:  goto L191;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
    }
L191:
    return 0;
L192:
/* Computing 2nd power */
    d__1 = x[1] - 2.;
/* Computing 2nd power */
    d__2 = x[2] - 3.;
/* Computing 2nd power */
    d__3 = x[3];
    *fa += (d__1 * d__1 * 3. + d__2 * d__2 * 4. + d__3 * d__3 * 2. - x[4] * 
	    7. - 120.) * 10.;
    return 0;
L193:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[3] - 6.;
    *fa += (d__1 * d__1 * 5. + x[2] * 8. + d__2 * d__2 - x[4] * 2. - 40.) * 
	    10.;
    return 0;
L194:
/* Computing 2nd power */
    d__1 = x[1] - 8.;
/* Computing 2nd power */
    d__2 = x[2] - 4.;
/* Computing 2nd power */
    d__3 = x[5];
    *fa += (d__1 * d__1 * .5 + d__2 * d__2 * 2. + d__3 * d__3 * 3. - x[6] - 
	    30.) * 10.;
    return 0;
L195:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - 2.;
    *fa += (d__1 * d__1 + d__2 * d__2 * 2. - x[1] * 2. * x[2] + x[5] * 14. - 
	    x[6] * 6.) * 10.;
    return 0;
L196:
/* Computing 2nd power */
    d__1 = x[9] - 8.;
    *fa += (x[2] * 6. - x[1] * 3. + d__1 * d__1 * 12. - x[10] * 7.) * 10.;
    return 0;
L200:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3] - 10.;
/* Computing 2nd power */
    d__4 = x[4] - 5.;
/* Computing 2nd power */
    d__5 = x[5] - 3.;
/* Computing 2nd power */
    d__6 = x[6] - 1.;
/* Computing 2nd power */
    d__7 = x[7];
/* Computing 2nd power */
    d__8 = x[8] - 11.;
/* Computing 2nd power */
    d__9 = x[9] - 10.;
/* Computing 2nd power */
    d__10 = x[10] - 7.;
/* Computing 2nd power */
    d__11 = x[11] - 9.;
/* Computing 2nd power */
    d__12 = x[12] - 1.;
/* Computing 2nd power */
    d__13 = x[13] - 7.;
/* Computing 2nd power */
    d__14 = x[14] - 14.;
/* Computing 2nd power */
    d__15 = x[15] - 1.;
/* Computing 4th power */
    d__16 = x[16], d__16 *= d__16;
/* Computing 2nd power */
    d__17 = x[17] - 2.;
/* Computing 2nd power */
    d__18 = x[18] - 2.;
/* Computing 2nd power */
    d__19 = x[19] - 3.;
/* Computing 2nd power */
    d__20 = x[20];
    *fa = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - x[1] * 14. - x[2] * 16. + 
	    d__3 * d__3 + d__4 * d__4 * 4. + d__5 * d__5 + d__6 * d__6 * 2. + 
	    d__7 * d__7 * 5. + d__8 * d__8 * 7. + d__9 * d__9 * 2. + d__10 * 
	    d__10 + d__11 * d__11 + d__12 * d__12 * 10. + d__13 * d__13 * 5. 
	    + d__14 * d__14 * 4. + d__15 * d__15 * 27. + d__16 * d__16 + 
	    d__17 * d__17 + d__18 * d__18 * 13. + d__19 * d__19 + d__20 * 
	    d__20 + 95.;
    switch (*ka) {
	case 1:  goto L191;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
	case 7:  goto L202;
	case 8:  goto L203;
	case 9:  goto L204;
	case 10:  goto L205;
	case 11:  goto L206;
	case 12:  goto L207;
	case 13:  goto L208;
	case 14:  goto L209;
    }
L202:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 + x[11] * 15. - x[12] * 8. - 28.) * 10.;
    return 0;
L203:
/* Computing 2nd power */
    d__1 = x[13];
    *fa += (x[1] * 4. + x[2] * 9. + d__1 * d__1 * 5. - x[14] * 9. - 87.) * 
	    10.;
    return 0;
L204:
/* Computing 2nd power */
    d__1 = x[13] - 6.;
    *fa += (x[1] * 3. + x[2] * 4. + d__1 * d__1 * 3. - x[14] * 14. - 10.) * 
	    10.;
    return 0;
L205:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 * 14. + x[15] * 35. - x[16] * 79. - 92.) * 10.;
    return 0;
L206:
/* Computing 2nd power */
    d__1 = x[2];
    *fa += (d__1 * d__1 * 15. + x[15] * 11. - x[16] * 61. - 54.) * 10.;
    return 0;
L207:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 4th power */
    d__2 = x[17], d__2 *= d__2;
    *fa += (d__1 * d__1 * 5. + x[2] * 2. + d__2 * d__2 * 9. - x[18] - 68.) * 
	    10.;
    return 0;
L208:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 - x[2] + x[19] * 19. - x[20] * 20. + 19.) * 10.;
    return 0;
L209:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[19];
    *fa += (d__1 * d__1 * 7. + d__2 * d__2 * 5. + d__3 * d__3 - x[20] * 30.) *
	     10.;
    return 0;
L60:
    *fa = -1.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	*fa += x[i__];
/* L61: */
    }
    if (*ka % 2 == 0) {
	i__ = (*ka + 2) / 2;
	*fa += x[i__] * (x[i__] - 1.);
    } else {
	i__ = (*ka + 1) / 2;
	*fa += x[i__] * (x[i__] * 2. - 1.);
    }
    return 0;
L70:
    k = (*n - 1) / 2;
    a = empr22_1.y[*ka - 1];
    s = 1.;
    if (*ka > 62 && *ka < 124) {
	s = -s;
    }
    p = log(a) * -8.;
    a *= a;
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__ + k];
	b = d__1 * d__1 - a;
/* Computing 2nd power */
	d__1 = x[i__];
	p += log(b * b + a * (d__1 * d__1));
/* L71: */
    }
    *fa = (p * .5 - x[*n]) * s;
    if (*ka == 1 || *ka == 124) {
	*fa += 3.0164;
    }
    return 0;
L90:
    p = 500.;
    a = .99;
    *fa = x[1] * 5.04 + x[2] * .035 + x[3] * 10. + x[5] * 3.36 - x[4] * .063 *
	     x[7];
    switch (*ka) {
	case 1:  goto L91;
	case 2:  goto L92;
	case 3:  goto L93;
	case 4:  goto L94;
	case 5:  goto L95;
	case 6:  goto L96;
	case 7:  goto L97;
	case 8:  goto L98;
	case 9:  goto L99;
    }
L91:
    return 0;
L92:
    *fa += p * (x[1] * 1.12 + x[1] * x[8] * (.13167 - x[8] * .00667) - x[4] / 
	    a);
    return 0;
L93:
    *fa -= p * (x[1] * 1.12 + x[1] * x[8] * (.13167 - x[8] * .00667) - x[4] * 
	    a);
    return 0;
L94:
    *fa += p * (x[8] * (1.098 - x[8] * .038) + 57.425 + x[6] * .325 - x[7] / 
	    a);
    return 0;
L95:
    *fa -= p * (x[8] * (1.098 - x[8] * .038) + 57.425 + x[6] * .325 - x[7] * 
	    a);
    return 0;
L96:
    *fa += p * (x[3] * 9.8e4 / (x[4] * x[9] + x[3] * 1e3) - x[6]);
    return 0;
L97:
    *fa -= p * (x[3] * 9.8e4 / (x[4] * x[9] + x[3] * 1e3) - x[6]);
    return 0;
L98:
    *fa += p * ((x[2] + x[5]) / x[1] - x[8]);
    return 0;
L99:
    *fa -= p * ((x[2] + x[5]) / x[1] - x[8]);
    return 0;
L130:
    p = 1e5;
    *fa = empr22_1.y[0] * x[1] + empr22_1.y[1] * x[1] * x[6] + empr22_1.y[2] *
	     x[3] + empr22_1.y[3] * x[2] + empr22_1.y[4] + empr22_1.y[5] * x[
	    3] * x[5];
    switch (*ka) {
	case 1:  goto L131;
	case 2:  goto L132;
	case 3:  goto L133;
	case 4:  goto L134;
	case 5:  goto L135;
	case 6:  goto L136;
	case 7:  goto L137;
	case 8:  goto L138;
	case 9:  goto L105;
	case 10:  goto L106;
	case 11:  goto L107;
	case 12:  goto L108;
	case 13:  goto L109;
    }
L131:
    return 0;
L132:
/* Computing 2nd power */
    d__1 = x[6];
    *fa += p * (empr22_1.y[6] * (d__1 * d__1) + empr22_1.y[7] * x[3] / x[1] + 
	    empr22_1.y[8] * x[6] - 1.);
    return 0;
L133:
/* Computing 2nd power */
    d__1 = x[6];
    *fa += p * ((empr22_1.y[9] + empr22_1.y[10] * x[6] + empr22_1.y[11] * (
	    d__1 * d__1)) * x[1] / x[3] - 1.);
    return 0;
L134:
/* Computing 2nd power */
    d__1 = x[6];
    *fa += p * (empr22_1.y[12] * (d__1 * d__1) + empr22_1.y[13] * x[5] + 
	    empr22_1.y[14] * x[4] + empr22_1.y[15] * x[6] - 1.);
    return 0;
L135:
/* Computing 2nd power */
    d__1 = x[6];
    *fa += p * ((empr22_1.y[16] + empr22_1.y[17] * x[6] + empr22_1.y[18] * x[
	    4] + empr22_1.y[19] * (d__1 * d__1)) / x[5] - 1.);
    return 0;
L136:
    *fa += p * (empr22_1.y[20] * x[7] + (empr22_1.y[21] / x[4] + empr22_1.y[
	    22]) * x[2] / x[3] - 1.);
    return 0;
L137:
    *fa += p * ((empr22_1.y[23] + (empr22_1.y[24] + empr22_1.y[25] / x[4]) * 
	    x[2] / x[3]) / x[7] - 1.);
    return 0;
L138:
    *fa += p * ((empr22_1.y[26] + empr22_1.y[27] * x[7]) / x[5] - 1.);
    return 0;
L105:
    *fa += p * ((empr22_1.y[32] * x[1] + empr22_1.y[33]) / x[3] - 1.);
    return 0;
L106:
    *fa += p * ((empr22_1.y[34] / x[4] + empr22_1.y[35]) * x[2] / x[3] - 1.);
    return 0;
L107:
    *fa += p * ((empr22_1.y[36] + empr22_1.y[37] * x[3] / x[2]) * x[4] - 1.);
    return 0;
L108:
    *fa += p * (empr22_1.y[38] * x[1] * x[6] + empr22_1.y[39] * x[1] + 
	    empr22_1.y[40] * x[3] - 1.);
    return 0;
L109:
    *fa += p * ((empr22_1.y[41] * x[3] + empr22_1.y[42]) / x[1] + empr22_1.y[
	    43] * x[6] - 1.);
    return 0;
L150:
    p = 1e5;
    *fa = x[1] + x[2] + x[3];
    switch (*ka) {
	case 1:  goto L151;
	case 2:  goto L152;
	case 3:  goto L153;
	case 4:  goto L154;
    }
L151:
    return 0;
L152:
    *fa += p * ((x[4] * 833.33252 / x[1] + 100. - 83333.333 / x[1]) / x[6] - 
	    1.);
    return 0;
L153:
    *fa += p * (((x[5] - x[4]) * 1250. / x[2] + x[4]) / x[7] - 1.);
    return 0;
L154:
    *fa += p * (((1.25e6 - x[5] * 2500.) / x[3] + x[5]) / x[8] - 1.);
    return 0;
L170:
    p = 1e3;
    *fa = (x[12] + x[13] + x[14] + x[15] + x[16]) * 1.262626 - (x[1] * x[12] 
	    + x[2] * x[13] + x[3] * x[14] + x[4] * x[15] + x[5] * x[16]) * 
	    1.23106;
    switch (*ka) {
	case 1:  goto L171;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
	case 5:  goto L175;
	case 6:  goto L176;
	case 7:  goto L177;
	case 8:  goto L178;
	case 9:  goto L179;
	case 10:  goto L116;
	case 11:  goto L117;
	case 12:  goto L118;
	case 13:  goto L128;
	case 14:  goto L129;
	case 15:  goto L146;
	case 16:  goto L147;
	case 17:  goto L148;
	case 18:  goto L149;
	case 19:  goto L159;
    }
L171:
    return 0;
L172:
    *fa += p * (x[1] * ((.03475 - x[1] * .00975) / x[6] + .975) - 1.);
    return 0;
L173:
    *fa += p * (x[2] * ((.03475 - x[2] * .00975) / x[7] + .975) - 1.);
    return 0;
L174:
    *fa += p * (x[3] * ((.03475 - x[3] * .00975) / x[8] + .975) - 1.);
    return 0;
L175:
    *fa += p * (x[4] * ((.03475 - x[4] * .00975) / x[9] + .975) - 1.);
    return 0;
L176:
    *fa += p * (x[5] * ((.03475 - x[5] * .00975) / x[10] + .975) - 1.);
    return 0;
L177:
    *fa += p * ((x[6] + (x[1] - x[6]) * x[12] / x[11]) / x[7] - 1.);
    return 0;
L178:
    *fa += p * ((x[7] + (x[7] - x[1]) * .002 * x[12]) / x[8] + x[13] * .002 * 
	    (x[2] / x[8] - 1.) - 1.);
    return 0;
L179:
    *fa += p * (x[8] + x[9] + x[13] * .002 * (x[8] - x[2]) + x[14] * .002 * (
	    x[3] - x[9]) - 1.);
    return 0;
L116:
    *fa += p * ((x[9] + ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) / x[
	    14]) / x[3] - 1.);
    return 0;
L117:
    *fa += p * (x[10] / x[4] + (x[5] / x[4] - 1.) * x[16] / x[15] + (1. - x[
	    10] / x[4]) * 500. / x[15] - 1.);
    return 0;
L118:
    *fa += p * (.9 / x[4] + x[16] * .002 * (1. - x[5] / x[4]) - 1.);
    return 0;
L128:
    *fa += p * (x[12] / x[11] - 1.);
    return 0;
L129:
    *fa += p * (x[4] / x[5] - 1.);
    return 0;
L146:
    *fa += p * (x[3] / x[4] - 1.);
    return 0;
L147:
    *fa += p * (x[2] / x[3] - 1.);
    return 0;
L148:
    *fa += p * (x[1] / x[2] - 1.);
    return 0;
L149:
    *fa += p * (x[9] / x[10] - 1.);
    return 0;
L159:
    *fa = *fa + p * x[8] / x[9] - p;
    return 0;
} /* tafu22_ */

/* SUBROUTINE TAGU22             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 94/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  GRADIENTS OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  GA(N)  GRADIENT OF THE PARTIAL FUNCTION AT THE */
/*          SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tagu22_(integer *n, integer *ka, doublereal *x, 
	doublereal *ga, integer *next)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double cos(doublereal), sin(doublereal), exp(doublereal), cosh(doublereal)
	    , sqrt(doublereal), pow_dd(doublereal *, doublereal *);

    /* Local variables */
    static doublereal a, b, c__;
    static integer i__, j, k;
    static doublereal p, s;

    /* Parameter adjustments */
    --ga;
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L10;
	case 3:  goto L30;
	case 4:  goto L30;
	case 5:  goto L80;
	case 6:  goto L50;
	case 7:  goto L100;
	case 8:  goto L190;
	case 9:  goto L200;
	case 10:  goto L60;
	case 11:  goto L70;
	case 12:  goto L90;
	case 13:  goto L130;
	case 14:  goto L150;
	case 15:  goto L170;
    }
L10:
    switch (*ka) {
	case 1:  goto L11;
	case 2:  goto L12;
	case 3:  goto L13;
    }
L11:
    ga[1] = x[1] * 2. + x[2];
    ga[2] = x[2] * 2. + x[1];
    return 0;
L12:
    ga[1] = cos(x[1]);
    ga[2] = 0.;
    return 0;
L13:
    ga[1] = 0.;
    ga[2] = sin(x[2]);
    return 0;
L30:
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L32;
	case 3:  goto L33;
    }
L31:
    ga[1] = -exp(x[1] - x[2]);
    ga[2] = exp(x[1] - x[2]);
    return 0;
L32:
    ga[1] = cosh(x[1] - 1.);
    ga[2] = 0.;
    return 0;
L33:
    ga[1] = 0.;
    ga[2] = -1. / x[2];
    return 0;
L80:
    switch (*ka) {
	case 1:  goto L81;
	case 2:  goto L82;
	case 3:  goto L83;
    }
L81:
/* Computing 2nd power */
    d__1 = x[1] - x[3];
/* Computing 2nd power */
    d__2 = x[2] - x[4];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    ga[1] = -(x[1] - x[3]) / a;
    ga[2] = -(x[2] - x[4]) / a;
    ga[3] = -ga[1];
    ga[4] = -ga[2];
    ga[5] = 0.;
    ga[6] = 0.;
    return 0;
L82:
/* Computing 2nd power */
    d__1 = x[3] - x[5];
/* Computing 2nd power */
    d__2 = x[4] - x[6];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    ga[1] = 0.;
    ga[2] = 0.;
    ga[3] = -(x[3] - x[5]) / a;
    ga[4] = -(x[4] - x[6]) / a;
    ga[5] = -ga[3];
    ga[6] = -ga[4];
    return 0;
L83:
/* Computing 2nd power */
    d__1 = x[5] - x[1];
/* Computing 2nd power */
    d__2 = x[6] - x[2];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    ga[1] = (x[5] - x[1]) / a;
    ga[2] = (x[6] - x[2]) / a;
    ga[3] = 0.;
    ga[4] = 0.;
    ga[5] = -ga[1];
    ga[6] = -ga[2];
    return 0;
L50:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ga[i__] = empr22_1.y[*ka - 1] * -2. * sin(empr22_1.y[*ka - 1] * x[i__]
		) / 15.;
/* L51: */
    }
    return 0;
L100:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ga[i__] = 0.;
/* L189: */
    }
    k = 0;
    for (i__ = 1; i__ <= 5; ++i__) {
	a = 0.;
	p = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    d__1 = 1. - empr22_1.y[i__ + 79];
	    a += empr22_1.y[k + j - 1] * pow_dd(&x[j], &d__1);
	    p += empr22_1.y[k + j + 39] * x[j];
/* L101: */
	}
	b = empr22_1.y[k + *ka - 1] / (pow_dd(&x[*ka], &empr22_1.y[i__ + 79]) 
		* a);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    c__ = empr22_1.y[k + j - 1] * (1. - empr22_1.y[i__ + 79]) / (
		    pow_dd(&x[j], &empr22_1.y[i__ + 79]) * a);
	    ga[j] += b * (empr22_1.y[k + j + 39] - c__ * p);
/* L102: */
	}
	ga[*ka] -= b * empr22_1.y[i__ + 79] * p / x[*ka];
	k += *n;
/* L103: */
    }
    return 0;
L190:
    ga[1] = x[1] * 2. + x[2] - 14.;
    ga[2] = x[2] * 2. + x[1] - 16.;
    ga[3] = (x[3] - 10.) * 2.;
    ga[4] = (x[4] - 5.) * 8.;
    ga[5] = (x[5] - 3.) * 2.;
    ga[6] = (x[6] - 1.) * 4.;
    ga[7] = x[7] * 10.;
    ga[8] = (x[8] - 11.) * 14.;
    ga[9] = (x[9] - 10.) * 4.;
    ga[10] = (x[10] - 7.) * 2.;
    switch (*ka) {
	case 1:  goto L191;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
    }
L191:
    return 0;
L192:
    ga[1] += (x[1] - 2.) * 60.;
    ga[2] += (x[2] - 3.) * 80.;
    ga[3] += x[3] * 40.;
    ga[4] += -70.;
    return 0;
L193:
    ga[1] += x[1] * 100.;
    ga[2] += 80.;
    ga[3] += (x[3] - 6.) * 20.;
    ga[4] += -20.;
    return 0;
L194:
    ga[1] += (x[1] - 8.) * 10.;
    ga[2] += (x[2] - 4.) * 40.;
    ga[5] += x[5] * 60.;
    ga[6] += -10.;
    return 0;
L195:
    ga[1] = ga[1] + x[1] * 20. - x[2] * 20.;
    ga[2] = ga[2] + (x[2] - 2.) * 40. - x[1] * 20.;
    ga[5] += 140.;
    ga[6] += -60.;
    return 0;
L196:
    ga[1] += -30.;
    ga[2] += 60.;
    ga[9] += (x[9] - 8.) * 240.;
    ga[10] += -70.;
    return 0;
L200:
    ga[1] = x[1] * 2. + x[2] - 14.;
    ga[2] = x[2] * 2. + x[1] - 16.;
    ga[3] = (x[3] - 10.) * 2.;
    ga[4] = (x[4] - 5.) * 8.;
    ga[5] = (x[5] - 3.) * 2.;
    ga[6] = (x[6] - 1.) * 4.;
    ga[7] = x[7] * 10.;
    ga[8] = (x[8] - 11.) * 14.;
    ga[9] = (x[9] - 10.) * 4.;
    ga[10] = (x[10] - 7.) * 2.;
    ga[11] = (x[11] - 9.) * 2.;
    ga[12] = (x[12] - 1.) * 20.;
    ga[13] = (x[13] - 7.) * 10.;
    ga[14] = (x[14] - 14.) * 8.;
    ga[15] = (x[15] - 1.) * 54.;
/* Computing 3rd power */
    d__1 = x[16];
    ga[16] = d__1 * (d__1 * d__1) * 4.;
    ga[17] = (x[17] - 2.) * 2.;
    ga[18] = (x[18] - 2.) * 26.;
    ga[19] = (x[19] - 3.) * 2.;
    ga[20] = x[20] * 2.;
    switch (*ka) {
	case 1:  goto L191;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
	case 7:  goto L202;
	case 8:  goto L203;
	case 9:  goto L204;
	case 10:  goto L205;
	case 11:  goto L206;
	case 12:  goto L207;
	case 13:  goto L208;
	case 14:  goto L209;
    }
L202:
    ga[1] += x[1] * 20.;
    ga[11] += 150.;
    ga[12] += -80.;
    return 0;
L203:
    ga[1] += 40.;
    ga[2] += 90.;
    ga[13] += x[13] * 100.;
    ga[14] += -90.;
    return 0;
L204:
    ga[1] += 30.;
    ga[2] += 40.;
    ga[13] += (x[13] - 6.) * 60.;
    ga[14] += -140.;
    return 0;
L205:
    ga[1] += x[1] * 280.;
    ga[15] += 350.;
    ga[16] += -790.;
    return 0;
L206:
    ga[2] += x[2] * 300.;
    ga[15] += 110.;
    ga[16] += -610.;
    return 0;
L207:
    ga[1] += x[1] * 100.;
    ga[2] += 20.;
/* Computing 3rd power */
    d__1 = x[17];
    ga[17] += d__1 * (d__1 * d__1) * 360.;
    ga[18] += -10.;
    return 0;
L208:
    ga[1] += x[1] * 20.;
    ga[2] += -10.;
    ga[19] += 190.;
    ga[20] += -200.;
    return 0;
L209:
    ga[1] += x[1] * 140.;
    ga[2] += x[2] * 100.;
    ga[19] += x[19] * 20.;
    ga[20] += -300.;
    return 0;
L60:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ga[i__] = 1.;
/* L61: */
    }
    if (*ka % 2 == 0) {
	i__ = (*ka + 2) / 2;
	ga[i__] = ga[i__] + x[i__] * 2. - 1.;
    } else {
	i__ = (*ka + 1) / 2;
	ga[i__] = ga[i__] + x[i__] * 4. - 1.;
    }
    return 0;
L70:
    k = (*n - 1) / 2;
/* Computing 2nd power */
    d__1 = empr22_1.y[*ka - 1];
    a = d__1 * d__1;
    s = 1.;
    if (*ka > 62 && *ka < 124) {
	s = -s;
    }
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__ + k];
	b = d__1 * d__1 - a;
/* Computing 2nd power */
	d__1 = x[i__];
	p = s * (b * b + a * (d__1 * d__1));
	ga[i__] = a * x[i__] / p;
	ga[i__ + k] = x[i__ + k] * 2. * b / p;
/* L71: */
    }
    ga[*n] = -s;
    return 0;
L90:
    p = 500.;
    a = .99;
    ga[1] = 5.04;
    ga[2] = .035;
    ga[3] = 10.;
    ga[4] = x[7] * -.063;
    ga[5] = 3.36;
    ga[6] = 0.;
    ga[7] = x[4] * -.063;
    ga[8] = 0.;
    ga[9] = 0.;
    ga[10] = 0.;
    switch (*ka) {
	case 1:  goto L91;
	case 2:  goto L92;
	case 3:  goto L93;
	case 4:  goto L94;
	case 5:  goto L95;
	case 6:  goto L96;
	case 7:  goto L97;
	case 8:  goto L98;
	case 9:  goto L99;
    }
L91:
    return 0;
L92:
    ga[1] += p * (x[8] * (.13167 - x[8] * .00667) + 1.12);
    ga[4] -= p / a;
    ga[8] += p * x[1] * (.13167 - x[8] * .01334);
    return 0;
L93:
    ga[1] -= p * (x[8] * (.13167 - x[8] * .00667) + 1.12);
    ga[4] += p * a;
    ga[8] -= p * x[1] * (.13167 - x[8] * .01334);
    return 0;
L94:
    ga[6] += p * .325;
    ga[7] -= p / a;
    ga[8] += p * (1.098 - x[8] * .076);
    return 0;
L95:
    ga[6] -= p * .325;
    ga[7] += p * a;
    ga[8] -= p * (1.098 - x[8] * .076);
    return 0;
L96:
/* Computing 2nd power */
    d__1 = x[4] * x[9] + x[3] * 1e3;
    c__ = d__1 * d__1;
    ga[3] += p * 9.8e4 * x[4] * x[9] / c__;
    ga[4] -= p * 9.8e4 * x[3] * x[9] / c__;
    ga[6] -= p;
    ga[9] -= p * 9.8e4 * x[3] * x[4] / c__;
    return 0;
L97:
/* Computing 2nd power */
    d__1 = x[4] * x[9] + x[3] * 1e3;
    c__ = d__1 * d__1;
    ga[3] -= p * 9.8e4 * x[4] * x[9] / c__;
    ga[4] += p * 9.8e4 * x[3] * x[9] / c__;
    ga[6] += p;
    ga[9] += p * 9.8e4 * x[3] * x[4] / c__;
    return 0;
L98:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] -= p * (x[2] + x[5]) / (d__1 * d__1);
    ga[2] += p / x[1];
    ga[5] += p / x[1];
    ga[8] -= p;
    return 0;
L99:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] += p * (x[2] + x[5]) / (d__1 * d__1);
    ga[2] -= p / x[1];
    ga[5] -= p / x[1];
    ga[8] += p;
    return 0;
L130:
    p = 1e5;
    ga[1] = empr22_1.y[0] + empr22_1.y[1] * x[6];
    ga[2] = empr22_1.y[3];
    ga[3] = empr22_1.y[2] + empr22_1.y[5] * x[5];
    ga[4] = 0.;
    ga[5] = empr22_1.y[5] * x[3];
    ga[6] = empr22_1.y[1] * x[1];
    ga[7] = 0.;
    switch (*ka) {
	case 1:  goto L131;
	case 2:  goto L132;
	case 3:  goto L133;
	case 4:  goto L134;
	case 5:  goto L135;
	case 6:  goto L136;
	case 7:  goto L137;
	case 8:  goto L138;
	case 9:  goto L105;
	case 10:  goto L106;
	case 11:  goto L107;
	case 12:  goto L108;
	case 13:  goto L109;
    }
L131:
    return 0;
L132:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] -= p * empr22_1.y[7] * x[3] / (d__1 * d__1);
    ga[3] += p * empr22_1.y[7] / x[1];
    ga[6] += p * (empr22_1.y[6] * 2. * x[6] + empr22_1.y[8]);
    return 0;
L133:
/* Computing 2nd power */
    d__1 = x[6];
    ga[1] += p * (empr22_1.y[9] + empr22_1.y[10] * x[6] + empr22_1.y[11] * (
	    d__1 * d__1)) / x[3];
/* Computing 2nd power */
    d__1 = x[6];
/* Computing 2nd power */
    d__2 = x[3];
    ga[3] -= p * (empr22_1.y[9] + empr22_1.y[10] * x[6] + empr22_1.y[11] * (
	    d__1 * d__1)) * x[1] / (d__2 * d__2);
    ga[6] += p * (empr22_1.y[10] + empr22_1.y[11] * 2. * x[6]) * x[1] / x[3];
    return 0;
L134:
    ga[4] += p * empr22_1.y[14];
    ga[5] += p * empr22_1.y[13];
    ga[6] += p * (empr22_1.y[12] * 2. * x[6] + empr22_1.y[15]);
    return 0;
L135:
    ga[4] += p * empr22_1.y[18] / x[5];
/* Computing 2nd power */
    d__1 = x[6];
/* Computing 2nd power */
    d__2 = x[5];
    ga[5] -= p * (empr22_1.y[16] + empr22_1.y[17] * x[6] + empr22_1.y[18] * x[
	    4] + empr22_1.y[19] * (d__1 * d__1)) / (d__2 * d__2);
    ga[6] += p * (empr22_1.y[17] + empr22_1.y[19] * 2. * x[6]) / x[5];
    return 0;
L136:
    ga[2] += p * (empr22_1.y[21] / x[4] + empr22_1.y[22]) / x[3];
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (empr22_1.y[21] / x[4] + empr22_1.y[22]) * x[2] / (d__1 * 
	    d__1);
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * empr22_1.y[21] * x[2] / (x[3] * (d__1 * d__1));
    ga[7] += p * empr22_1.y[20];
    return 0;
L137:
    ga[2] += p * (empr22_1.y[24] + empr22_1.y[25] / x[4]) / (x[3] * x[7]);
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (empr22_1.y[24] + empr22_1.y[25] / x[4]) * x[2] / (d__1 * 
	    d__1 * x[7]);
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * empr22_1.y[25] * x[2] / (x[3] * (d__1 * d__1) * x[7]);
/* Computing 2nd power */
    d__1 = x[7];
    ga[7] -= p * (empr22_1.y[23] + (empr22_1.y[24] + empr22_1.y[25] / x[4]) * 
	    x[2] / x[3]) / (d__1 * d__1);
    return 0;
L138:
/* Computing 2nd power */
    d__1 = x[5];
    ga[5] -= p * (empr22_1.y[26] + empr22_1.y[27] * x[7]) / (d__1 * d__1);
    ga[7] += p * empr22_1.y[27] / x[5];
    return 0;
L105:
    ga[1] += p * empr22_1.y[32] / x[3];
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (empr22_1.y[32] * x[1] + empr22_1.y[33]) / (d__1 * d__1);
    return 0;
L106:
    ga[2] += p * (empr22_1.y[34] / x[4] + empr22_1.y[35]) / x[3];
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (empr22_1.y[34] / x[4] + empr22_1.y[35]) * x[2] / (d__1 * 
	    d__1);
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * empr22_1.y[34] * x[2] / (x[3] * (d__1 * d__1));
    return 0;
L107:
/* Computing 2nd power */
    d__1 = x[2];
    ga[2] -= p * empr22_1.y[37] * x[3] * x[4] / (d__1 * d__1);
    ga[3] += p * empr22_1.y[37] * x[4] / x[2];
    ga[4] += p * (empr22_1.y[36] + empr22_1.y[37] * x[3] / x[2]);
    return 0;
L108:
    ga[1] += p * (empr22_1.y[38] * x[6] + empr22_1.y[39]);
    ga[3] += p * empr22_1.y[40];
    ga[6] += p * empr22_1.y[38] * x[1];
    return 0;
L109:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] -= p * (empr22_1.y[41] * x[3] + empr22_1.y[42]) / (d__1 * d__1);
    ga[3] += p * empr22_1.y[41] / x[1];
    ga[6] += p * empr22_1.y[43];
    return 0;
L150:
    p = 1e5;
    ga[1] = 1.;
    ga[2] = 1.;
    ga[3] = 1.;
    ga[4] = 0.;
    ga[5] = 0.;
    ga[6] = 0.;
    ga[7] = 0.;
    ga[8] = 0.;
    switch (*ka) {
	case 1:  goto L151;
	case 2:  goto L152;
	case 3:  goto L153;
	case 4:  goto L154;
    }
L151:
    return 0;
L152:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] -= p * (x[4] * 833.33252 - 83333.333) / (d__1 * d__1 * x[6]);
    ga[4] += p * 833.33252 / (x[1] * x[6]);
/* Computing 2nd power */
    d__1 = x[6];
    ga[6] -= p * (x[4] * 833.33252 / x[1] + 100. - 83333.333 / x[1]) / (d__1 *
	     d__1);
    return 0;
L153:
/* Computing 2nd power */
    d__1 = x[2];
    ga[2] -= p * 1250. * (x[5] - x[4]) / (d__1 * d__1 * x[7]);
    ga[4] += p * (1. - 1250. / x[2]) / x[7];
    ga[5] += p * 1250. / (x[2] * x[7]);
/* Computing 2nd power */
    d__1 = x[7];
    ga[7] -= p * ((x[5] - x[4]) * 1250. / x[2] + x[4]) / (d__1 * d__1);
    return 0;
L154:
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (1.25e6 - x[5] * 2500.) / (d__1 * d__1 * x[8]);
    ga[5] += p * (1. - 2500. / x[3]) / x[8];
/* Computing 2nd power */
    d__1 = x[8];
    ga[8] -= p * ((1.25e6 - x[5] * 2500.) / x[3] + x[5]) / (d__1 * d__1);
    return 0;
L170:
    p = 1e3;
    ga[1] = x[12] * -1.23106;
    ga[2] = x[13] * -1.23106;
    ga[3] = x[14] * -1.23106;
    ga[4] = x[15] * -1.23106;
    ga[5] = x[16] * -1.23106;
    ga[6] = 0.;
    ga[7] = 0.;
    ga[8] = 0.;
    ga[9] = 0.;
    ga[10] = 0.;
    ga[11] = 0.;
    ga[12] = 1.262626 - x[1] * 1.23106;
    ga[13] = 1.262626 - x[2] * 1.23106;
    ga[14] = 1.262626 - x[3] * 1.23106;
    ga[15] = 1.262626 - x[4] * 1.23106;
    ga[16] = 1.262626 - x[5] * 1.23106;
    switch (*ka) {
	case 1:  goto L171;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
	case 5:  goto L175;
	case 6:  goto L176;
	case 7:  goto L177;
	case 8:  goto L178;
	case 9:  goto L179;
	case 10:  goto L116;
	case 11:  goto L117;
	case 12:  goto L118;
	case 13:  goto L128;
	case 14:  goto L129;
	case 15:  goto L146;
	case 16:  goto L147;
	case 17:  goto L148;
	case 18:  goto L149;
	case 19:  goto L159;
    }
L171:
    return 0;
L172:
    ga[1] += p * ((.03475 - x[1] * .0195) / x[6] + .975);
/* Computing 2nd power */
    d__1 = x[6];
    ga[6] -= p * x[1] * (.03475 - x[1] * .00975) / (d__1 * d__1);
    return 0;
L173:
    ga[2] += p * ((.03475 - x[2] * .0195) / x[7] + .975);
/* Computing 2nd power */
    d__1 = x[7];
    ga[7] -= p * x[2] * (.03475 - x[2] * .00975) / (d__1 * d__1);
    return 0;
L174:
    ga[3] += p * ((.03475 - x[3] * .0195) / x[8] + .975);
/* Computing 2nd power */
    d__1 = x[8];
    ga[8] -= p * x[3] * (.03475 - x[3] * .00975) / (d__1 * d__1);
    return 0;
L175:
    ga[4] += p * ((.03475 - x[4] * .0195) / x[9] + .975);
/* Computing 2nd power */
    d__1 = x[9];
    ga[9] -= p * x[4] * (.03475 - x[4] * .00975) / (d__1 * d__1);
    return 0;
L176:
    ga[5] += p * ((.03475 - x[5] * .0195) / x[10] + .975);
/* Computing 2nd power */
    d__1 = x[10];
    ga[10] -= p * x[5] * (.03475 - x[5] * .00975) / (d__1 * d__1);
    return 0;
L177:
    ga[1] += p * x[12] / (x[7] * x[11]);
    ga[6] += p * (1. - x[12] / x[11]) / x[7];
/* Computing 2nd power */
    d__1 = x[7];
    ga[7] -= p * (x[6] + (x[1] - x[6]) * x[12] / x[11]) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[11];
    ga[11] -= p * (x[1] - x[6]) * x[12] / (x[7] * (d__1 * d__1));
    ga[12] += p * (x[1] - x[6]) / (x[11] * x[7]);
    return 0;
L178:
    ga[1] -= p * .002 * x[12] / x[8];
    ga[2] += p * .002 * x[13] / x[8];
    ga[7] += p * (x[12] * .002 + 1.) / x[8];
/* Computing 2nd power */
    d__1 = x[8];
    ga[8] -= p * (x[7] + ((x[7] - x[1]) * x[12] + x[2] * x[13]) * .002) / (
	    d__1 * d__1);
    ga[12] += p * .002 * (x[7] - x[1]) / x[8];
    ga[13] += p * .002 * (x[2] / x[8] - 1.);
    return 0;
L179:
    ga[2] -= p * .002 * x[13];
    ga[3] += p * .002 * x[14];
    ga[8] += p * (x[13] * .002 + 1.);
    ga[9] += p * (1. - x[14] * .002);
    ga[13] += p * .002 * (x[8] - x[2]);
    ga[14] += p * .002 * (x[3] - x[9]);
    return 0;
L116:
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * (x[9] + ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) / x[
	    14]) / (d__1 * d__1);
    ga[4] += p * x[15] / (x[3] * x[14]);
    ga[8] -= p * x[15] / (x[3] * x[14]);
    ga[9] += p * (1. - 500. / x[14]) / x[3];
    ga[10] += p * 500. / (x[3] * x[14]);
/* Computing 2nd power */
    d__1 = x[14];
    ga[14] -= p * ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) / (x[3] * (
	    d__1 * d__1));
    ga[15] += p * (x[4] - x[8]) / (x[3] * x[14]);
    return 0;
L117:
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * (x[10] + (x[5] * x[16] - x[10] * 500.) / x[15]) / (d__1 * 
	    d__1);
    ga[5] += p * x[16] / (x[4] * x[15]);
    ga[10] += p * (1. - 500. / x[15]) / x[4];
/* Computing 2nd power */
    d__1 = x[15];
    ga[15] -= p * (500. - x[16] + (x[5] * x[16] - x[10] * 500.) / x[4]) / (
	    d__1 * d__1);
    ga[16] += p * (x[5] / x[4] - 1.) / x[15];
    return 0;
L118:
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * (.9 - x[5] * .002 * x[16]) / (d__1 * d__1);
    ga[5] -= p * .002 * x[16] / x[4];
    ga[16] += p * .002 * (1. - x[5] / x[4]);
    return 0;
L128:
/* Computing 2nd power */
    d__1 = x[11];
    ga[11] -= p * x[12] / (d__1 * d__1);
    ga[12] += p / x[11];
    return 0;
L129:
    ga[4] += p / x[5];
/* Computing 2nd power */
    d__1 = x[5];
    ga[5] -= p * x[4] / (d__1 * d__1);
    return 0;
L146:
    ga[3] += p / x[4];
/* Computing 2nd power */
    d__1 = x[4];
    ga[4] -= p * x[3] / (d__1 * d__1);
    return 0;
L147:
    ga[2] += p / x[3];
/* Computing 2nd power */
    d__1 = x[3];
    ga[3] -= p * x[2] / (d__1 * d__1);
    return 0;
L148:
    ga[1] += p / x[2];
/* Computing 2nd power */
    d__1 = x[2];
    ga[2] -= p * x[1] / (d__1 * d__1);
    return 0;
L149:
    ga[9] += p / x[10];
/* Computing 2nd power */
    d__1 = x[10];
    ga[10] -= p * x[9] / (d__1 * d__1);
    return 0;
L159:
    ga[8] += p / x[9];
/* Computing 2nd power */
    d__1 = x[9];
    ga[9] -= p * x[8] / (d__1 * d__1);
    return 0;
} /* tagu22_ */

/* SUBROUTINE TAHD22             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 95/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  HESSIAN MATRICES OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */
/*  DENSE VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  HA(N*(N+1)/2)  GRADIENT OF THE PARTIAL FUNCTION */
/*         AT THE SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tahd22_(integer *n, integer *ka, doublereal *x, 
	doublereal *ha, integer *next)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sin(doublereal), cos(doublereal), exp(doublereal), sinh(doublereal)
	    , sqrt(doublereal), pow_dd(doublereal *, doublereal *);

    /* Local variables */
    static doublereal a, b, c__;
    static integer i__, j, k, l;
    static doublereal p, q, r__, s;
    static integer kk, ll;

    /* Parameter adjustments */
    --ha;
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L10;
	case 3:  goto L30;
	case 4:  goto L30;
	case 5:  goto L80;
	case 6:  goto L50;
	case 7:  goto L100;
	case 8:  goto L190;
	case 9:  goto L200;
	case 10:  goto L60;
	case 11:  goto L70;
	case 12:  goto L90;
	case 13:  goto L130;
	case 14:  goto L150;
	case 15:  goto L170;
    }
L10:
    switch (*ka) {
	case 1:  goto L11;
	case 2:  goto L12;
	case 3:  goto L13;
    }
L11:
    ha[1] = 2.;
    ha[2] = 1.;
    ha[3] = 2.;
    return 0;
L12:
    ha[1] = -sin(x[1]);
    ha[2] = 0.;
    ha[3] = 0.;
    return 0;
L13:
    ha[1] = 0.;
    ha[2] = 0.;
    ha[3] = cos(x[2]);
    return 0;
L30:
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L32;
	case 3:  goto L33;
    }
L31:
    ha[1] = -exp(x[1] - x[2]);
    ha[2] = exp(x[1] - x[2]);
    ha[3] = -exp(x[1] - x[2]);
    return 0;
L32:
    ha[1] = sinh(x[1] - 1.);
    ha[2] = 0.;
    ha[3] = 0.;
    return 0;
L33:
    ha[1] = 0.;
    ha[2] = 0.;
/* Computing 2nd power */
    d__1 = x[2];
    ha[3] = 1. / (d__1 * d__1);
    return 0;
L80:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L81: */
    }
    switch (*ka) {
	case 1:  goto L82;
	case 2:  goto L83;
	case 3:  goto L84;
    }
L82:
/* Computing 2nd power */
    d__1 = x[1] - x[3];
/* Computing 2nd power */
    d__2 = x[2] - x[4];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    b = (x[1] - x[3]) / a;
    c__ = (x[2] - x[4]) / a;
    ha[1] = -1. / a + b * b / a;
    ha[2] = b * c__ / a;
    ha[3] = -1. / a + c__ * c__ / a;
    ha[4] = -ha[1];
    ha[5] = -ha[2];
    ha[6] = ha[1];
    ha[7] = -ha[2];
    ha[8] = -ha[3];
    ha[9] = ha[3];
    ha[10] = ha[3];
    return 0;
L83:
/* Computing 2nd power */
    d__1 = x[3] - x[5];
/* Computing 2nd power */
    d__2 = x[4] - x[6];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    b = (x[3] - x[5]) / a;
    c__ = (x[4] - x[6]) / a;
    ha[6] = -1. / a + b * b / a;
    ha[9] = b * c__ / a;
    ha[10] = -1. / a + c__ * c__ / a;
    ha[13] = -ha[6];
    ha[14] = -ha[9];
    ha[15] = ha[6];
    ha[18] = -ha[9];
    ha[19] = -ha[10];
    ha[20] = ha[9];
    ha[21] = ha[10];
    return 0;
L84:
/* Computing 2nd power */
    d__1 = x[5] - x[1];
/* Computing 2nd power */
    d__2 = x[6] - x[2];
    a = sqrt(d__1 * d__1 + d__2 * d__2);
    b = (x[5] - x[1]) / a;
    c__ = (x[6] - x[2]) / a;
    ha[1] = -1. / a + b * b / a;
    ha[2] = b * c__ / a;
    ha[3] = -1. / a + c__ * c__ / a;
    ha[11] = -ha[1];
    ha[12] = -ha[2];
    ha[15] = ha[1];
    ha[16] = -ha[2];
    ha[17] = -ha[3];
    ha[20] = ha[2];
    ha[21] = ha[3];
    return 0;
L50:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L51: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = i__ * (i__ + 1) / 2;
	ha[j] = empr22_1.y[*ka - 1] * -2. * empr22_1.y[*ka - 1] * cos(
		empr22_1.y[*ka - 1] * x[i__]) / 15.;
/* L52: */
    }
    return 0;
L100:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L101: */
    }
    k = 0;
    for (i__ = 1; i__ <= 5; ++i__) {
	a = 0.;
	p = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    d__1 = 1. - empr22_1.y[i__ + 79];
	    a += empr22_1.y[k + j - 1] * pow_dd(&x[j], &d__1);
	    p += empr22_1.y[k + j + 39] * x[j];
/* L102: */
	}
	b = empr22_1.y[k + *ka - 1] / (pow_dd(&x[*ka], &empr22_1.y[i__ + 79]) 
		* a);
	c__ = b * empr22_1.y[i__ + 79] / x[*ka];
	kk = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    q = empr22_1.y[k + j - 1] * (1. - empr22_1.y[i__ + 79]) / (pow_dd(
		    &x[j], &empr22_1.y[i__ + 79]) * a);
	    i__2 = j;
	    for (l = 1; l <= i__2; ++l) {
		r__ = empr22_1.y[k + l - 1] * (1. - empr22_1.y[i__ + 79]) / (
			pow_dd(&x[l], &empr22_1.y[i__ + 79]) * a);
		++kk;
		ha[kk] += b * (p * 2. * q * r__ - q * empr22_1.y[k + l + 39] 
			- r__ * empr22_1.y[k + j + 39]);
		if (j == l) {
		    ha[kk] += c__ * q * p;
		}
		if (l == *ka) {
		    ha[kk] -= c__ * (empr22_1.y[k + j + 39] - q * p);
		}
		if (j == *ka) {
		    ha[kk] -= c__ * (empr22_1.y[k + l + 39] - r__ * p);
		}
/* L103: */
	    }
/* L104: */
	}
	kk = *ka * (*ka + 1) / 2;
	q = empr22_1.y[k + *ka - 1] * (1. - empr22_1.y[i__ + 79]) / (pow_dd(&
		x[*ka], &empr22_1.y[i__ + 79]) * a);
	ha[kk] += c__ * p * (empr22_1.y[i__ + 79] + 1.) / x[*ka];
	k += *n;
/* L105: */
    }
    return 0;
L190:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L191: */
    }
    ha[1] = 2.;
    ha[2] = 1.;
    ha[3] = 2.;
    ha[6] = 2.;
    ha[10] = 8.;
    ha[15] = 2.;
    ha[21] = 4.;
    ha[28] = 10.;
    ha[36] = 14.;
    ha[45] = 4.;
    ha[55] = 2.;
    switch (*ka) {
	case 1:  goto L197;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
    }
L192:
    ha[1] += 60.;
    ha[3] += 80.;
    ha[6] += 40.;
    return 0;
L193:
    ha[1] += 100.;
    ha[6] += 20.;
    return 0;
L194:
    ha[1] += 10.;
    ha[3] += 40.;
    ha[15] += 60.;
    return 0;
L195:
    ha[1] += 10.;
    ha[2] += -20.;
    ha[3] += 40.;
    return 0;
L196:
    ha[45] += 240.;
L197:
    return 0;
L200:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L201: */
    }
    ha[1] = 2.;
    ha[2] = 1.;
    ha[3] = 2.;
    ha[6] = 2.;
    ha[10] = 8.;
    ha[15] = 2.;
    ha[21] = 4.;
    ha[28] = 10.;
    ha[36] = 14.;
    ha[45] = 4.;
    ha[55] = 2.;
    ha[66] = 2.;
    ha[78] = 20.;
    ha[91] = 10.;
    ha[105] = 8.;
    ha[120] = 54.;
/* Computing 2nd power */
    d__1 = x[16];
    ha[136] = d__1 * d__1 * 12.;
    ha[153] = 2.;
    ha[171] = 26.;
    ha[190] = 2.;
    ha[210] = 2.;
    switch (*ka) {
	case 1:  goto L197;
	case 2:  goto L192;
	case 3:  goto L193;
	case 4:  goto L194;
	case 5:  goto L195;
	case 6:  goto L196;
	case 7:  goto L202;
	case 8:  goto L203;
	case 9:  goto L204;
	case 10:  goto L205;
	case 11:  goto L206;
	case 12:  goto L207;
	case 13:  goto L208;
	case 14:  goto L209;
    }
L202:
    ha[1] += 20.;
    return 0;
L203:
    ha[91] += 100.;
    return 0;
L204:
    ha[91] += 60.;
    return 0;
L205:
    ha[1] += 280.;
    return 0;
L206:
    ha[3] += 300.;
    return 0;
L207:
    ha[1] += 100.;
/* Computing 2nd power */
    d__1 = x[17];
    ha[153] += d__1 * d__1 * 1080.;
    return 0;
L208:
    ha[1] += 20.;
    return 0;
L209:
    ha[1] += 140.;
    ha[3] += 100.;
    ha[190] += 20.;
    return 0;
L60:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L61: */
    }
    if (*ka % 2 == 0) {
	i__ = (*ka + 2) / 2;
	j = i__ * (i__ + 1) / 2;
	ha[j] += 2.;
    } else {
	i__ = (*ka + 1) / 2;
	j = i__ * (i__ + 1) / 2;
	ha[j] += 4.;
    }
    return 0;
L70:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L71: */
    }
    k = (*n - 1) / 2;
    kk = k * (k + 1) / 2;
/* Computing 2nd power */
    d__1 = empr22_1.y[*ka - 1];
    a = d__1 * d__1;
    l = 0;
    ll = kk;
    s = 1.;
    if (*ka > 62 && *ka < 124) {
	s = -s;
    }
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	l += i__;
/* Computing 2nd power */
	d__1 = x[i__ + k];
	b = d__1 * d__1 - a;
/* Computing 2nd power */
	d__1 = x[i__];
	c__ = a * (d__1 * d__1);
	p = b * b + c__;
	q = b * b - c__;
	r__ = s * p * p;
	ha[l] = a * q / r__;
	ha[l + ll] = a * -4. * b * x[i__] * x[i__ + k] / r__;
	ll += k;
/* Computing 2nd power */
	d__1 = x[i__ + k];
	ha[l + ll] = s * 2. * b / p - q * 4. * (d__1 * d__1) / r__;
/* L72: */
    }
    return 0;
L90:
    p = 500.;
    a = .99;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L89: */
    }
    ha[25] = -.063;
    switch (*ka) {
	case 1:  goto L91;
	case 2:  goto L92;
	case 3:  goto L93;
	case 4:  goto L94;
	case 5:  goto L95;
	case 6:  goto L96;
	case 7:  goto L97;
	case 8:  goto L98;
	case 9:  goto L99;
    }
L91:
    return 0;
L92:
    ha[29] = p * (.13167 - x[8] * .01334);
    ha[36] = -p * x[1] * .01334;
    return 0;
L93:
    ha[29] = -p * (.13167 - x[8] * .01334);
    ha[36] = p * x[1] * .01334;
    return 0;
L94:
    ha[36] = p * -.076;
    return 0;
L95:
    ha[36] = p * .076;
    return 0;
L96:
/* Computing 3rd power */
    d__1 = x[4] * x[9] + x[3] * 1e3;
    c__ = d__1 * (d__1 * d__1);
    q = x[4] * x[9] - x[3] * 1e3;
    ha[6] = p * -1.96e8 * x[4] * x[9] / c__;
    ha[9] = p * -9.8e4 * x[9] * q / c__;
/* Computing 2nd power */
    d__1 = x[9];
    ha[10] = p * 1.96e5 * x[3] * (d__1 * d__1) / c__;
    ha[39] = p * -9.8e4 * x[4] * q / c__;
    ha[40] = p * 9.8e4 * x[3] * q / c__;
/* Computing 2nd power */
    d__1 = x[4];
    ha[45] = p * 1.96e5 * x[3] * (d__1 * d__1) / c__;
    return 0;
L97:
/* Computing 3rd power */
    d__1 = x[4] * x[9] + x[3] * 1e3;
    c__ = d__1 * (d__1 * d__1);
    q = x[4] * x[9] - x[3] * 1e3;
    ha[6] = p * 1.96e8 * x[4] * x[9] / c__;
    ha[9] = p * 9.8e4 * x[9] * q / c__;
/* Computing 2nd power */
    d__1 = x[9];
    ha[10] = p * -1.96e5 * x[3] * (d__1 * d__1) / c__;
    ha[39] = p * 9.8e4 * x[4] * q / c__;
    ha[40] = p * -9.8e4 * x[3] * q / c__;
/* Computing 2nd power */
    d__1 = x[4];
    ha[45] = p * -1.96e5 * x[3] * (d__1 * d__1) / c__;
    return 0;
L98:
/* Computing 3rd power */
    d__1 = x[1];
    ha[1] = p * 2. * (x[2] + x[5]) / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[1];
    ha[2] = -p / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[1];
    ha[11] = -p / (d__1 * d__1);
    return 0;
L99:
/* Computing 3rd power */
    d__1 = x[1];
    ha[1] = p * -2. * (x[2] + x[5]) / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[1];
    ha[2] = p / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[1];
    ha[11] = p / (d__1 * d__1);
    return 0;
L130:
    p = 1e5;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L131: */
    }
    ha[13] = empr22_1.y[5];
    ha[16] = empr22_1.y[1];
    switch (*ka) {
	case 1:  goto L152;
	case 2:  goto L132;
	case 3:  goto L133;
	case 4:  goto L134;
	case 5:  goto L135;
	case 6:  goto L136;
	case 7:  goto L137;
	case 8:  goto L138;
	case 9:  goto L139;
	case 10:  goto L106;
	case 11:  goto L107;
	case 12:  goto L108;
	case 13:  goto L109;
    }
L132:
/* Computing 3rd power */
    d__1 = x[1];
    ha[1] = p * 2. * empr22_1.y[7] * x[3] / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[1];
    ha[4] = -p * empr22_1.y[7] / (d__1 * d__1);
    ha[21] = p * 2. * empr22_1.y[6];
    return 0;
L133:
/* Computing 2nd power */
    d__1 = x[6];
/* Computing 2nd power */
    d__2 = x[3];
    ha[4] = -p * (empr22_1.y[9] + empr22_1.y[10] * x[6] + empr22_1.y[11] * (
	    d__1 * d__1)) / (d__2 * d__2);
/* Computing 2nd power */
    d__1 = x[6];
/* Computing 3rd power */
    d__2 = x[3];
    ha[6] = p * 2. * (empr22_1.y[9] + empr22_1.y[10] * x[6] + empr22_1.y[11] *
	     (d__1 * d__1)) * x[1] / (d__2 * (d__2 * d__2));
    ha[16] += p * (empr22_1.y[10] + empr22_1.y[11] * 2. * x[6]) / x[3];
/* Computing 2nd power */
    d__1 = x[3];
    ha[18] = -p * (empr22_1.y[10] + empr22_1.y[11] * 2. * x[6]) * x[1] / (
	    d__1 * d__1);
    ha[21] = p * 2. * empr22_1.y[11] * x[1] / x[3];
    return 0;
L134:
    ha[21] = p * 2. * empr22_1.y[12];
    return 0;
L135:
/* Computing 2nd power */
    d__1 = x[5];
    ha[14] = -p * empr22_1.y[18] / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[6];
/* Computing 3rd power */
    d__2 = x[5];
    ha[15] = p * 2. * (empr22_1.y[16] + empr22_1.y[17] * x[6] + empr22_1.y[18]
	     * x[4] + empr22_1.y[19] * (d__1 * d__1)) / (d__2 * (d__2 * d__2))
	    ;
/* Computing 2nd power */
    d__1 = x[5];
    ha[20] = -p * (empr22_1.y[17] + empr22_1.y[19] * 2. * x[6]) / (d__1 * 
	    d__1);
    ha[21] = p * 2. * empr22_1.y[19] / x[5];
    return 0;
L136:
/* Computing 2nd power */
    d__1 = x[3];
    ha[5] = -p * (empr22_1.y[21] / x[4] + empr22_1.y[22]) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (empr22_1.y[21] / x[4] + empr22_1.y[22]) * x[2] / (d__1 *
	     (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[8] = -p * empr22_1.y[21] / (x[3] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[3] * x[4];
    ha[9] = p * empr22_1.y[21] * x[2] / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * empr22_1.y[21] * x[2] / (x[3] * (d__1 * (d__1 * d__1)));
    return 0;
L137:
/* Computing 2nd power */
    d__1 = x[3];
    ha[5] = -p * (empr22_1.y[24] + empr22_1.y[25] / x[4]) / (d__1 * d__1 * x[
	    7]);
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (empr22_1.y[24] + empr22_1.y[25] / x[4]) * x[2] / (d__1 *
	     (d__1 * d__1) * x[7]);
/* Computing 2nd power */
    d__1 = x[4];
    ha[8] = -p * empr22_1.y[25] / (x[3] * (d__1 * d__1) * x[7]);
/* Computing 2nd power */
    d__1 = x[3] * x[4];
    ha[9] = p * empr22_1.y[25] * x[2] / (d__1 * d__1 * x[7]);
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * empr22_1.y[25] * x[2] / (x[3] * (d__1 * (d__1 * d__1)) *
	     x[7]);
/* Computing 2nd power */
    d__1 = x[7];
    ha[23] = -p * (empr22_1.y[24] + empr22_1.y[25] / x[4]) / (x[3] * (d__1 * 
	    d__1));
/* Computing 2nd power */
    d__1 = x[3] * x[7];
    ha[24] = p * (empr22_1.y[24] + empr22_1.y[25] / x[4]) * x[2] / (d__1 * 
	    d__1);
/* Computing 2nd power */
    d__1 = x[4] * x[7];
    ha[25] = p * empr22_1.y[25] * x[2] / (x[3] * (d__1 * d__1));
/* Computing 3rd power */
    d__1 = x[7];
    ha[28] = p * 2. * (empr22_1.y[23] + (empr22_1.y[24] + empr22_1.y[25] / x[
	    4]) * x[2] / x[3]) / (d__1 * (d__1 * d__1));
    return 0;
L138:
/* Computing 3rd power */
    d__1 = x[5];
    ha[15] = p * 2. * (empr22_1.y[26] + empr22_1.y[27] * x[7]) / (d__1 * (
	    d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[5];
    ha[26] = -p * empr22_1.y[27] / (d__1 * d__1);
    return 0;
L139:
/* Computing 2nd power */
    d__1 = x[3];
    ha[4] = -p * empr22_1.y[32] / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (empr22_1.y[32] * x[1] + empr22_1.y[33]) / (d__1 * (d__1 
	    * d__1));
    return 0;
L106:
/* Computing 2nd power */
    d__1 = x[3];
    ha[5] = -p * (empr22_1.y[34] / x[4] + empr22_1.y[35]) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (empr22_1.y[34] / x[4] + empr22_1.y[35]) * x[2] / (d__1 *
	     (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[8] = -p * empr22_1.y[34] / (x[3] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[3] * x[4];
    ha[9] = p * empr22_1.y[34] * x[2] / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * empr22_1.y[34] * x[2] / (x[3] * (d__1 * (d__1 * d__1)));
    return 0;
L107:
/* Computing 3rd power */
    d__1 = x[2];
    ha[3] = p * 2. * empr22_1.y[37] * x[3] * x[4] / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[2];
    ha[5] = -p * empr22_1.y[37] * x[4] / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[2];
    ha[8] = -p * empr22_1.y[37] * x[3] / (d__1 * d__1);
    ha[9] = p * empr22_1.y[37] / x[2];
    return 0;
L108:
    ha[16] += p * empr22_1.y[38];
    return 0;
L109:
/* Computing 3rd power */
    d__1 = x[1];
    ha[1] = p * 2. * (empr22_1.y[41] * x[3] + empr22_1.y[42]) / (d__1 * (d__1 
	    * d__1));
/* Computing 2nd power */
    d__1 = x[1];
    ha[4] = -p * empr22_1.y[41] / (d__1 * d__1);
    return 0;
L150:
    p = 1e5;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L151: */
    }
    switch (*ka) {
	case 1:  goto L152;
	case 2:  goto L153;
	case 3:  goto L154;
	case 4:  goto L155;
    }
L152:
    return 0;
L153:
/* Computing 3rd power */
    d__1 = x[1];
    ha[1] = p * 2. * (x[4] * 833.33252 - 83333.333) / (d__1 * (d__1 * d__1) * 
	    x[6]);
/* Computing 2nd power */
    d__1 = x[1];
    ha[7] = -p * 833.33252 / (d__1 * d__1 * x[6]);
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[6];
    ha[16] = p * (x[4] * 833.33252 - 83333.333) / (d__1 * d__1 * (d__2 * d__2)
	    );
/* Computing 2nd power */
    d__1 = x[6];
    ha[19] = -p * 833.33252 / (x[1] * (d__1 * d__1));
/* Computing 3rd power */
    d__1 = x[6];
    ha[21] = p * 2. * (x[4] * 833.33252 / x[1] + 100. - 83333.333 / x[1]) / (
	    d__1 * (d__1 * d__1));
    return 0;
L154:
/* Computing 3rd power */
    d__1 = x[2];
    ha[3] = p * 2500. * (x[5] - x[4]) / (d__1 * (d__1 * d__1) * x[7]);
/* Computing 2nd power */
    d__1 = x[2];
    ha[8] = p * 1250. / (d__1 * d__1 * x[7]);
/* Computing 2nd power */
    d__1 = x[2];
    ha[12] = -p * 1250. / (d__1 * d__1 * x[7]);
/* Computing 2nd power */
    d__1 = x[2];
/* Computing 2nd power */
    d__2 = x[7];
    ha[23] = p * 1250. * (x[5] - x[4]) / (d__1 * d__1 * (d__2 * d__2));
/* Computing 2nd power */
    d__1 = x[7];
    ha[25] = -p * (1. - 1250. / x[2]) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[7];
    ha[26] = -p * 1250. / (x[2] * (d__1 * d__1));
/* Computing 3rd power */
    d__1 = x[7];
    ha[28] = p * 2. * ((x[5] - x[4]) * 1250. / x[2] + x[4]) / (d__1 * (d__1 * 
	    d__1));
    return 0;
L155:
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (1.25e6 - x[5] * 2500.) / (d__1 * (d__1 * d__1) * x[8]);
/* Computing 2nd power */
    d__1 = x[3];
    ha[13] = p * 2500. / (d__1 * d__1 * x[8]);
/* Computing 2nd power */
    d__1 = x[3];
/* Computing 2nd power */
    d__2 = x[8];
    ha[31] = p * (1.25e6 - x[5] * 2500.) / (d__1 * d__1 * (d__2 * d__2));
/* Computing 2nd power */
    d__1 = x[8];
    ha[33] = -p * (1. - 2500. / x[3]) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[8];
    ha[36] = p * 2. * ((1.25e6 - x[5] * 2500.) / x[3] + x[5]) / (d__1 * (d__1 
	    * d__1));
    return 0;
L170:
    p = 1e3;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L171: */
    }
    ha[67] = -1.23106;
    ha[80] = -1.23106;
    ha[94] = -1.23106;
    ha[109] = -1.23106;
    ha[125] = -1.23106;
    switch (*ka) {
	case 1:  goto L152;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
	case 5:  goto L175;
	case 6:  goto L176;
	case 7:  goto L177;
	case 8:  goto L178;
	case 9:  goto L179;
	case 10:  goto L116;
	case 11:  goto L117;
	case 12:  goto L118;
	case 13:  goto L128;
	case 14:  goto L129;
	case 15:  goto L146;
	case 16:  goto L147;
	case 17:  goto L148;
	case 18:  goto L149;
	case 19:  goto L159;
    }
L172:
    ha[1] = -p * .0195 / x[6];
/* Computing 2nd power */
    d__1 = x[6];
    ha[16] = -p * (.03475 - x[1] * .0195) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[6];
    ha[21] = p * 2. * x[1] * (.03475 - x[1] * .00975) / (d__1 * (d__1 * d__1))
	    ;
    return 0;
L173:
    ha[3] = -p * .0195 / x[7];
/* Computing 2nd power */
    d__1 = x[7];
    ha[23] = -p * (.03475 - x[2] * .0195) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[7];
    ha[28] = p * 2. * x[2] * (.03475 - x[2] * .00975) / (d__1 * (d__1 * d__1))
	    ;
    return 0;
L174:
    ha[6] = -p * .0195 / x[8];
/* Computing 2nd power */
    d__1 = x[8];
    ha[31] = -p * (.03475 - x[3] * .0195) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[8];
    ha[36] = p * 2. * x[3] * (.03475 - x[3] * .00975) / (d__1 * (d__1 * d__1))
	    ;
    return 0;
L175:
    ha[10] = -p * .0195 / x[9];
/* Computing 2nd power */
    d__1 = x[9];
    ha[40] = -p * (.03475 - x[4] * .0195) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[9];
    ha[45] = p * 2. * x[4] * (.03475 - x[4] * .00975) / (d__1 * (d__1 * d__1))
	    ;
    return 0;
L176:
    ha[15] = -p * .0195 / x[10];
/* Computing 2nd power */
    d__1 = x[10];
    ha[50] = -p * (.03475 - x[5] * .0195) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[10];
    ha[55] = p * 2. * x[5] * (.03475 - x[5] * .00975) / (d__1 * (d__1 * d__1))
	    ;
    return 0;
L177:
/* Computing 2nd power */
    d__1 = x[7];
    ha[22] = -p * x[12] / (x[11] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[7];
    ha[27] = -p * (1. - x[12] / x[11]) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[7];
    ha[28] = p * 2. * (x[6] + (x[1] - x[6]) * x[12] / x[11]) / (d__1 * (d__1 *
	     d__1));
/* Computing 2nd power */
    d__1 = x[11];
    ha[56] = -p * x[12] / (x[7] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[11];
    ha[61] = p * x[12] / (x[7] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[11] * x[7];
    ha[62] = p * (x[1] - x[6]) * x[12] / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[11];
    ha[66] = p * 2. * (x[1] - x[6]) * x[12] / (x[7] * (d__1 * (d__1 * d__1)));
    ha[67] += p / (x[11] * x[7]);
    ha[72] = -p / (x[11] * x[7]);
/* Computing 2nd power */
    d__1 = x[7];
    ha[73] = -p * (x[1] - x[6]) / (x[11] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[11];
    ha[77] = -p * (x[1] - x[6]) / (x[7] * (d__1 * d__1));
    return 0;
L178:
/* Computing 2nd power */
    d__1 = x[8];
    ha[29] = p * .002 * x[12] / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[8];
    ha[30] = p * -.002 * x[13] / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[8];
    ha[35] = -p * (x[12] * .002 + 1.) / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[8];
    ha[36] = p * 2. * (x[7] + ((x[7] - x[1]) * x[12] + x[2] * x[13]) * .002) /
	     (d__1 * (d__1 * d__1));
    ha[67] -= p * .002 / x[8];
    ha[73] = p * .002 / x[8];
/* Computing 2nd power */
    d__1 = x[8];
    ha[74] = p * -.002 * (x[7] - x[1]) / (d__1 * d__1);
    ha[80] += p * .002 / x[8];
/* Computing 2nd power */
    d__1 = x[8];
    ha[86] = p * -.002 * x[2] / (d__1 * d__1);
    return 0;
L179:
    ha[80] -= p * .002;
    ha[86] = p * .002;
    ha[94] += p * .002;
    ha[100] = -p * .002;
    return 0;
L116:
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * (x[9] + ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) /
	     x[14]) / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[3];
    ha[9] = -p * x[15] / (x[14] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[3];
    ha[31] = p * x[15] / (x[14] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[3];
    ha[39] = -p * (1. - 500. / x[14]) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[3];
    ha[48] = p * -500. / (x[14] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[14] * x[3];
    ha[94] += p * ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) / (d__1 * 
	    d__1);
/* Computing 2nd power */
    d__1 = x[14];
    ha[95] = -p * x[15] / (x[3] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[14];
    ha[99] = p * x[15] / (x[3] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[14];
    ha[100] = p * 500. / (x[3] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[14];
    ha[101] = p * -500. / (x[3] * (d__1 * d__1));
/* Computing 3rd power */
    d__1 = x[14];
    ha[105] = p * 2. * ((x[4] - x[8]) * x[15] + (x[10] - x[9]) * 500.) / (x[3]
	     * (d__1 * (d__1 * d__1)));
/* Computing 2nd power */
    d__1 = x[3];
    ha[108] = -p * (x[4] - x[8]) / (x[14] * (d__1 * d__1));
    ha[109] += p / (x[3] * x[14]);
    ha[113] = -p / (x[3] * x[14]);
/* Computing 2nd power */
    d__1 = x[14];
    ha[119] = -p * (x[4] - x[8]) / (x[3] * (d__1 * d__1));
    return 0;
L117:
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * (x[10] + (x[5] * x[16] - x[10] * 500.) / x[15]) / (d__1 
	    * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[14] = -p * x[16] / (x[15] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[49] = -p * (1. - 500. / x[15]) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[15] * x[4];
    ha[109] += p * (x[5] * x[16] - x[10] * 500.) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[15];
    ha[110] = -p * x[16] / (x[4] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[15];
    ha[115] = p * 500. / (x[4] * (d__1 * d__1));
/* Computing 3rd power */
    d__1 = x[15];
    ha[120] = p * 2. * (500. - x[16] + (x[5] * x[16] - x[10] * 500.) / x[4]) /
	     (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[124] = -p * x[5] / (x[15] * (d__1 * d__1));
    ha[125] += p / (x[4] * x[15]);
/* Computing 2nd power */
    d__1 = x[15];
    ha[135] = -p * (x[5] / x[4] - 1.) / (d__1 * d__1);
    return 0;
L118:
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * (.9 - x[5] * .002 * x[16]) / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[4];
    ha[14] = p * (x[16] * .002) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[4];
    ha[124] = p * (x[5] * .002) / (d__1 * d__1);
    ha[125] -= p * .002 / x[4];
    return 0;
L128:
/* Computing 3rd power */
    d__1 = x[11];
    ha[66] = p * 2. * x[12] / (d__1 * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[11];
    ha[77] = -p / (d__1 * d__1);
    return 0;
L129:
/* Computing 2nd power */
    d__1 = x[5];
    ha[14] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[5];
    ha[15] = p * 2. * x[4] / (d__1 * (d__1 * d__1));
    return 0;
L146:
/* Computing 2nd power */
    d__1 = x[4];
    ha[9] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[4];
    ha[10] = p * 2. * x[3] / (d__1 * (d__1 * d__1));
    return 0;
L147:
/* Computing 2nd power */
    d__1 = x[3];
    ha[5] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[3];
    ha[6] = p * 2. * x[2] / (d__1 * (d__1 * d__1));
    return 0;
L148:
/* Computing 2nd power */
    d__1 = x[2];
    ha[2] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[2];
    ha[3] = p * 2. * x[1] / (d__1 * (d__1 * d__1));
    return 0;
L149:
/* Computing 2nd power */
    d__1 = x[10];
    ha[54] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[10];
    ha[55] = p * 2. * x[9] / (d__1 * (d__1 * d__1));
    return 0;
L159:
/* Computing 2nd power */
    d__1 = x[9];
    ha[44] = -p / (d__1 * d__1);
/* Computing 3rd power */
    d__1 = x[9];
    ha[45] = p * 2. * x[8] / (d__1 * (d__1 * d__1));
    return 0;
} /* tahd22_ */

