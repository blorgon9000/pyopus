/* test28.f -- translated by f2c (version 20100827).
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
    doublereal y[20], par;
    integer na, m;
} empr28_;

#define empr28_1 empr28_

/* Table of constant values */

static doublereal c_b347 = 1.;
static doublereal c_b413 = 0.;
static doublereal c_b532 = 2.;

/* SUBROUTINE TIUD28                ALL SYSTEMS                92/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 92/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  INITIAL VALUES OF VARIABLES FOR DENSE UNCONSTRAINED MINIMIZATION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RO  X(N)  VECTOR OF VARIABLES. */
/*  RO  FMIN  LOWER BOUND FOR THE OBJECTIVE FUNCTION. */
/*  RO  XMAX  MAXIMUM STEPSIZE. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */
/*  IO  IERR  ERROR INDICATOR. */

/* Subroutine */ int tiud28_(integer *n, doublereal *x, doublereal *fmin, 
	doublereal *xmax, integer *next, integer *ierr)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double exp(doublereal), sin(doublereal), sqrt(doublereal), pow_dd(
	    doublereal *, doublereal *), log(doublereal), cos(doublereal);

    /* Local variables */
    static doublereal f;
    static integer i__, j, k;
    static doublereal p, q, s, t, z__[1000];
    static integer n1;
    static doublereal s1, alf, gam, bet;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    *fmin = 0.;
    *xmax = 1e3;
    *ierr = 0;
    empr28_1.na = *n;
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
	case 11:  goto L110;
	case 12:  goto L120;
	case 13:  goto L130;
	case 14:  goto L140;
	case 15:  goto L150;
	case 16:  goto L160;
	case 17:  goto L170;
	case 18:  goto L180;
	case 19:  goto L190;
	case 20:  goto L200;
	case 21:  goto L210;
	case 22:  goto L220;
	case 23:  goto L230;
	case 24:  goto L250;
	case 25:  goto L310;
	case 26:  goto L320;
	case 27:  goto L330;
	case 28:  goto L350;
	case 29:  goto L370;
	case 30:  goto L390;
	case 31:  goto L400;
	case 32:  goto L450;
	case 33:  goto L460;
	case 34:  goto L470;
	case 35:  goto L480;
	case 36:  goto L490;
	case 37:  goto L500;
	case 38:  goto L510;
	case 39:  goto L520;
	case 40:  goto L530;
	case 41:  goto L540;
	case 42:  goto L550;
	case 43:  goto L560;
	case 44:  goto L570;
	case 45:  goto L580;
	case 46:  goto L590;
	case 47:  goto L600;
	case 48:  goto L610;
	case 49:  goto L620;
	case 50:  goto L630;
	case 51:  goto L720;
	case 52:  goto L740;
	case 53:  goto L750;
	case 54:  goto L760;
	case 55:  goto L780;
	case 56:  goto L790;
	case 57:  goto L810;
	case 58:  goto L830;
	case 59:  goto L840;
	case 60:  goto L860;
	case 61:  goto L870;
	case 62:  goto L880;
	case 63:  goto L900;
	case 64:  goto L910;
	case 65:  goto L920;
	case 66:  goto L930;
	case 67:  goto L940;
	case 68:  goto L950;
	case 69:  goto L960;
	case 70:  goto L970;
	case 71:  goto L980;
	case 72:  goto L990;
	case 73:  goto L800;
	case 74:  goto L240;
	case 75:  goto L410;
	case 76:  goto L420;
	case 77:  goto L650;
	case 78:  goto L660;
	case 79:  goto L670;
	case 80:  goto L680;
	case 81:  goto L690;
	case 82:  goto L340;
	case 83:  goto L360;
	case 84:  goto L380;
	case 85:  goto L430;
	case 86:  goto L440;
	case 87:  goto L270;
	case 88:  goto L280;
	case 89:  goto L290;
	case 90:  goto L300;
	case 91:  goto L710;
	case 92:  goto L820;
    }
L10:
    if (*n < 2) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -1.2;
	} else {
	    x[i__] = 1.;
	}
/* L11: */
    }
    return 0;
L20:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -2.;
	    if (i__ <= 4) {
		x[i__] = -3.;
	    }
	} else {
	    x[i__] = 0.;
	    if (i__ <= 4) {
		x[i__] = -1.;
	    }
	}
/* L21: */
    }
    return 0;
L30:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 4 == 1) {
	    x[i__] = 3.;
	} else if (i__ % 4 == 2) {
	    x[i__] = -1.;
	} else if (i__ % 4 == 3) {
	    x[i__] = 0.;
	} else {
	    x[i__] = 1.;
	}
/* L31: */
    }
    return 0;
L40:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L41: */
    }
    x[1] = 1.;
    return 0;
L50:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L51: */
    }
    return 0;
L60:
    if (*n < 7) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L61: */
    }
    return 0;
L70:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L71: */
    }
    return 0;
L80:
    if (*n < 6) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1. / (doublereal) (*n);
/* L81: */
    }
    return 0;
L90:
    if (*n < 6) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1. / (doublereal) (*n);
/* L91: */
    }
    *fmin = -1e120;
    return 0;
L100:
    if (*n < 6) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L101: */
    }
    *fmin = -1e120;
    return 0;
L110:
    if (*n < 5) {
	goto L999;
    }
    *n -= *n % 5;
    i__1 = *n - 5;
    for (i__ = 0; i__ <= i__1; i__ += 5) {
	x[i__ + 1] = -1.;
	x[i__ + 2] = -1.;
	x[i__ + 3] = 2.;
	x[i__ + 4] = -1.;
	x[i__ + 5] = -1.;
/* L111: */
    }
    x[1] = -2.;
    x[2] = 2.;
    *xmax = 1.;
    return 0;
L120:
    if (*n < 2) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	x[i__ - 1] = 0.;
	x[i__] = -1.;
/* L121: */
    }
    *xmax = 10.;
    return 0;
L130:
    if (*n < 2) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	x[i__ - 1] = -1.;
	x[i__] = 1.;
/* L131: */
    }
    *xmax = 10.;
    return 0;
L140:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q = p * (doublereal) i__;
	x[i__] = q * (1. - q);
/* L141: */
    }
    return 0;
L150:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = p;
	x[i__] = (doublereal) i__ * (doublereal) (*n + 1 - i__) * (d__1 * 
		d__1);
/* L151: */
    }
    *fmin = -1e120;
    return 0;
L160:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L161: */
    }
    *fmin = -1e120;
    return 0;
L170:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = p;
	x[i__] = (doublereal) i__ * (doublereal) (*n + 1 - i__) * (d__1 * 
		d__1);
/* L171: */
    }
    *fmin = -1e120;
    return 0;
L180:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = p;
	x[i__] = (doublereal) i__ * (doublereal) (*n + 1 - i__) * (d__1 * 
		d__1);
/* L181: */
    }
    *fmin = -1e120;
    return 0;
L190:
    if (*n < 3) {
	goto L999;
    }
    p = exp(2.) / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = (p * (doublereal) i__ + 1.) / 3.;
/* L191: */
    }
    *fmin = -1e120;
    return 0;
L200:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = p * (doublereal) i__;
/* L201: */
    }
    *fmin = -1e120;
    return 0;
L210:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = p * (doublereal) i__ + 1.;
/* L211: */
    }
    *fmin = -1e120;
    return 0;
L220:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = p;
	x[i__] = (doublereal) i__ * (doublereal) (*n + 1 - i__) * (d__1 * 
		d__1);
/* L221: */
    }
    *fmin = -1e120;
    return 0;
L230:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L231: */
    }
    return 0;
L250:
    if (*n < 3) {
	goto L999;
    }
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = (doublereal) i__ * p;
/* L251: */
    }
    *xmax = 1.;
    return 0;
L310:
    if (*n >= 2) {
	*n -= *n % 2;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 2 == 1) {
		x[i__] = -1.2;
	    } else {
		x[i__] = 1.;
	    }
/* L311: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L320:
    if (*n >= 4) {
	*n -= *n % 4;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 4 == 1) {
		x[i__] = 3.;
	    } else if (i__ % 4 == 2) {
		x[i__] = -1.;
	    } else if (i__ % 4 == 3) {
		x[i__] = 0.;
	    } else {
		x[i__] = 1.;
	    }
/* L321: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L330:
    if (*n >= 2) {
	empr28_1.na = *n + 1;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = (doublereal) i__;
/* L331: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L350:
    if (*n >= 2) {
	empr28_1.na = *n + 2;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1. - (doublereal) i__ / (doublereal) (*n);
/* L351: */
	}
	*xmax = 100.;
    } else {
	*ierr = 1;
    }
    return 0;
L370:
    if (*n >= 2) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = .5;
/* L371: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L390:
    if (*n >= 2) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = (doublereal) i__ / (doublereal) (*n + 1);
	    x[i__] *= x[i__] - 1.;
/* L391: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L400:
    if (*n >= 2) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = -1.;
/* L401: */
	}
    } else {
	*ierr = 1;
    }
L450:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L451: */
    }
    return 0;
L460:
    if (*n < 6) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L461: */
    }
    return 0;
L470:
    if (*n < 2) {
	goto L999;
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L471: */
    }
    x[*n] = -2.;
    empr28_1.na = *n - 1 << 1;
    return 0;
L480:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 4;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = sin((doublereal) i__);
	x[i__] = d__1 * d__1;
/* L481: */
    }
    empr28_1.na = *n * 5;
    return 0;
L490:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 5.;
/* L491: */
    }
    empr28_1.na = (*n - 2) * 3;
    return 0;
L500:
    if (*n < 2) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .2;
/* L501: */
    }
    empr28_1.na = *n - 1 << 1;
    return 0;
L510:
    if (*n < 2) {
	goto L999;
    }
    *n -= *n % 2;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -.8;
	} else {
	    x[i__] = -.8;
	}
/* L511: */
    }
    empr28_1.na = *n - 1 << 1;
    return 0;
L520:
    if (*n < 5) {
	goto L999;
    }
    if ((*n - 5) % 3 != 0) {
	*n -= (*n - 5) % 3;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L521: */
    }
    empr28_1.na = ((*n - 5) / 3 + 1) * 6;
    return 0;
L530:
    if (*n < 5) {
	goto L999;
    }
    if ((*n - 5) % 3 != 0) {
	*n -= (*n - 5) % 3;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L531: */
    }
    empr28_1.na = ((*n - 5) / 3 + 1) * 7;
    return 0;
L540:
    if (*n < 4) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 4 == 1) {
	    x[i__] = -.8;
	} else if (i__ % 4 == 2) {
	    x[i__] = 1.2;
	} else if (i__ % 4 == 3) {
	    x[i__] = -1.2;
	} else {
	    x[i__] = .8;
	}
/* L541: */
    }
    empr28_1.y[0] = 14.4;
    empr28_1.y[1] = 6.8;
    empr28_1.y[2] = 4.2;
    empr28_1.y[3] = 3.2;
L542:
    if ((*n - 4) % 2 != 0) {
	*n -= (*n - 4) % 2;
    }
    empr28_1.na = (*n - 4) / 2 + 1 << 2;
    return 0;
L550:
    if (*n < 4) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 4 == 1) {
	    x[i__] = -.8;
	} else if (i__ % 4 == 2) {
	    x[i__] = 1.2;
	} else if (i__ % 4 == 3) {
	    x[i__] = -1.2;
	} else {
	    x[i__] = .8;
	}
/* L551: */
    }
    empr28_1.y[0] = 35.8;
    empr28_1.y[1] = 11.2;
    empr28_1.y[2] = 6.2;
    empr28_1.y[3] = 4.4;
    goto L542;
L560:
    if (*n < 4) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 4 == 1) {
	    x[i__] = -.8;
	} else if (i__ % 4 == 2) {
	    x[i__] = 1.2;
	} else if (i__ % 4 == 3) {
	    x[i__] = -1.2;
	} else {
	    x[i__] = .8;
	}
/* L561: */
    }
    empr28_1.y[0] = 30.6;
    empr28_1.y[1] = 72.2;
    empr28_1.y[2] = 124.4;
    empr28_1.y[3] = 187.4;
    goto L542;
L570:
    if (*n < 4) {
	goto L999;
    }
    *n -= *n % 2;
    empr28_1.na = *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 8 == 1) {
	    x[i__] = .1;
	}
	if (i__ % 8 == 2 || i__ % 8 == 0) {
	    x[i__] = .2;
	}
	if (i__ % 8 == 3 || i__ % 8 == 7) {
	    x[i__] = .3;
	}
	if (i__ % 8 == 4 || i__ % 8 == 6) {
	    x[i__] = .4;
	}
	if (i__ % 8 == 5) {
	    x[i__] = .5;
	}
/* L571: */
    }
    return 0;
L580:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 12.;
/* L581: */
    }
    *xmax = 10.;
    return 0;
L590:
    if (*n < 7) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L591: */
    }
    return 0;
L600:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = (doublereal) i__ / (doublereal) (*n + 1);
	x[i__] *= x[i__] - 1.;
/* L601: */
    }
    return 0;
L610:
    if (*n < 5) {
	goto L999;
    }
    if ((*n - 5) % 3 != 0) {
	*n -= (*n - 5) % 3;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L611: */
    }
    empr28_1.na = ((*n - 5) / 3 + 1) * 7;
    return 0;
L620:
    if (*n < 3) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ % 2 == 1) {
	    x[i__] = -1.2;
	} else {
	    x[i__] = 1.;
	}
/* L621: */
    }
    empr28_1.na = *n - 1 << 1;
    return 0;
L630:
    if (*n < 7) {
	goto L999;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 5.;
/* L631: */
    }
    empr28_1.y[0] = sin(1.);
    empr28_1.na = (*n - 6) * 13;
    return 0;
L720:
    if (*n >= 5) {
	*n -= *n % 2;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 8 == 1) {
		x[i__] = .1;
	    }
	    if (i__ % 8 == 2 || i__ % 8 == 0) {
		x[i__] = .2;
	    }
	    if (i__ % 8 == 3 || i__ % 8 == 7) {
		x[i__] = .3;
	    }
	    if (i__ % 8 == 4 || i__ % 8 == 6) {
		x[i__] = .4;
	    }
	    if (i__ % 8 == 5) {
		x[i__] = .5;
	    }
/* L721: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L740:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 0.;
/* L741: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L750:
    if (*n >= 3) {
	if (*n % 2 != 1) {
	    --(*n);
	}
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L751: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L760:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = -1.;
/* L761: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L780:
    if (*n >= 5) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = -2.;
/* L781: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L790:
    if (*n >= 7) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = -3.;
/* L791: */
	}
	*xmax = 10.;
    } else {
	*ierr = 1;
    }
    return 0;
L810:
    if (*n >= 2) {
	*n -= *n % 2;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 2 == 1) {
		x[i__] = 90.;
	    } else {
		x[i__] = 60.;
	    }
/* L811: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L830:
    if (*n >= 4) {
	*n -= *n % 4;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 4 == 1) {
		x[i__] = 1.;
	    } else if (i__ % 4 == 2) {
		x[i__] = 2.;
	    } else if (i__ % 4 == 3) {
		x[i__] = 2.;
	    } else {
		x[i__] = 2.;
	    }
/* L831: */
	}
	*xmax = 10.;
    } else {
	*ierr = 1;
    }
    return 0;
L840:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = -1.;
/* L841: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L860:
    if (*n >= 2) {
	*n -= *n % 2;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 2 == 1) {
		x[i__] = 0.;
	    } else {
		x[i__] = 1.;
	    }
/* L861: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L870:
    if (*n >= 4) {
	*n -= *n % 4;
	empr28_1.na = *n;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; i__ += 2) {
	    x[i__ - 1] = -3.;
	    x[i__] = -1.;
/* L871: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L880:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.5;
/* L881: */
	}
	*xmax = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L900:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 10.;
/* L901: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L910:
    if (*n >= 3) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L911: */
	}
	empr28_1.par = 10.;
    } else {
	*ierr = 1;
    }
    return 0;
L920:
    if (*n >= 5) {
	empr28_1.par = 500. / (doublereal) (*n + 2);
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	    d__1 = ((doublereal) i__ + .5) / (doublereal) (*n + 2) - .5;
	    x[i__] = d__1 * d__1;
/* L921: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L930:
    if (*n >= 10) {
	*n -= *n % 2;
	empr28_1.m = *n / 2;
	empr28_1.par = 500.;
	empr28_1.na = *n;
	i__1 = empr28_1.m;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	    d__1 = (doublereal) i__ / (doublereal) (empr28_1.m + 1) - .5;
	    x[i__] = d__1 * d__1;
/* L931: */
	}
	i__1 = *n;
	for (i__ = empr28_1.m + 1; i__ <= i__1; ++i__) {
	    k = i__ - empr28_1.m;
	    x[i__] = (doublereal) k / (doublereal) (empr28_1.m + 1) - .5;
/* L932: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L940:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
/* Computing 2nd power */
	d__1 = (doublereal) (empr28_1.m + 1);
	empr28_1.par = 6.8 / (d__1 * d__1);
	*n = empr28_1.m * empr28_1.m;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = 0.;
/* L941: */
	    }
/* L942: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L950:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
/* Computing 2nd power */
	d__1 = (doublereal) (empr28_1.m + 1);
	empr28_1.par = 1. / (d__1 * d__1);
	*n = empr28_1.m * empr28_1.m;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = -1.;
/* L951: */
	    }
/* L952: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L960:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
/* Computing 2nd power */
	d__1 = (doublereal) (empr28_1.m + 1);
	empr28_1.par = 1. / (d__1 * d__1);
	*n = empr28_1.m * empr28_1.m;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = 0.;
/* L961: */
	    }
/* L962: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L970:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
	empr28_1.par = 50. / (doublereal) (empr28_1.m + 1);
	*n = empr28_1.m * empr28_1.m;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
/* Computing 2nd power */
		d__1 = (doublereal) (empr28_1.m + 1);
		x[k] = 1. - (doublereal) i__ * (doublereal) j / (d__1 * d__1);
/* L971: */
	    }
/* L972: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L980:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
	empr28_1.par = 1. / (doublereal) (empr28_1.m + 1);
	*n = empr28_1.m * empr28_1.m;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = 0.;
/* L981: */
	    }
/* L982: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L990:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
	*n = empr28_1.m * empr28_1.m;
/* Computing 4th power */
	d__1 = (doublereal) (empr28_1.m + 2), d__1 *= d__1;
	empr28_1.par = 500. / (d__1 * d__1);
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = 0.;
/* L991: */
	    }
/* L992: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L800:
    if (*n >= 16) {
	empr28_1.m = (integer) sqrt((doublereal) (*n));
	*n = empr28_1.m * empr28_1.m;
	empr28_1.par = 500.;
	k = 0;
	i__1 = empr28_1.m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = empr28_1.m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		++k;
		x[k] = 0.;
/* L801: */
	    }
/* L802: */
	}
	empr28_1.na = *n;
    } else {
	*ierr = 1;
    }
    return 0;
L240:
    if (*n >= 2) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ % 2 == 1) {
		x[i__] = 1.;
	    } else {
		x[i__] = 3.;
	    }
/* L241: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L410:
    n1 = *n - 1;
    i__1 = n1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.2;
/* L411: */
    }
    x[*n] = -1.;
    return 0;
L420:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L421: */
    }
    return 0;
L650:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.5;
/* L651: */
    }
    return 0;
L660:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L661: */
    }
    return 0;
L670:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L671: */
    }
    return 0;
L680:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L681: */
    }
    return 0;
L690:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L691: */
    }
    return 0;
L340:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L341: */
    }
    return 0;
L360:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L361: */
    }
    return 0;
L380:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L381: */
    }
    return 0;
L430:
    alf = 5.;
    bet = 14.;
    gam = 3.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L431: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = (doublereal) i__ - (doublereal) (*n) / 2.;
	f = bet * *n * x[i__] + pow_dd(&d__1, &gam);
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    if (j != i__) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) i__ / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		f += t * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf));
	    }
/* L432: */
	}
	z__[i__ - 1] = -f;
/* L433: */
    }
/* Computing 2nd power */
    d__1 = bet;
/* Computing 2nd power */
    i__1 = *n;
/* Computing 2nd power */
    d__2 = alf + 1;
/* Computing 2nd power */
    i__2 = *n - 1;
    s = bet * *n / (d__1 * d__1 * (i__1 * i__1) - d__2 * d__2 * (i__2 * i__2))
	    ;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = s * z__[i__ - 1];
/* L434: */
    }
    return 0;
L440:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L441: */
    }
    return 0;
L270:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L271: */
    }
    return 0;
L280:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L281: */
    }
    return 0;
L290:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L291: */
    }
    return 0;
L300:
    t = 2. / (doublereal) (*n + 2);
    n1 = *n / 2;
    i__1 = n1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s = (doublereal) i__ * t;
	x[i__] = s * (1. - s);
	x[n1 + i__] = x[i__];
/* L301: */
    }
    return 0;
L710:
    n1 = (integer) sqrt((doublereal) (*n));
    *n = n1 * n1;
    empr28_1.na = *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L711: */
    }
    return 0;
L820:
    n1 = (integer) sqrt((doublereal) (*n));
    *n = n1 * n1;
    empr28_1.na = *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L821: */
    }
    return 0;
L999:
    *ierr = 1;
    return 0;
} /* tiud28_ */

/* SUBROUTINE TFFU28                ALL SYSTEMS                92/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 92/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  VALUES OF MODEL FUNCTIONS FOR UNCONSTRAINED MINIMIZATION. */
/*  UNIVERSAL VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  F  VALUE OF THE MODEL FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tffu28_(integer *n, doublereal *x, doublereal *f, 
	integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */
    double exp(doublereal), sin(doublereal), cos(doublereal), pow_dd(
	    doublereal *, doublereal *), atan(doublereal), sqrt(doublereal), 
	    log(doublereal), pow_di(doublereal *, integer *), d_sign(
	    doublereal *, doublereal *), sinh(doublereal);

    /* Local variables */
    static doublereal a, b, c__, d__, e, h__;
    static integer i__, j, k, l;
    static doublereal p, q, r__, s, t, u, v, w, a1, a2, a3, a4, h2;
    static integer i1, i2, j1, j2, n1;
    static doublereal s1, s2, s3, t1, ca, cb, fa, be, ga;
    static integer ia, ib;
    static doublereal ff, al;
    static integer ic, ka, la, nd;
    static doublereal pi, ex, be1, be2, al1, al2, d1s, d2s, alf, gam, bet, 
	    alfa;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    pi = 3.14159265358979323846;
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
	case 10:  goto L100;
	case 11:  goto L110;
	case 12:  goto L120;
	case 13:  goto L130;
	case 14:  goto L140;
	case 15:  goto L150;
	case 16:  goto L160;
	case 17:  goto L170;
	case 18:  goto L180;
	case 19:  goto L190;
	case 20:  goto L200;
	case 21:  goto L210;
	case 22:  goto L220;
	case 23:  goto L230;
	case 24:  goto L250;
	case 25:  goto L310;
	case 26:  goto L320;
	case 27:  goto L330;
	case 28:  goto L350;
	case 29:  goto L370;
	case 30:  goto L390;
	case 31:  goto L400;
	case 32:  goto L450;
	case 33:  goto L460;
	case 34:  goto L470;
	case 35:  goto L480;
	case 36:  goto L490;
	case 37:  goto L500;
	case 38:  goto L510;
	case 39:  goto L520;
	case 40:  goto L530;
	case 41:  goto L540;
	case 42:  goto L550;
	case 43:  goto L560;
	case 44:  goto L570;
	case 45:  goto L580;
	case 46:  goto L590;
	case 47:  goto L600;
	case 48:  goto L610;
	case 49:  goto L620;
	case 50:  goto L630;
	case 51:  goto L720;
	case 52:  goto L740;
	case 53:  goto L750;
	case 54:  goto L760;
	case 55:  goto L780;
	case 56:  goto L790;
	case 57:  goto L810;
	case 58:  goto L830;
	case 59:  goto L840;
	case 60:  goto L860;
	case 61:  goto L870;
	case 62:  goto L880;
	case 63:  goto L900;
	case 64:  goto L910;
	case 65:  goto L920;
	case 66:  goto L930;
	case 67:  goto L940;
	case 68:  goto L950;
	case 69:  goto L960;
	case 70:  goto L970;
	case 71:  goto L980;
	case 72:  goto L990;
	case 73:  goto L800;
	case 74:  goto L240;
	case 75:  goto L410;
	case 76:  goto L420;
	case 77:  goto L650;
	case 78:  goto L660;
	case 79:  goto L670;
	case 80:  goto L680;
	case 81:  goto L690;
	case 82:  goto L340;
	case 83:  goto L360;
	case 84:  goto L380;
	case 85:  goto L430;
	case 86:  goto L440;
	case 87:  goto L270;
	case 88:  goto L280;
	case 89:  goto L290;
	case 90:  goto L300;
	case 91:  goto L710;
	case 92:  goto L820;
    }
L10:
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
	*f = *f + d__1 * d__1 * 100. + d__2 * d__2;
/* L11: */
    }
    return 0;
L20:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
/* Computing 2nd power */
	d__1 = x[j + 1];
	c__ = d__1 * d__1 - x[j + 2];
	d__ = x[j + 1] - 1.;
	u = x[j] + x[j + 2] - 2.;
	v = x[j] - x[j + 2];
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
/* Computing 2nd power */
	d__3 = c__;
/* Computing 2nd power */
	d__4 = d__;
/* Computing 2nd power */
	d__5 = u;
/* Computing 2nd power */
	d__6 = v;
	*f = *f + d__1 * d__1 * 100. + d__2 * d__2 + d__3 * d__3 * 90. + d__4 
		* d__4 + d__5 * d__5 * 10. + d__6 * d__6 * .1;
/* L21: */
    }
    return 0;
L30:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] + x[j] * 10.;
	b = x[j + 1] - x[j + 2];
	c__ = x[j] - x[j + 1] * 2.;
	d__ = x[j - 1] - x[j + 2];
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
/* Computing 4th power */
	d__3 = c__, d__3 *= d__3;
/* Computing 4th power */
	d__4 = d__, d__4 *= d__4;
	*f = *f + d__1 * d__1 + d__2 * d__2 * 5. + d__3 * d__3 + d__4 * d__4 *
		 10.;
/* L31: */
    }
    return 0;
L40:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = exp(x[j - 1]);
	b = a - x[j];
	d__ = x[j] - x[j + 1];
	p = x[j + 1] - x[j + 2];
	q = sin(p) / cos(p);
	u = x[j - 1];
	v = x[j + 2] - 1.;
/* Computing 4th power */
	d__1 = b, d__1 *= d__1;
/* Computing 6th power */
	d__2 = d__, d__2 *= d__2;
/* Computing 4th power */
	d__3 = q, d__3 *= d__3;
/* Computing 8th power */
	d__4 = u, d__4 *= d__4, d__4 *= d__4;
/* Computing 2nd power */
	d__5 = v;
	*f = *f + d__1 * d__1 + d__2 * (d__2 * d__2) * 100. + d__3 * d__3 + 
		d__4 * d__4 + d__5 * d__5;
/* L41: */
    }
    return 0;
L50:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
/* L51: */
    }
    return 0;
L60:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = (d__1 * d__1 * 5. + 2.) * x[j] + 1.;
/* Computing MAX */
	i__2 = 1, i__3 = j - 5;
/* Computing MIN */
	i__5 = *n, i__6 = j + 1;
	i__4 = min(i__5,i__6);
	for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
	    if (i__ != j) {
		a += x[i__] * (x[i__] + 1.);
	    }
/* L61: */
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
/* L63: */
    }
    return 0;
L70:
    p = 2.3333333333333335;
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
	if (j <= k) {
	    a = x[j] + x[j + k];
	    d__1 = abs(a);
	    *f += pow_dd(&d__1, &p);
	}
/* L71: */
    }
    return 0;
L80:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__4 = j + 2;
	for (i__ = j - 2; i__ <= i__4; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L81;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
L81:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	}
/* Computing 2nd power */
	d__1 = (doublereal) (*n + j) - p;
	*f += d__1 * d__1 / (doublereal) (*n);
/* L83: */
    }
    return 0;
L90:
    k = *n / 2;
    q = 1. / (doublereal) (*n);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__4 = j + 2;
	for (i__ = j - 2; i__ <= i__4; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L91;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
L91:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	}
	*f += (p + (doublereal) j * (1. - cos(x[j]))) * q;
/* L92: */
    }
    return 0;
L100:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	q = (doublereal) j / 10. + 1.;
	i__4 = j + 2;
	for (i__ = j - 2; i__ <= i__4; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L101;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
L101:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	}
	*f += p;
/* L102: */
    }
    *f /= (doublereal) (*n);
    return 0;
L110:
    p = -.002008;
    q = -.0019;
    r__ = -2.61e-4;
    i__1 = *n - 5;
    for (i__ = 0; i__ <= i__1; i__ += 5) {
	a = 1.;
	b = 0.;
	for (j = 1; j <= 5; ++j) {
	    a *= x[i__ + j];
/* Computing 2nd power */
	    d__1 = x[i__ + j];
	    b += d__1 * d__1;
/* L111: */
	}
	a = exp(a);
	b = b - 10. - p;
	c__ = x[i__ + 2] * x[i__ + 3] - x[i__ + 4] * 5. * x[i__ + 5] - q;
/* Computing 3rd power */
	d__1 = x[i__ + 1];
/* Computing 3rd power */
	d__2 = x[i__ + 2];
	d__ = d__1 * (d__1 * d__1) + d__2 * (d__2 * d__2) + 1. - r__;
/* Computing 2nd power */
	d__1 = b;
/* Computing 2nd power */
	d__2 = c__;
/* Computing 2nd power */
	d__3 = d__;
	*f = *f + a + (d__1 * d__1 + d__2 * d__2 + d__3 * d__3) * 10.;
/* L112: */
    }
    return 0;
L120:
    c__ = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] - 3.;
	b = x[j - 1] - x[j];
	c__ += a;
/* Computing 2nd power */
	d__1 = a;
	*f = *f + d__1 * d__1 * 1e-4 - b + exp(b * 20.);
/* L121: */
    }
/* Computing 2nd power */
    d__1 = c__;
    *f += d__1 * d__1;
    return 0;
L130:
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j];
	a = d__1 * d__1;
	if (a == 0.) {
	    a = 1e-60;
	}
/* Computing 2nd power */
	d__1 = x[j - 1];
	b = d__1 * d__1;
	if (b == 0.) {
	    b = 1e-60;
	}
	c__ = a + 1.;
	d__ = b + 1.;
	*f = *f + pow_dd(&b, &c__) + pow_dd(&a, &d__);
/* L131: */
    }
    return 0;
L140:
    p = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
    d__1 = p;
    q = d__1 * d__1 * .5;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 3rd power */
	d__1 = x[j] + (doublereal) j * p + 1.;
	a = x[j] * 2. + q * (d__1 * (d__1 * d__1));
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
/* Computing 2nd power */
	d__1 = a;
	*f += d__1 * d__1;
/* L141: */
    }
    return 0;
L150:
    p = 1. / (doublereal) (*n + 1);
    q = 2. / p;
    r__ = p * 2.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	a = x[j - 1] - x[j];
	*f += q * x[j - 1] * a;
	if (abs(a) <= 1e-6) {
	    *f += r__ * exp(x[j]) * (a / 2. * (a / 3. * (a / 4. + 1.) + 1.) + 
		    1.);
	} else {
	    b = exp(x[j - 1]) - exp(x[j]);
	    *f += r__ * b / a;
	}
/* L151: */
    }
/* Computing 2nd power */
    d__1 = x[*n];
    *f = *f + q * (d__1 * d__1) + r__ * (exp(x[1]) - 1.) / x[1] + r__ * (exp(
	    x[*n]) - 1.) / x[*n];
    return 0;
L160:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (doublereal) j * (1. - cos(x[j]));
	if (j > 1) {
	    a += (doublereal) j * sin(x[j - 1]);
	}
	if (j < *n) {
	    a -= (doublereal) j * sin(x[j + 1]);
	}
	*f += a;
/* L161: */
    }
    return 0;
L170:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
	    *f = *f + d__1 * d__1 * .25 / p + d__2 * d__2 * .125 / p + p * (
		    exp(x[j]) - 1.);
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
	    *f = *f + d__1 * d__1 * .25 / p + d__2 * d__2 * .125 / p + p * (
		    exp(x[j]) - 1.);
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
	    *f = *f + d__1 * d__1 * .125 / p + p * (exp(x[j]) - 1.);
	}
/* L171: */
    }
    return 0;
L180:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = (doublereal) j * p;
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
/* Computing 2nd power */
	    d__3 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * .25 / p - p * (
		    d__3 * d__3 + x[j] * 2. * q);
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
/* Computing 2nd power */
	    d__3 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * .25 / p - p * (
		    d__3 * d__3 + x[j] * 2. * q);
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
/* Computing 2nd power */
	    d__2 = x[j];
	    *f = *f + d__1 * d__1 * .25 / p - p * (d__2 * d__2 + x[j] * 2. * 
		    q);
	}
/* L181: */
    }
    return 0;
L190:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = exp((doublereal) j * 2. * p);
	if (j == 1) {
	    r__ = .33333333333333331;
/* Computing 2nd power */
	    d__1 = x[j] - r__;
/* Computing 2nd power */
	    d__2 = r__;
/* Computing 2nd power */
	    d__3 = x[j + 1] - r__;
/* Computing 2nd power */
	    d__4 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * 7. + d__3 * d__3 * 
		    .25 / p + p * (d__4 * d__4 + x[j] * 2. * q);
	} else if (j == *n) {
	    r__ = exp(2.) / 3.;
/* Computing 2nd power */
	    d__1 = x[j] - r__;
/* Computing 2nd power */
	    d__2 = r__;
/* Computing 2nd power */
	    d__3 = x[j - 1] - r__;
/* Computing 2nd power */
	    d__4 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * 7. + d__3 * d__3 * 
		    .25 / p + p * (d__4 * d__4 + x[j] * 2. * q);
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
/* Computing 2nd power */
	    d__2 = x[j];
	    *f = *f + d__1 * d__1 * .25 / p + p * (d__2 * d__2 + x[j] * 2. * 
		    q);
	}
/* L191: */
    }
    return 0;
L200:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = exp(d__1 * d__1 * -2.);
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
	    *f = *f + (d__1 * d__1 * .5 / p - p) + (d__2 * d__2 * .25 / p - p)
		     * a;
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
	    *f = *f + (d__1 * d__1 * .5 / p - p) * exp(-2.) + (d__2 * d__2 * 
		    .25 / p - p) * a;
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
	    *f += (d__1 * d__1 * .25 / p - p) * a;
	}
/* L201: */
    }
    return 0;
L210:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = (x[j + 1] - 1.) * .5 / p;
	    b = (x[j] - 1.) / p;
	    u = atan(a);
	    v = atan(b);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__3 = b;
	    *f = *f + p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.))) 
		    + p * .5 * (b * v + 1. - log(sqrt(d__3 * d__3 + 1.)));
	} else if (j == *n) {
	    a = (2. - x[j - 1]) * .5 / p;
	    b = (2. - x[j]) / p;
	    u = atan(a);
	    v = atan(b);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__3 = b;
	    *f = *f + p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.))) 
		    + p * .5 * (b * v + 4. - log(sqrt(d__3 * d__3 + 1.)));
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    u = atan(a);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
	    *f += p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.)));
	}
/* L211: */
    }
    return 0;
L220:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = x[j + 1] * .5 / p;
	    b = x[j] / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
/* Computing 4th power */
	    d__4 = b, d__4 *= d__4;
/* Computing 2nd power */
	    d__5 = 1. - b;
	    *f = *f + p * (d__1 * d__1 * 100. + d__3 * d__3) + p * .5 * (d__4 
		    * d__4 * 100. + d__5 * d__5);
	} else if (j == *n) {
	    a = x[j - 1] * -.5 / p;
	    b = -x[j] / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
/* Computing 4th power */
	    d__4 = b, d__4 *= d__4;
/* Computing 2nd power */
	    d__5 = 1. - b;
	    *f = *f + p * (d__1 * d__1 * 100. + d__3 * d__3) + p * .5 * (d__4 
		    * d__4 * 100. + d__5 * d__5);
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
	    *f += p * (d__1 * d__1 * 100. + d__3 * d__3);
	}
/* L221: */
    }
    return 0;
L230:
    a1 = -1.;
    a2 = 0.;
    a3 = 0.;
    d1s = exp(.01);
    d2s = 1.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a1 += (doublereal) (*n - j + 1) * (d__1 * d__1);
/* L231: */
    }
    a = a1 * 4.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	s1 = exp(x[j] / 100.);
	if (j > 1) {
	    s3 = s1 + s2 - d2s * (d1s - 1.);
/* Computing 2nd power */
	    d__1 = s3;
	    a2 += d__1 * d__1;
/* Computing 2nd power */
	    d__1 = s1 - 1. / d1s;
	    a3 += d__1 * d__1;
	}
	s2 = s1;
	d2s = d1s * d2s;
/* L232: */
    }
/* Computing 2nd power */
    d__1 = a1;
/* Computing 2nd power */
    d__2 = x[1] - .2;
    *f = (a2 + a3) * 1e-5 + d__1 * d__1 + d__2 * d__2;
    return 0;
L250:
    a = 1.;
    b = 0.;
    c__ = 0.;
    d__ = 0.;
    *f = 0.;
    u = exp(x[*n]);
    v = exp(x[*n - 1]);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *n / 2) {
/* Computing 2nd power */
	    d__1 = x[j] - 1.;
	    *f += d__1 * d__1;
	}
	if (j <= *n - 2) {
/* Computing 2nd power */
	    d__1 = x[j] + x[j + 1] * 2. + x[j + 2] * 10. - 1.;
	    b += d__1 * d__1;
/* Computing 2nd power */
	    d__1 = x[j] * 2. + x[j + 1] - 3.;
	    c__ += d__1 * d__1;
	}
/* Computing 2nd power */
	d__1 = x[j];
	d__ = d__ + d__1 * d__1 - (doublereal) (*n);
/* L251: */
    }
/* Computing 2nd power */
    d__1 = d__;
    *f = *f + a * (u * b + 1. + b * c__ + v * c__) + d__1 * d__1;
    return 0;
L310:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = (x[ka + 1] - d__1 * d__1) * 10.;
	} else {
	    fa = 1. - x[ka - 1];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L311: */
    }
    *f *= .5;
    return 0;
L320:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    fa = x[ka] + x[ka + 1] * 10.;
	} else if (ka % 4 == 2) {
	    fa = (x[ka + 1] - x[ka + 2]) * 2.23606797749979;
	} else if (ka % 4 == 3) {
	    a = x[ka - 1] - x[ka] * 2.;
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 3] - x[ka];
	    fa = d__1 * d__1 * 3.16227766016838;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L321: */
    }
    *f *= .5;
    return 0;
L330:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = (x[ka] - 1.) / 316.22776601683825;
	} else {
	    fa = -.25;
	    i__4 = *n;
	    for (j = 1; j <= i__4; ++j) {
/* Computing 2nd power */
		d__1 = x[j];
		fa += d__1 * d__1;
/* L331: */
	    }
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L333: */
    }
    *f *= .5;
    return 0;
L350:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = x[ka] - 1.;
	} else {
	    fa = 0.;
	    i__4 = *n;
	    for (j = 1; j <= i__4; ++j) {
		fa += (doublereal) j * (x[j] - 1.);
/* L351: */
	    }
	    if (ka == *n + 1) {
	    } else if (ka == *n + 2) {
/* Computing 2nd power */
		d__1 = fa;
		fa = d__1 * d__1;
	    }
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L354: */
    }
    *f *= .5;
    return 0;
L370:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < *n) {
	    a = 0.;
	    i__4 = *n;
	    for (j = 1; j <= i__4; ++j) {
		a += x[j];
/* L371: */
	    }
	    fa = x[ka] + a - (doublereal) (*n + 1);
	} else {
	    a = 1.;
	    i__4 = *n;
	    for (j = 1; j <= i__4; ++j) {
		a *= x[j];
/* L373: */
	    }
	    fa = a - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L376: */
    }
    *f *= .5;
    return 0;
L390:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
	a = 0.;
	b = 0.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    w = (doublereal) j * u;
	    if (j <= ka) {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		a += w * (d__1 * (d__1 * d__1));
	    } else {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		b += (1. - w) * (d__1 * (d__1 * d__1));
	    }
/* L391: */
	}
	fa = x[ka] + u * ((1. - v) * a + v * b) / 2.;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L393: */
    }
    *f *= .5;
    return 0;
L400:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = (3. - x[ka] * 2.) * x[ka] + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 2.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L401: */
    }
    *f *= .5;
    return 0;
L450:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
	fa = (3. - x[i__] * 2.) * x[i__] + 1.;
	if (i__ > 1) {
	    fa -= x[i__ - 1];
	}
	if (i__ < *n) {
	    fa -= x[i__ + 1];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L451: */
    }
    *f *= .5;
    return 0;
L460:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
/* Computing 2nd power */
	d__1 = x[i__];
	fa = (d__1 * d__1 * 5. + 2.) * x[i__] + 1.;
/* Computing MAX */
	i__4 = 1, i__2 = i__ - 5;
/* Computing MIN */
	i__5 = *n, i__6 = i__ + 1;
	i__3 = min(i__5,i__6);
	for (j = max(i__4,i__2); j <= i__3; ++j) {
	    if (i__ != j) {
		fa += x[j] * (x[j] + 1.);
	    }
/* L461: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L463: */
    }
    *f *= .5;
    return 0;
L470:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
	    fa = x[i__] + x[i__ + 1] * ((5. - x[i__ + 1]) * x[i__ + 1] - 2.) 
		    - 13.;
	} else {
	    fa = x[i__] + x[i__ + 1] * ((x[i__ + 1] + 1.) * x[i__ + 1] - 14.) 
		    - 29.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L471: */
    }
    *f *= .5;
    return 0;
L480:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka % (*n / 2) + 1;
	j = i__ + *n / 2;
	empr28_1.m = *n * 5;
	if (ka <= empr28_1.m / 2) {
	    ia = 1;
	} else {
	    ia = 2;
	}
	ib = 5 - ka / (empr28_1.m / 4);
	ic = ka % 5 + 1;
	d__1 = pow_di(&x[i__], &ia) - pow_di(&x[j], &ib);
	fa = pow_di(&d__1, &ic);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L481: */
    }
    *f *= .5;
    return 0;
L490:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 5) / 6 << 1) - 1;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3];
	    fa = x[i__] + x[i__ + 1] * 3. * (x[i__ + 2] - 1.) + d__1 * d__1 - 
		    1.;
	} else if (ka % 6 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1];
/* Computing 2nd power */
	    d__2 = x[i__ + 2] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2 - x[i__ + 3] - 3.;
	} else if (ka % 6 == 3) {
	    fa = x[i__] * x[i__ + 1] - x[i__ + 2] * x[i__ + 3];
	} else if (ka % 6 == 4) {
	    fa = x[i__] * 2. * x[i__ + 2] + x[i__ + 1] * x[i__ + 3] - 3.;
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3];
/* Computing 2nd power */
	    d__2 = x[i__] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = x[i__] * x[i__ + 1] * x[i__ + 2] * x[i__ + 3] + d__1 * d__1 
		    - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L491: */
    }
    *f *= .5;
    return 0;
L500:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	j = ka % 2;
	if (j == 0) {
	    fa = 6. - exp(x[i__] * 2.) - exp(x[i__ + 1] * 2.);
	} else if (i__ == 1) {
	    fa = 4. - exp(x[i__]) - exp(x[i__ + 1]);
	} else if (i__ == *n) {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.);
	} else {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.) + 4. - exp(x[
		    i__]) - exp(x[i__ + 1]);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L501: */
    }
    *f *= .5;
    return 0;
L510:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (x[i__] * 2. / (d__1 * d__1 + 1.) - x[i__ + 1]) * 10.;
	} else {
	    fa = x[i__] - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L511: */
    }
    *f *= .5;
    return 0;
L520:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 5) / 6 * 3 - 2;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	} else if (ka % 6 == 2) {
	    fa = x[i__ + 2] - 1.;
	} else if (ka % 6 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = d__1 * d__1;
	} else if (ka % 6 == 4) {
/* Computing 3rd power */
	    d__1 = x[i__ + 4] - 1.;
	    fa = d__1 * (d__1 * d__1);
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = d__1 * d__1 * x[i__ + 3] + sin(x[i__ + 3] - x[i__ + 4]) - 
		    10.;
	} else {
/* Computing 2nd power */
	    d__2 = x[i__ + 2];
/* Computing 2nd power */
	    d__1 = d__2 * d__2 * x[i__ + 3];
	    fa = x[i__ + 1] + d__1 * d__1 - 20.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L521: */
    }
    *f *= .5;
    return 0;
L530:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	} else if (ka % 7 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 2]) * 10.;
	} else if (ka % 7 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2] - x[i__ + 3];
	    fa = d__1 * d__1;
	} else if (ka % 7 == 4) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - x[i__ + 4];
	    fa = d__1 * d__1;
	} else if (ka % 7 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = x[i__] + d__1 * d__1 + x[i__ + 2] - 30.;
	} else if (ka % 7 == 6) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2];
	    fa = x[i__ + 1] - d__1 * d__1 + x[i__ + 3] - 10.;
	} else {
	    fa = x[i__] * x[i__ + 4] - 10.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L531: */
    }
    *f *= .5;
    return 0;
L540:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		if (x[i__ + j] == 0.) {
		    x[i__ + j] = 1e-16;
		}
		d__2 = (d__1 = x[i__ + j], abs(d__1));
		d__3 = (doublereal) j / (doublereal) (k * l);
		a = a * d_sign(&c_b347, &x[i__ + j]) * pow_dd(&d__2, &d__3);
/* L541: */
	    }
	    fa += a;
/* L542: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L546: */
    }
    *f *= .5;
    return 0;
L550:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = 0.;
	    for (j = 1; j <= 4; ++j) {
		a += x[i__ + j] * ((doublereal) j / (doublereal) (k * l));
/* L551: */
	    }
	    fa += exp(a) * (doublereal) (k * k) / (doublereal) l;
/* L552: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L556: */
    }
    *f *= .5;
    return 0;
L560:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (j = 1; j <= 4; ++j) {
	    fa = fa + (doublereal) ((1 - (j % 2 << 1)) * l * j * j) * sin(x[
		    i__ + j]) + (doublereal) (l * l * j) * cos(x[i__ + j]);
/* L561: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L563: */
    }
    *f *= .5;
    return 0;
L570:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alfa = .5;
	if (ka == 1) {
	    fa = alfa - (1. - alfa) * x[3] - x[1] * (x[2] * 4. + 1.);
	} else if (ka == 2) {
	    fa = -(2. - alfa) * x[4] - x[2] * (x[1] * 4. + 1.);
	} else if (ka == *n - 1) {
	    fa = alfa * x[*n - 3] - x[*n - 1] * (x[*n] * 4. + 1.);
	} else if (ka == *n) {
	    fa = alfa * x[*n - 2] - (2. - alfa) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	} else if (ka % 2 == 1) {
	    fa = alfa * x[ka - 2] - (1. - alfa) * x[ka + 2] - x[ka] * (x[ka + 
		    1] * 4. + 1.);
	} else {
	    fa = alfa * x[ka - 2] - (2. - alfa) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L571: */
    }
    *f *= .5;
    return 0;
L580:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = (x[ka] - d__1 * d__1) * 4.;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4.;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L581: */
    }
    *f *= .5;
    return 0;
L590:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka + 1] * 2. + x[*n - 4] *
		     3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka - 1] - x[ka + 1] * 2. 
		    + x[*n - 4] * 3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 
		    - x[*n] + 1.;
	} else {
/* Computing 2nd power */
	    d__1 = x[*n];
	    fa = d__1 * d__1 * -2. + x[*n] * 3. - x[*n - 1] + x[*n - 4] * 3. 
		    - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L591: */
    }
    *f *= .5;
    return 0;
L600:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
/* Computing 3rd power */
	d__1 = x[ka] + v + 1.;
	fa = x[ka] * 2. + u * .5 * u * (d__1 * (d__1 * d__1)) + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L601: */
    }
    *f *= .5;
    return 0;
L610:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	} else if (ka % 7 == 2) {
	    fa = x[i__ + 1] + x[i__ + 2] - 2.;
	} else if (ka % 7 == 3) {
	    fa = x[i__ + 3] - 1.;
	} else if (ka % 7 == 4) {
	    fa = x[i__ + 4] - 1.;
	} else if (ka % 7 == 5) {
	    fa = x[i__] + x[i__ + 1] * 3.;
	} else if (ka % 7 == 6) {
	    fa = x[i__ + 2] + x[i__ + 3] - x[i__ + 4] * 2.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 4]) * 10.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L611: */
    }
    *f *= .5;
    return 0;
L620:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka / 2;
	if (ka == 1) {
	    fa = x[ka] - 1.;
	} else if (ka % 2 == 0) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__] - x[i__ + 1];
	    a = exp(-(d__1 * d__1)) * 2.;
/* Computing 2nd power */
	    d__1 = x[i__ + 1] - x[i__ + 2];
	    b = exp(d__1 * d__1 * -2.);
	    fa = a + b;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L621: */
    }
    *f *= .5;
    return 0;
L630:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing MIN */
/* Computing MAX */
	i__4 = ka % 13 - 2;
	i__3 = max(i__4,1);
	ia = min(i__3,7);
	ib = (ka + 12) / 13;
	i__ = ia + ib - 1;
	if (ia == 7) {
	    j = ib;
	} else {
	    j = ia + ib;
	}
	c__ = (doublereal) ia * 3. / 10.;
	a = 0.;
	b = exp(sin(c__) * x[j]);
	d__ = x[j] - sin(x[i__]) - 1. + empr28_1.y[0];
	e = cos(c__) + 1.;
	for (l = 0; l <= 6; ++l) {
	    if (ib + l != i__ && ib + l != j) {
		a = a + sin(x[ib + l]) - empr28_1.y[0];
	    }
/* L631: */
	}
/* Computing 2nd power */
	d__1 = d__;
	fa = e * (d__1 * d__1) + (x[i__] - 1.) * 5. * b + a * .5;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L633: */
    }
    *f *= .5;
    return 0;
L720:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a1 = .414214;
	if (ka == 1) {
	    fa = x[1] - (1. - x[1]) * x[3] - a1 * (x[2] * 4. + 1.);
	} else if (ka == 2) {
	    fa = -(1. - x[1]) * x[4] - a1 * (x[2] * 4. + 1.);
	} else if (ka == 3) {
	    fa = a1 * x[1] - (1. - x[1]) * x[5] - x[3] * (x[2] * 4. + 1.);
	} else if (ka <= *n - 2) {
	    fa = x[1] * x[ka - 2] - (1. - x[1]) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	} else if (ka == *n - 1) {
	    fa = x[1] * x[*n - 3] - x[*n - 1] * (x[*n - 2] * 4. + 1.);
	} else {
	    fa = x[1] * x[*n - 2] - (1. - x[1]) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L721: */
    }
    *f *= .5;
    return 0;
L740:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	} else if (ka < *n) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]) + x[ka] * 4. - x[ka 
		    - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	} else {
	    fa = x[ka] * 4. - x[ka - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L741: */
    }
    *f *= .5;
    return 0;
L750:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = 0.;
	    if (ka != 1) {
/* Computing 3rd power */
		d__1 = x[ka - 2] - x[ka];
		fa = fa - d__1 * (d__1 * d__1) * 6. + 10. - x[ka - 1] * 4. - 
			sin(x[ka - 2] - x[ka - 1] - x[ka]) * 2. * sin(x[ka - 
			2] + x[ka - 1] - x[ka]);
	    }
	    if (ka != *n) {
/* Computing 3rd power */
		d__1 = x[ka] - x[ka + 2];
		fa = fa + d__1 * (d__1 * d__1) * 3. - 5. + x[ka + 1] * 2. + 
			sin(x[ka] - x[ka + 1] - x[ka + 2]) * sin(x[ka] + x[ka 
			+ 1] - x[ka + 2]);
	    }
	} else {
	    ex = exp(x[ka - 1] - x[ka] - x[ka + 1]);
	    fa = x[ka] * 4. - (x[ka - 1] - x[ka + 1]) * ex - 3.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L751: */
    }
    *f *= .5;
    return 0;
L760:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.;
	    fa = d__1 * d__1;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] * 2. + 
		    1.;
	    fa = d__1 * d__1;
	} else {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.;
	    fa = d__1 * d__1;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L761: */
    }
    *f *= .5;
    return 0;
L780:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + x[ka + 1] - d__3 * d__3;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2];
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L781: */
    }
    *f *= .5;
    return 0;
L790:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
/* Computing 2nd power */
	    d__3 = x[ka + 3];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2 + x[ka 
		    + 2] - d__3 * d__3;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 + x[ka + 1] - 
		    d__4 * d__4 + x[ka + 2] - d__5 * d__5;
	} else if (ka < 4) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 + x[ka + 2] - d__6 * 
		    d__6;
	} else if (ka < *n - 2) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2] - d__6 * d__6;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2];
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] + d__4 * d__4 - x[ka - 3];
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
/* Computing 2nd power */
	    d__3 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2] + d__3 * d__3 - x[ka - 3];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L791: */
    }
    *f *= .5;
    return 0;
L810:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] + ((5. - x[ka + 1]) * x[ka + 1] - 2.) * x[ka + 1] - 
		    13.;
	} else {
	    fa = x[ka - 1] + ((x[ka] + 1.) * x[ka] - 14.) * x[ka] - 29.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L811: */
    }
    *f *= .5;
    return 0;
L830:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    a = exp(x[ka]) - x[ka + 1];
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	} else if (ka % 4 == 2) {
/* Computing 3rd power */
	    d__1 = x[ka] - x[ka + 1];
	    fa = d__1 * (d__1 * d__1) * 10.;
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka + 1];
	    a = d__1 * d__1 * 30. * fa;
	} else if (ka % 4 == 3) {
	    a = x[ka] - x[ka + 1];
/* Computing 2nd power */
	    d__1 = sin(a) / cos(a);
	    fa = d__1 * d__1;
	} else {
	    fa = x[ka] - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L831: */
    }
    *f *= .5;
    return 0;
L840:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    fa = x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	} else if (ka < *n) {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	} else {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L841: */
    }
    *f *= .5;
    return 0;
L860:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] * 1e4 * x[ka + 1] - 1.;
	} else {
	    fa = exp(-x[ka - 1]) + exp(-x[ka]) - 1.0001;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L861: */
    }
    *f *= .5;
    return 0;
L870:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -200. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
	} else if (ka % 4 == 2) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 200. + (x[ka] - 1.) * 20.2 + (x[ka + 
		    2] - 1.) * 19.8;
	} else if (ka % 4 == 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -180. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 180. + (x[ka] - 1.) * 20.2 + (x[ka - 
		    2] - 1.) * 19.8;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L871: */
    }
    *f *= .5;
    return 0;
L880:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    a = exp(cos((doublereal) ka * (x[ka] + x[ka + 1])));
	    b = a * (doublereal) ka * sin((doublereal) ka * (x[ka] + x[ka + 1]
		    ));
	    fa = x[ka] - a;
	} else if (ka < *n) {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])) * (
		    doublereal) ka;
	    fa = x[ka] - a;
	} else {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka])) * (doublereal) 
		    ka;
	    fa = x[ka] - a;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L881: */
    }
    *f *= .5;
    return 0;
L900:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2.) + d__1 * d__1 * .25;
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = 20. - x[ka - 1];
	    fa = x[ka] * 3. * (20. - x[ka] * 2. + x[ka - 1]) + d__1 * d__1 * 
		    .25;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2. + x[ka - 1]) + d__1 * 
		    d__1 * .25;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L901: */
    }
    *f *= .5;
    return 0;
L910:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka + 1];
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - x[ka + 1];
	} else {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L911: */
    }
    *f *= .5;
    return 0;
L920:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 6.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	if (ka > 1) {
	    fa -= x[ka - 1] * 4.;
	    a1 -= x[ka - 1];
	    a2 += x[ka - 1];
	    a3 += x[ka - 1] * 2.;
	}
	if (ka > 2) {
	    fa += x[ka - 2];
	    a3 -= x[ka - 2];
	}
	if (ka < *n - 1) {
	    fa += x[ka + 2];
	    a3 += x[ka + 2];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 4.;
	    a1 += x[ka + 1];
	    a2 += x[ka + 1];
	    a3 -= x[ka + 1] * 2.;
	}
	if (ka >= *n - 1) {
	    fa += 1.;
	    a3 += 1.;
	}
	if (ka >= *n) {
	    fa += -4.;
	    a1 += 1.;
	    a2 += 1.;
	    a3 += -2.;
	}
	fa -= empr28_1.par * .5 * (a1 * a2 - x[ka] * a3);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L921: */
    }
    *f *= .5;
    return 0;
L930:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (empr28_1.m + 1);
	if (ka <= empr28_1.m) {
	    j = ka + empr28_1.m;
	    fa = x[ka] * 6.;
	    a1 = 0.;
	    a2 = 0.;
	    if (ka == 1) {
		a1 += 1.;
	    }
	    if (ka > 1) {
		fa -= x[ka - 1] * 4.;
		a1 -= x[j - 1];
		a2 += x[ka - 1] * 2.;
	    }
	    if (ka > 2) {
		fa += x[ka - 2];
		a2 -= x[ka - 2];
	    }
	    if (ka < empr28_1.m - 1) {
		fa += x[ka + 2];
		a2 += x[ka + 2];
	    }
	    if (ka < empr28_1.m) {
		fa -= x[ka + 1] * 4.;
		a1 += x[j + 1];
		a2 -= x[ka + 1] * 2.;
	    }
	    if (ka == empr28_1.m) {
		a1 += 1.;
	    }
/* Computing 2nd power */
	    d__1 = h__;
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a2 + x[j] * a1 * (d__1 * 
		    d__1));
	} else {
	    j = ka - empr28_1.m;
	    fa = x[ka] * -2.;
	    a1 = 0.;
	    a2 = 0.;
	    if (j == 1) {
		a2 += 1.;
	    }
	    if (j > 1) {
		fa += x[ka - 1];
		a1 -= x[j - 1];
		a2 -= x[ka - 1];
	    }
	    if (j < empr28_1.m) {
		fa += x[ka + 1];
		a1 += x[j + 1];
		a2 += x[ka + 1];
	    }
	    if (j == empr28_1.m) {
		a2 += 1.;
	    }
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a1 + x[j] * a2);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L931: */
    }
    *f *= .5;
    return 0;
L940:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 4. - empr28_1.par * exp(x[ka]);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L941: */
    }
    *f *= .5;
    return 0;
L950:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 3rd power */
	d__1 = x[ka];
/* Computing 2nd power */
	d__2 = (doublereal) i__;
/* Computing 2nd power */
	d__3 = (doublereal) j;
	fa = x[ka] * 4. + empr28_1.par * (d__1 * (d__1 * d__1)) / (
		empr28_1.par * (d__2 * d__2) + 1. + empr28_1.par * (d__3 * 
		d__3));
	if (i__ == 1) {
	    fa += -1.;
	}
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (i__ == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) j / (doublereal) (empr28_1.m + 1))
		    ;
	}
	if (j == 1) {
	    fa += -1.;
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) i__ / (doublereal) (empr28_1.m + 
		    1));
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L951: */
    }
    *f *= .5;
    return 0;
L960:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = (doublereal) i__ / (doublereal) (empr28_1.m + 1);
	a2 = (doublereal) j / (doublereal) (empr28_1.m + 1);
/* Computing 2nd power */
	d__1 = a1 - .25;
/* Computing 2nd power */
	d__2 = a2 - .75;
	fa = x[ka] * 4. - empr28_1.par * sin(pi * 2. * x[ka]) - (d__1 * d__1 
		+ d__2 * d__2) * 1e4 * empr28_1.par;
	if (i__ == 1) {
	    fa = fa - x[ka + 1] - empr28_1.par * sin(pi * x[ka + 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    fa = fa - x[ka + 1] - x[ka - 1] - empr28_1.par * sin(pi * (x[ka + 
		    1] - x[ka - 1]) * (doublereal) (empr28_1.m + 1));
	}
	if (i__ == empr28_1.m) {
	    fa = fa - x[ka - 1] + empr28_1.par * sin(pi * x[ka - 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == 1) {
	    fa = fa - x[ka + empr28_1.m] - empr28_1.par * sin(pi * x[ka + 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
	if (j > 1 && j < empr28_1.m) {
	    fa = fa - x[ka + empr28_1.m] - x[ka - empr28_1.m] - empr28_1.par *
		     sin(pi * (x[ka + empr28_1.m] - x[ka - empr28_1.m]) * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == empr28_1.m) {
	    fa = fa - x[ka - empr28_1.m] + empr28_1.par * sin(pi * x[ka - 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L961: */
    }
    *f *= .5;
    return 0;
L970:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 2nd power */
	d__1 = x[ka];
	fa = d__1 * d__1 * 8.;
	if (i__ == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - 1.;
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + 1.) - d__1 * d__1 * .5 - d__2 
		    * d__2 * 1.5 * (x[ka + 1] - 1.) * empr28_1.par;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + x[ka - 1]) - d__1 * d__1 * .5 
		    - d__2 * d__2 * 1.5 * (x[ka + 1] - x[ka - 1]) * 
		    empr28_1.par;
	}
	if (i__ == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * x[ka - 1] - d__1 * d__1 * .5 + d__2 * d__2 
		    * 1.5 * x[ka - 1] * empr28_1.par;
	}
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - 1.;
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + 1.) - d__1 * d__1 * 
		    .5;
	}
	if (j > 1 && j < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + x[ka - empr28_1.m]) 
		    - d__1 * d__1 * .5;
	}
	if (j == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * x[ka - empr28_1.m] - d__1 * d__1 * .5;
	}
	if (i__ == 1 && j == 1) {
	    fa -= empr28_1.par / (doublereal) (empr28_1.m + 1);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L971: */
    }
    *f *= .5;
    return 0;
L980:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a3 = 0.;
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = empr28_1.par * (doublereal) i__;
	a2 = empr28_1.par * (doublereal) j;
/* Computing 2nd power */
	d__1 = empr28_1.par;
	fa = x[ka] * 4. - a1 * 2e3 * a2 * (1. - a1) * (1. - a2) * (d__1 * 
		d__1);
	if (i__ > 1) {
	    fa -= x[ka - 1];
	    a3 -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	    a3 += x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	    a3 -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	    a3 += x[ka + empr28_1.m];
	}
	fa += empr28_1.par * 20. * a3 * x[ka];
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L981: */
    }
    *f *= .5;
    return 0;
L990:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing MAX */
	d__1 = 0., d__2 = x[ka];
	d__3 = (doublereal) i__ / (doublereal) (empr28_1.m + 2) - .5;
	fa = x[ka] * 20. - empr28_1.par * max(d__1,d__2) - d_sign(&
		empr28_1.par, &d__3);
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L991: */
    }
    *f *= .5;
    return 0;
L800:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = .5 / (doublereal) (empr28_1.m + 2);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	fa = x[ka] * 20.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	a4 = 0.;
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	    a4 += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
		a3 += x[ka - empr28_1.m - 1];
		a4 += x[ka - empr28_1.m - 1];
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    a1 -= x[ka - empr28_1.m];
	    a4 -= x[ka - empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
		a3 -= x[ka - empr28_1.m + 1];
		a4 += x[ka - empr28_1.m + 1];
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
		a3 += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	    a2 -= x[ka - 1];
	    a3 -= x[ka - 1] * 4.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    a2 += x[ka + 1];
	    a3 += x[ka + 1] * 4.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
		a3 -= x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
		a3 += x[ka + empr28_1.m - 1];
		a4 -= x[ka + empr28_1.m - 1];
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    a1 += x[ka + empr28_1.m];
	    a4 += x[ka + empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
		a3 -= x[ka + empr28_1.m + 1];
		a4 -= x[ka + empr28_1.m + 1];
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	    a4 -= x[ka + empr28_1.m + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    if (i__ > 1) {
		fa = fa - h__ - h__;
		a3 -= h__;
		a4 += h__;
	    }
	    fa += h__ * 8.;
	    a1 -= h__;
	    a4 -= h__ * 4.;
	    if (i__ < empr28_1.m) {
		fa -= h__ * 2.;
		a3 += h__;
		a4 += h__;
	    }
	    fa += h__;
	    a4 -= h__;
	}
	if (j == empr28_1.m - 1) {
	    fa -= h__;
	    a4 += h__;
	}
	fa += empr28_1.par * .25 * (a1 * a3 - a2 * a4);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L802: */
    }
    *f *= .5;
    return 0;
L240:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	w = 0.;
	i__3 = *n - 1;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    w += (doublereal) ka / (doublereal) (ka + i__) * x[i__];
/* L241: */
	}
	fa = x[ka] - (.4 / (doublereal) (*n) * x[ka] * (w + .5 + (doublereal) 
		ka / (doublereal) (ka + *n) * .5 * x[*n]) + 1.);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L243: */
    }
    *f *= .5;
    return 0;
L410:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = 1. - x[1];
	} else {
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka - 1];
	    fa = (doublereal) (ka - 1) * 10. * (d__1 * d__1);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L411: */
    }
    *f *= .5;
    return 0;
L420:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = x[ka] - d__1 * d__1 * .1;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] - d__1 * d__1 * .1;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L421: */
    }
    *f *= .5;
    return 0;
L650:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = 0.;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
/* Computing 3rd power */
	    d__1 = x[j];
	    s += d__1 * (d__1 * d__1);
/* L651: */
	}
	fa = x[ka] - 1. / (doublereal) (*n << 1) * (s + (doublereal) ka);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L653: */
    }
    *f *= .5;
    return 0;
L660:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1 * exp(x[ka]);
	if (*n == 1) {
	    fa = x[ka] * -2. - s;
	} else if (ka == 1) {
	    fa = x[ka] * -2. + x[ka + 1] - s;
	} else if (ka == *n) {
	    fa = x[ka - 1] - x[ka] * 2. - s;
	} else {
	    fa = x[ka - 1] - x[ka] * 2. + x[ka + 1] - s;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L661: */
    }
    *f *= .5;
    return 0;
L670:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .1;
	if (*n == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1.;
	} else if (ka == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka + 1] * 2.;
	} else if (ka == *n) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1];
	} else {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1] - x[ka + 1] * 2.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L671: */
    }
    *f *= .5;
    return 0;
L680:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s1 = 1.;
	s2 = 1.;
	s3 = 1.;
	j1 = 3;
	j2 = 3;
	if (ka - j1 > 1) {
	    i1 = ka - j1;
	} else {
	    i1 = 1;
	}
	if (ka + j2 < *n) {
	    i2 = ka + j2;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__3 = i2;
	for (j = i1; j <= i__3; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		s = s + x[j] + d__1 * d__1;
	    }
/* L681: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = (s1 + s2 * (d__1 * d__1)) * x[ka] + 1. - s3 * s;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L683: */
    }
    *f *= .5;
    return 0;
L690:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = d__1 * d__1 - 1.;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = d__1 * d__1 + log(x[ka]) - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L691: */
    }
    *f *= .5;
    return 0;
L340:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = x[1];
	} else {
	    fa = cos(x[ka - 1]) + x[ka] - 1.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L341: */
    }
    *f *= .5;
    return 0;
L360:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1;
	if (*n == 1) {
	    fa = x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	} else if (ka == 1) {
	    fa = x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka]));
	} else if (ka == *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka])
		    );
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L361: */
    }
    *f *= .5;
    return 0;
L380:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka - 5 > 1) {
	    i1 = ka - 5;
	} else {
	    i1 = 1;
	}
	if (ka + 1 < *n) {
	    i2 = ka + 1;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__3 = i2;
	for (j = i1; j <= i__3; ++j) {
	    if (j != ka) {
		s += x[j] * (x[j] + 1.);
	    }
/* L381: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = x[ka] * (d__1 * d__1 * 5. + 2.) + 1. - s;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L383: */
    }
    *f *= .5;
    return 0;
L430:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alf = 5.;
	bet = 14.;
	gam = 3.;
	d__1 = (doublereal) ka - (doublereal) (*n) / 2.;
	fa = bet * *n * x[ka] + pow_dd(&d__1, &gam);
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) ka / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		fa += t * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf));
	    }
/* L431: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L433: */
    }
    *f *= .5;
    return 0;
L440:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	c__ = .5;
	h__ = 1. / (doublereal) (*n);
	fa = 1. - c__ * h__ / 4.;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    s = c__ * h__ * (doublereal) ka / (doublereal) (ka + j << 1);
	    if (j == *n) {
		s /= 2.;
	    }
	    fa -= s * x[j];
/* L441: */
	}
	fa = x[ka] * fa - 1.;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L445: */
    }
    *f *= .5;
    return 0;
L270:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 * 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - h__ * x[ka + 1];
	} else if (1 < ka && ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - 
		    h__ * (x[ka + 1] - x[ka - 1]);
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - .5 - t * (d__1 * d__1) - h__ * (.5 
		    - x[ka - 1]);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L271: */
    }
    *f *= .5;
    return 0;
L280:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .5;
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 / s;
	t1 = h__ * 2.;
	al = 0.;
	be = .5;
	s1 = 0.;
	i__3 = ka;
	for (j = 1; j <= i__3; ++j) {
	    if (j == 1) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - al) / t1);
	    }
	    if (1 < j && j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - x[j - 1]) / 
			t1);
	    }
	    if (j == *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (be - x[j - 1]) / t1);
	    }
/* L281: */
	}
	s1 = (1. - (doublereal) ka * h__) * s1;
	if (ka == *n) {
	    goto L283;
	}
	s2 = 0.;
	i__3 = *n;
	for (j = ka + 1; j <= i__3; ++j) {
	    if (j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (x[j + 1] 
			- x[j - 1]) / t1);
	    } else {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (be - x[j 
			- 1]) / t1);
	    }
/* L282: */
	}
	s1 += (doublereal) ka * s2;
L283:
	fa = x[ka] - (doublereal) ka * .5 * h__ - t * s1;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L288: */
    }
    *f *= .5;
    return 0;
L290:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a = -.009;
	b = .001;
	al = 0.;
	be = 25.;
	ga = 20.;
	ca = .3;
	cb = .3;
	h__ = (b - a) / (doublereal) (*n + 1);
	t = a + (doublereal) ka * h__;
/* Computing 2nd power */
	d__1 = h__;
	h__ = d__1 * d__1;
	s = (doublereal) ka / (doublereal) (*n + 1);
	u = al * (1. - s) + be * s + x[ka];
	ff = cb * exp(ga * (u - be)) - ca * exp(ga * (al - u));
	if (t <= 0.) {
	    ff += ca;
	} else {
	    ff -= cb;
	}
	if (*n == 1) {
	    fa = -al + x[ka] * 2. - be + h__ * ff;
	} else if (ka == 1) {
	    fa = -al + x[ka] * 2. - x[ka + 1] + h__ * ff;
	} else if (ka < *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + h__ * ff;
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - be + h__ * ff;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L291: */
    }
    *f *= .5;
    return 0;
L300:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	al1 = 0.;
	al2 = 0.;
	be1 = 0.;
	be2 = 0.;
	n1 = *n / 2;
	h__ = 1. / (doublereal) (n1 + 1);
	t = (doublereal) ka * h__;
	if (ka == 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al1;
	} else if (ka == n1 + 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al2;
	} else if (ka == n1) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be1;
	} else if (ka == *n) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be2;
	} else {
	    s1 = -x[ka - 1] + x[ka] * 2. - x[ka + 1];
	    b = 0.;
	}
	if (ka <= n1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[n1 + ka];
	    s2 = d__1 * d__1 + x[ka] + d__2 * d__2 * .1 - 1.2;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - n1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    s2 = d__1 * d__1 * .2 + d__2 * d__2 + x[ka] * 2. - .6;
	}
/* Computing 2nd power */
	d__1 = h__;
	fa = s1 + d__1 * d__1 * s2 - b;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L301: */
    }
    *f *= .5;
    return 0;
L710:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	la = 1;
	h__ = 1. / (doublereal) (nd + 1);
	h2 = la * h__ * h__;
	if (l == 1 && k == 1) {
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * exp(x[1]);
	}
	if (1 < l && l < nd && k == 1) {
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * exp(x[l]);
	}
	if (l == nd && k == 1) {
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * exp(x[nd]);
	}
	if (l == 1 && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == 1 && k == nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * exp(x[
		    ka]);
	}
	if (l == nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * exp(x[ka]);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L711: */
    }
    *f *= .5;
    return 0;
L820:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	h__ = 1. / (doublereal) (nd + 1);
	h2 = h__ * h__;
	if (l == 1 && k == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
/* Computing 2nd power */
	    d__2 = h__ + 1.;
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * (d__1 * d__1) - 24. / (
		    d__2 * d__2);
	}
	if (1 < l && l < nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[l];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 1.;
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * (d__1 * 
		    d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[nd];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (l == 1 && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 1.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == 1 && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (1 < l && l < nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * (d__1 
		    * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    24. / (d__2 * d__2);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * (d__1 * d__1);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L821: */
    }
    *f *= .5;
    return 0;
} /* tffu28_ */

/* SUBROUTINE TFGU28                ALL SYSTEMS                92/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 92/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  GRADIENTS OF MODEL FUNCTIONS FOR UNCONSTRAINED MINIMIZATION. */
/*  UNIVERSAL VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RI  G(N)  GRADIENG OF THE MODEL FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tfgu28_(integer *n, doublereal *x, doublereal *g, 
	integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */
    double exp(doublereal), cos(doublereal), sin(doublereal), d_sign(
	    doublereal *, doublereal *), pow_dd(doublereal *, doublereal *), 
	    log(doublereal), atan(doublereal), pow_di(doublereal *, integer *)
	    , sinh(doublereal), cosh(doublereal), sqrt(doublereal);

    /* Local variables */
    static doublereal a, b, c__, d__, e, h__;
    static integer i__, j, k, l;
    static doublereal p, q, r__, s, t, u, v, w, a1, a2, a3, a4, h2;
    static integer i1, i2, j1, j2, n1;
    static doublereal s1, s2, s3, t1, ca, cb, fa, be, ga;
    static integer ia, ib;
    static doublereal ff, al, fg;
    static integer ic, ka, la, nd;
    static doublereal pi, ex, sx[1000], be1, ga1[2], ga2[2], ga3[6], ga4[6], 
	    be2, al1, al2, d1s, d2s, alf, gam, bet, alfa;
    extern /* Subroutine */ int uxvset_(integer *, doublereal *, doublereal *)
	    ;

    /* Parameter adjustments */
    --g;
    --x;

    /* Function Body */
    pi = 3.14159265358979323846;
    uxvset_(n, &c_b413, &g[1]);
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
	case 11:  goto L110;
	case 12:  goto L120;
	case 13:  goto L130;
	case 14:  goto L140;
	case 15:  goto L150;
	case 16:  goto L160;
	case 17:  goto L170;
	case 18:  goto L180;
	case 19:  goto L190;
	case 20:  goto L200;
	case 21:  goto L210;
	case 22:  goto L220;
	case 23:  goto L230;
	case 24:  goto L250;
	case 25:  goto L310;
	case 26:  goto L320;
	case 27:  goto L330;
	case 28:  goto L350;
	case 29:  goto L370;
	case 30:  goto L390;
	case 31:  goto L400;
	case 32:  goto L450;
	case 33:  goto L460;
	case 34:  goto L470;
	case 35:  goto L480;
	case 36:  goto L490;
	case 37:  goto L500;
	case 38:  goto L510;
	case 39:  goto L520;
	case 40:  goto L530;
	case 41:  goto L540;
	case 42:  goto L550;
	case 43:  goto L560;
	case 44:  goto L570;
	case 45:  goto L580;
	case 46:  goto L590;
	case 47:  goto L600;
	case 48:  goto L610;
	case 49:  goto L620;
	case 50:  goto L630;
	case 51:  goto L720;
	case 52:  goto L740;
	case 53:  goto L750;
	case 54:  goto L760;
	case 55:  goto L780;
	case 56:  goto L790;
	case 57:  goto L810;
	case 58:  goto L830;
	case 59:  goto L840;
	case 60:  goto L860;
	case 61:  goto L870;
	case 62:  goto L880;
	case 63:  goto L900;
	case 64:  goto L910;
	case 65:  goto L920;
	case 66:  goto L930;
	case 67:  goto L940;
	case 68:  goto L950;
	case 69:  goto L960;
	case 70:  goto L970;
	case 71:  goto L980;
	case 72:  goto L990;
	case 73:  goto L800;
	case 74:  goto L240;
	case 75:  goto L410;
	case 76:  goto L420;
	case 77:  goto L650;
	case 78:  goto L660;
	case 79:  goto L670;
	case 80:  goto L680;
	case 81:  goto L690;
	case 82:  goto L340;
	case 83:  goto L360;
	case 84:  goto L380;
	case 85:  goto L430;
	case 86:  goto L440;
	case 87:  goto L270;
	case 88:  goto L280;
	case 89:  goto L290;
	case 90:  goto L300;
	case 91:  goto L710;
	case 92:  goto L820;
    }
L10:
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
	g[j - 1] = g[j - 1] + x[j - 1] * 400. * a + b * 2.;
	g[j] -= a * 200.;
/* L11: */
    }
    return 0;
L20:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
/* Computing 2nd power */
	d__1 = x[j + 1];
	c__ = d__1 * d__1 - x[j + 2];
	d__ = x[j + 1] - 1.;
	u = x[j] + x[j + 2] - 2.;
	v = x[j] - x[j + 2];
	g[j - 1] = g[j - 1] + x[j - 1] * 400. * a + b * 2.;
	g[j] = g[j] - a * 200. + u * 20. + v * .2;
	g[j + 1] = g[j + 1] + x[j + 1] * 360. * c__ + d__ * 2.;
	g[j + 2] = g[j + 2] - c__ * 180. + u * 20. - v * .2;
/* L21: */
    }
    return 0;
L30:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] + x[j] * 10.;
	b = x[j + 1] - x[j + 2];
	c__ = x[j] - x[j + 1] * 2.;
	d__ = x[j - 1] - x[j + 2];
/* Computing 3rd power */
	d__1 = d__;
	g[j - 1] = g[j - 1] + a * 2. + d__1 * (d__1 * d__1) * 40.;
/* Computing 3rd power */
	d__1 = c__;
	g[j] = g[j] + a * 20. + d__1 * (d__1 * d__1) * 4.;
/* Computing 3rd power */
	d__1 = c__;
	g[j + 1] = g[j + 1] - d__1 * (d__1 * d__1) * 8. + b * 10.;
/* Computing 3rd power */
	d__1 = d__;
	g[j + 2] = g[j + 2] - d__1 * (d__1 * d__1) * 40. - b * 10.;
/* L31: */
    }
    return 0;
L40:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = exp(x[j - 1]);
	b = a - x[j];
	d__ = x[j] - x[j + 1];
	p = x[j + 1] - x[j + 2];
	c__ = cos(p);
	q = sin(p) / cos(p);
	u = x[j - 1];
	v = x[j + 2] - 1.;
/* Computing 3rd power */
	d__1 = b;
	b = d__1 * (d__1 * d__1) * 4.;
/* Computing 5th power */
	d__1 = d__, d__2 = d__1, d__1 *= d__1;
	d__ = d__2 * (d__1 * d__1) * 600.;
/* Computing 3rd power */
	d__1 = q;
/* Computing 2nd power */
	d__2 = c__;
	q = d__1 * (d__1 * d__1) * 4. / (d__2 * d__2);
/* Computing 7th power */
	d__1 = u, d__2 = d__1, d__1 *= d__1, d__2 *= d__1;
	g[j - 1] = g[j - 1] + a * b + d__2 * (d__1 * d__1) * 8.;
	g[j] = g[j] + d__ - b;
	g[j + 1] = g[j + 1] + q - d__;
	g[j + 2] = g[j + 2] + v * 2. - q;
/* L41: */
    }
    return 0;
L50:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	g[j] += b * (3. - x[j] * 4.);
	if (j > 1) {
	    g[j - 1] -= b;
	}
	if (j < *n) {
	    g[j + 1] -= b;
	}
/* L51: */
    }
    return 0;
L60:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = (d__1 * d__1 * 5. + 2.) * x[j] + 1.;
/* Computing MAX */
	i__2 = 1, i__3 = j - 5;
/* Computing MIN */
	i__5 = *n, i__6 = j + 1;
	i__4 = min(i__5,i__6);
	for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
	    if (i__ != j) {
		a += x[i__] * (x[i__] + 1.);
	    }
/* L61: */
	}
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
/* Computing 2nd power */
	d__1 = x[j];
	g[j] += b * (d__1 * d__1 * 15. + 2.);
/* Computing MAX */
	i__4 = 1, i__2 = j - 5;
/* Computing MIN */
	i__5 = *n, i__6 = j + 1;
	i__3 = min(i__5,i__6);
	for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
	    if (i__ != j) {
		g[i__] += b * (x[i__] * 2. + 1.);
	    }
/* L62: */
	}
/* L63: */
    }
    return 0;
L70:
    p = 2.3333333333333335;
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	g[j] += b * (3. - x[j] * 4.);
	if (j > 1) {
	    g[j - 1] -= b;
	}
	if (j < *n) {
	    g[j + 1] -= b;
	}
	if (j <= k) {
	    a = x[j] + x[j + k];
	    d__1 = abs(a);
	    d__2 = p - 1.;
	    b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	    g[j] += b;
	    g[j + k] += b;
	}
/* L71: */
    }
    return 0;
L80:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L81;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
L81:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	}
	p = ((doublereal) (*n + j) - p) * 2. / (doublereal) (*n);
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L82;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
L82:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
	}
/* L83: */
    }
    return 0;
L90:
    k = *n / 2;
    q = 1. / (doublereal) (*n);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L91;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
L91:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
	}
	g[j] += q * (doublereal) j * sin(x[j]);
/* L92: */
    }
    return 0;
L100:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	q = (doublereal) j / 10. + 1.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L101;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
L101:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
	}
/* L102: */
    }
    return 0;
L110:
    p = -.002008;
    q = -.0019;
    r__ = -2.61e-4;
    i__1 = *n - 5;
    for (i__ = 0; i__ <= i__1; i__ += 5) {
	a = 1.;
	b = 0.;
	for (j = 1; j <= 5; ++j) {
	    a *= x[i__ + j];
/* Computing 2nd power */
	    d__1 = x[i__ + j];
	    b += d__1 * d__1;
/* L111: */
	}
	w = exp(a);
	a *= w;
	b = b - 10. - p;
	c__ = x[i__ + 2] * x[i__ + 3] - x[i__ + 4] * 5. * x[i__ + 5] - q;
/* Computing 3rd power */
	d__1 = x[i__ + 1];
/* Computing 3rd power */
	d__2 = x[i__ + 2];
	d__ = d__1 * (d__1 * d__1) + d__2 * (d__2 * d__2) + 1. - r__;
/* Computing 2nd power */
	d__1 = x[i__ + 1];
	g[i__ + 1] = g[i__ + 1] + a / x[i__ + 1] + (b * 2. * x[i__ + 1] + d__ 
		* 3. * (d__1 * d__1)) * 20.;
/* Computing 2nd power */
	d__1 = x[i__ + 2];
	g[i__ + 2] = g[i__ + 2] + a / x[i__ + 2] + (b * 2. * x[i__ + 2] + c__ 
		* x[i__ + 3] + d__ * 3. * (d__1 * d__1)) * 20.;
	g[i__ + 3] = g[i__ + 3] + a / x[i__ + 3] + (b * 2. * x[i__ + 3] + c__ 
		* x[i__ + 2]) * 20.;
	g[i__ + 4] = g[i__ + 4] + a / x[i__ + 4] + (b * 2. * x[i__ + 4] - c__ 
		* 5. * x[i__ + 5]) * 20.;
	g[i__ + 5] = g[i__ + 5] + a / x[i__ + 5] + (b * 2. * x[i__ + 5] - c__ 
		* 5. * x[i__ + 4]) * 20.;
/* L112: */
    }
    return 0;
L120:
    c__ = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] - 3.;
	b = x[j - 1] - x[j];
	c__ += a;
	g[j - 1] = g[j - 1] + a * 2e-4 - 1. + exp(b * 20.) * 20.;
	g[j] = g[j] + 1. - exp(b * 20.) * 20.;
/* L121: */
    }
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
	g[j - 1] += c__ * 2.;
/* L122: */
    }
    return 0;
L130:
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j];
	a = d__1 * d__1;
	if (a == 0.) {
	    a = 1e-60;
	}
/* Computing 2nd power */
	d__1 = x[j - 1];
	b = d__1 * d__1;
	if (b == 0.) {
	    b = 1e-60;
	}
	c__ = a + 1.;
	d__ = b + 1.;
	p = 0.;
	if (a > p) {
	    p = log(a);
	}
	q = 0.;
	if (b > q) {
	    q = log(b);
	}
	g[j - 1] += x[j - 1] * 2. * (c__ * pow_dd(&b, &a) + p * pow_dd(&a, &
		d__));
	g[j] += x[j] * 2. * (d__ * pow_dd(&a, &b) + q * pow_dd(&b, &c__));
/* L131: */
    }
    return 0;
L140:
    p = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
    d__1 = p;
    q = d__1 * d__1 * .5;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 3rd power */
	d__1 = x[j] + (doublereal) j * p + 1.;
	a = x[j] * 2. + q * (d__1 * (d__1 * d__1));
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = x[j] + (doublereal) j * p + 1.;
	g[j] += a * (q * 6. * pow_dd(&d__1, &c_b532) + 4.);
	if (j > 1) {
	    g[j - 1] -= a * 2.;
	}
	if (j < *n) {
	    g[j + 1] -= a * 2.;
	}
/* L141: */
    }
    return 0;
L150:
    p = 1. / (doublereal) (*n + 1);
    q = 2. / p;
    r__ = p * 2.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	a = x[j - 1] - x[j];
	g[j - 1] += q * (x[j - 1] * 2. - x[j]);
	g[j] -= q * x[j - 1];
	if (abs(a) <= 1e-6) {
	    g[j - 1] += r__ * exp(x[j]) * (a * (a / 8. + .33333333333333331) 
		    + .5);
	    g[j] += r__ * exp(x[j]) * (a * (a / 24. + .16666666666666666) + 
		    .5);
	} else {
	    b = exp(x[j - 1]) - exp(x[j]);
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] += r__ * (exp(x[j - 1]) * a - b) / (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = a;
	    g[j] -= r__ * (exp(x[j]) * a - b) / (d__1 * d__1);
	}
/* L151: */
    }
/* Computing 2nd power */
    d__1 = x[1];
    g[1] += r__ * (exp(x[1]) * (x[1] - 1.) + 1.) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[*n];
    g[*n] = g[*n] + q * 2. * x[*n] + r__ * (exp(x[*n]) * (x[*n] - 1.) + 1.) / 
	    (d__1 * d__1);
    return 0;
L160:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (doublereal) j * (1. - cos(x[j]));
	if (j > 1) {
	    a += (doublereal) j * sin(x[j - 1]);
	}
	if (j < *n) {
	    a -= (doublereal) j * sin(x[j + 1]);
	}
	a = (doublereal) j * sin(x[j]);
	g[j] += a;
	if (j > 1) {
	    g[j - 1] += (doublereal) j * cos(x[j - 1]);
	}
	if (j < *n) {
	    g[j + 1] -= (doublereal) j * cos(x[j + 1]);
	}
/* L161: */
    }
    return 0;
L170:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    g[j] = g[j] + x[j] * .5 / p + p * exp(x[j]);
	    g[j + 1] += x[j + 1] * .25 / p;
	} else if (j == *n) {
	    g[j] = g[j] + x[j] * .5 / p + p * exp(x[j]);
	    g[j - 1] += x[j - 1] * .25 / p;
	} else {
	    a = (x[j + 1] - x[j - 1]) * .25 / p;
	    g[j] += p * exp(x[j]);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L171: */
    }
    return 0;
L180:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = (doublereal) j * p;
	if (j == 1) {
	    g[j] = g[j] + x[j] / p - p * 2. * (x[j] + q);
	    g[j + 1] += x[j + 1] * .5 / p;
	} else if (j == *n) {
	    g[j] = g[j] + x[j] / p - p * 2. * (x[j] + q);
	    g[j - 1] += x[j - 1] * .5 / p;
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    g[j] -= p * 2. * (x[j] + q);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L181: */
    }
    return 0;
L190:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = exp((doublereal) j * 2. * p);
	if (j == 1) {
	    r__ = .33333333333333331;
	    a = (x[j + 1] - r__) * .5 / p;
	    g[j] = g[j] + p * 2. * (x[j] + q) + (x[j] - r__) / p;
	    g[j + 1] += a;
	} else if (j == *n) {
	    r__ = exp(2.) / 3.;
	    a = (x[j - 1] - r__) * .5 / p;
	    g[j] = g[j] + p * 2. * (x[j] + q) + (x[j] - r__) / p;
	    g[j - 1] += a;
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    g[j] += p * 2. * (x[j] + q);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L191: */
    }
    return 0;
L200:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = exp(d__1 * d__1 * -2.);
	if (j == 1) {
	    b = x[j + 1] * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] = g[j] + x[j] / p - x[j] * 4. * a * p * (d__1 * d__1 - 1.);
	    g[j + 1] += a * b;
	} else if (j == *n) {
	    b = x[j - 1] * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] = g[j] + x[j] / p * exp(-2.) - x[j] * 4. * a * p * (d__1 * 
		    d__1 - 1.);
	    g[j - 1] += a * b;
	} else {
	    b = (x[j + 1] - x[j - 1]) * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] -= x[j] * 4. * a * p * (d__1 * d__1 - 1.);
	    g[j - 1] -= a * b;
	    g[j + 1] += a * b;
	}
/* L201: */
    }
    return 0;
L210:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = (x[j + 1] - 1.) * .5 / p;
	    b = (x[j] - 1.) / p;
	    u = atan(a);
	    v = atan(b);
	    g[j] = g[j] + p * 2. * x[j] + v * .5;
	    g[j + 1] += u * .5;
	} else if (j == *n) {
	    a = (2. - x[j - 1]) * .5 / p;
	    b = (2. - x[j]) / p;
	    u = atan(a);
	    v = atan(b);
	    g[j] = g[j] + p * 2. * x[j] - v * .5;
	    g[j - 1] -= u * .5;
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    u = atan(a);
	    g[j] += p * 2. * x[j];
	    g[j - 1] -= u * .5;
	    g[j + 1] += u * .5;
	}
/* L211: */
    }
    return 0;
L220:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = x[j + 1] * .5 / p;
	    b = x[j] / p;
/* Computing 2nd power */
	    d__1 = a;
/* Computing 3rd power */
	    d__2 = b;
	    g[j] = g[j] + p * 200. * (x[j] - d__1 * d__1) + d__2 * (d__2 * 
		    d__2) * 200. - (1. - b);
/* Computing 2nd power */
	    d__1 = a;
	    g[j + 1] = g[j + 1] - (x[j] - d__1 * d__1) * 200. * a - (1. - a);
	} else if (j == *n) {
	    a = x[j - 1] * -.5 / p;
	    b = -x[j] / p;
/* Computing 2nd power */
	    d__1 = a;
/* Computing 3rd power */
	    d__2 = b;
	    g[j] = g[j] + p * 200. * (x[j] - d__1 * d__1) - d__2 * (d__2 * 
		    d__2) * 200. + (1. - b);
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] = g[j - 1] + (x[j] - d__1 * d__1) * 200. * a + (1. - a);
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
/* Computing 2nd power */
	    d__1 = a;
	    g[j] += p * 200. * (x[j] - d__1 * d__1);
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] = g[j - 1] + (x[j] - d__1 * d__1) * 200. * a + (1. - a);
/* Computing 2nd power */
	    d__1 = a;
	    g[j + 1] = g[j + 1] - (x[j] - d__1 * d__1) * 200. * a - (1. - a);
	}
/* L221: */
    }
    return 0;
L230:
    a1 = -1.;
    d1s = exp(.01);
    d2s = 1.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a1 += (doublereal) (*n - j + 1) * (d__1 * d__1);
/* L231: */
    }
    a = a1 * 4.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	g[j] = a * (doublereal) (*n - j + 1) * x[j];
	s1 = exp(x[j] / 100.);
	if (j > 1) {
	    s3 = s1 + s2 - d2s * (d1s - 1.);
	    g[j] += s1 * 1e-5 * (s3 + s1 - 1. / d1s) / 50.;
	    g[j - 1] += s2 * 1e-5 * s3 / 50.;
	}
	s2 = s1;
	d2s = d1s * d2s;
/* L232: */
    }
    g[1] += (x[1] - .2) * 2.;
    return 0;
L250:
    a = 1.;
    b = 0.;
    c__ = 0.;
    d__ = 0.;
    u = exp(x[*n]);
    v = exp(x[*n - 1]);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *n - 2) {
/* Computing 2nd power */
	    d__1 = x[j] + x[j + 1] * 2. + x[j + 2] * 10. - 1.;
	    b += d__1 * d__1;
/* Computing 2nd power */
	    d__1 = x[j] * 2. + x[j + 1] - 3.;
	    c__ += d__1 * d__1;
	}
/* Computing 2nd power */
	d__1 = x[j];
	d__ = d__ + d__1 * d__1 - (doublereal) (*n);
/* L251: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *n / 2) {
	    g[j] += (x[j] - 1.) * 2.;
	}
	if (j <= *n - 2) {
	    p = a * (u + c__) * (x[j] + x[j + 1] * 2. + x[j + 2] * 10. - 1.);
	    q = a * (v + b) * (x[j] * 2. + x[j + 1] - 3.);
	    g[j] = g[j] + p * 2. + q * 4.;
	    g[j + 1] = g[j + 1] + p * 4. + q * 2.;
	    g[j + 2] += p * 20.;
	}
	g[j] += d__ * 4. * x[j];
/* L252: */
    }
    g[*n - 1] += a * v * c__;
    g[*n] += a * u * b;
    return 0;
L310:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = (x[ka + 1] - d__1 * d__1) * 10.;
	    g[ka] -= x[ka] * 20. * fa;
	    g[ka + 1] += fa * 10.;
	} else {
	    fa = 1. - x[ka - 1];
	    g[ka - 1] -= fa;
	}
/* L311: */
    }
    return 0;
L320:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    fa = x[ka] + x[ka + 1] * 10.;
	    g[ka] += fa;
	    g[ka + 1] += fa * 10.;
	} else if (ka % 4 == 2) {
	    fa = (x[ka + 1] - x[ka + 2]) * 2.23606797749979;
	    g[ka + 1] += fa * 2.23606797749979;
	    g[ka + 2] -= fa * 2.23606797749979;
	} else if (ka % 4 == 3) {
	    a = x[ka - 1] - x[ka] * 2.;
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	    g[ka - 1] += a * 2. * fa;
	    g[ka] -= a * 4. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 3] - x[ka];
	    fa = d__1 * d__1 * 3.16227766016838;
	    a = (x[ka - 3] - x[ka]) * 2.;
	    g[ka - 3] += a * 3.16227766016838 * fa;
	    g[ka] -= a * 3.16227766016838 * fa;
	}
/* L321: */
    }
    return 0;
L330:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = (x[ka] - 1.) / 316.22776601683825;
	    g[ka] += fa * .0031622776601683764;
	} else {
	    fa = -.25;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
/* Computing 2nd power */
		d__1 = x[j];
		fa += d__1 * d__1;
/* L331: */
	    }
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		g[j] += x[j] * 2. * fa;
/* L332: */
	    }
	}
/* L333: */
    }
    return 0;
L350:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	} else {
	    fa = 0.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		fa += (doublereal) j * (x[j] - 1.);
/* L351: */
	    }
	    if (ka == *n + 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    g[j] += (doublereal) j * fa;
/* L352: */
		}
	    } else if (ka == *n + 2) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
/* Computing 3rd power */
		    d__1 = fa;
		    g[j] += (doublereal) j * 2. * (d__1 * (d__1 * d__1));
/* L353: */
		}
/* Computing 2nd power */
		d__1 = fa;
		fa = d__1 * d__1;
	    }
	}
/* L354: */
    }
    return 0;
L370:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < *n) {
	    a = 0.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		a += x[j];
/* L371: */
	    }
	    fa = x[ka] + a - (doublereal) (*n + 1);
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		g[j] += fa;
/* L372: */
	    }
	    g[ka] += fa;
	} else {
	    a = 1.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		a *= x[j];
/* L373: */
	    }
	    fa = a - 1.;
	    i__ = 0;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		b = x[j];
		if (b == 0. && i__ == 0) {
		    i__ = j;
		}
		if (i__ != j) {
		    a *= b;
		}
/* L374: */
	    }
	    if (i__ == 0) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    g[j] += a / x[j] * fa;
/* L375: */
		}
	    } else {
		g[i__] += a * fa;
	    }
	}
/* L376: */
    }
    return 0;
L390:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
	a = 0.;
	b = 0.;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    w = (doublereal) j * u;
	    if (j <= ka) {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		a += w * (d__1 * (d__1 * d__1));
	    } else {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		b += (1. - w) * (d__1 * (d__1 * d__1));
	    }
/* L391: */
	}
	fa = x[ka] + u * ((1. - v) * a + v * b) / 2.;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    w = (doublereal) j * u;
/* Computing 2nd power */
	    d__1 = x[j] + w + 1.;
	    a = d__1 * d__1;
	    if (j <= ka) {
		g[j] += u * 1.5 * (1. - v) * w * a * fa;
	    } else {
		g[j] += u * 1.5 * (1. - w) * v * a * fa;
	    }
/* L392: */
	}
	g[ka] += fa;
/* L393: */
    }
    return 0;
L400:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = (3. - x[ka] * 2.) * x[ka] + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 2.;
	}
	g[ka] += (3. - x[ka] * 4.) * fa;
	if (ka > 1) {
	    g[ka - 1] -= fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= fa * 2.;
	}
/* L401: */
    }
    return 0;
L450:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
	fa = (3. - x[i__] * 2.) * x[i__] + 1.;
	if (i__ > 1) {
	    fa -= x[i__ - 1];
	}
	if (i__ < *n) {
	    fa -= x[i__ + 1];
	}
	g[i__] += (3. - x[i__] * 4.) * fa;
	if (i__ > 1) {
	    g[i__ - 1] -= fa;
	}
	if (i__ < *n) {
	    g[i__ + 1] -= fa;
	}
/* L451: */
    }
    return 0;
L460:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
/* Computing 2nd power */
	d__1 = x[i__];
	fa = (d__1 * d__1 * 5. + 2.) * x[i__] + 1.;
/* Computing MAX */
	i__3 = 1, i__4 = i__ - 5;
/* Computing MIN */
	i__5 = *n, i__6 = i__ + 1;
	i__2 = min(i__5,i__6);
	for (j = max(i__3,i__4); j <= i__2; ++j) {
	    if (i__ != j) {
		fa += x[j] * (x[j] + 1.);
	    }
/* L461: */
	}
/* Computing MAX */
	i__2 = 1, i__3 = i__ - 5;
/* Computing MIN */
	i__5 = *n, i__6 = i__ + 1;
	i__4 = min(i__5,i__6);
	for (j = max(i__2,i__3); j <= i__4; ++j) {
	    if (i__ != j) {
		g[j] += (x[j] * 2. + 1.) * fa;
	    }
/* L462: */
	}
/* Computing 2nd power */
	d__1 = x[i__];
	g[i__] += (d__1 * d__1 * 15. + 2.) * fa;
/* L463: */
    }
    return 0;
L470:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
	    fa = x[i__] + x[i__ + 1] * ((5. - x[i__ + 1]) * x[i__ + 1] - 2.) 
		    - 13.;
	    g[i__] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    g[i__ + 1] += (x[i__ + 1] * 10. - d__1 * d__1 * 3. - 2.) * fa;
	} else {
	    fa = x[i__] + x[i__ + 1] * ((x[i__ + 1] + 1.) * x[i__ + 1] - 14.) 
		    - 29.;
	    g[i__] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    g[i__ + 1] += (x[i__ + 1] * 2. + d__1 * d__1 * 3. - 14.) * fa;
	}
/* L471: */
    }
    return 0;
L480:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka % (*n / 2) + 1;
	j = i__ + *n / 2;
	empr28_1.m = *n * 5;
	if (ka <= empr28_1.m / 2) {
	    ia = 1;
	} else {
	    ia = 2;
	}
	ib = 5 - ka / (empr28_1.m / 4);
	ic = ka % 5 + 1;
	d__1 = pow_di(&x[i__], &ia) - pow_di(&x[j], &ib);
	fa = pow_di(&d__1, &ic);
	a = (doublereal) ia;
	b = (doublereal) ib;
	c__ = (doublereal) ic;
	d__ = pow_di(&x[i__], &ia) - pow_di(&x[j], &ib);
	if (d__ != 0.) {
	    i__4 = ic - 1;
	    e = c__ * pow_di(&d__, &i__4);
	    if (x[i__] == 0. && ia <= 1) {
	    } else {
		i__4 = ia - 1;
		g[i__] += e * a * pow_di(&x[i__], &i__4) * fa;
	    }
	    if (x[j] == 0. && ib <= 1) {
	    } else {
		i__4 = ib - 1;
		g[j] -= e * b * pow_di(&x[j], &i__4) * fa;
	    }
	}
/* L481: */
    }
    return 0;
L490:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 5) / 6 << 1) - 1;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3];
	    fa = x[i__] + x[i__ + 1] * 3. * (x[i__ + 2] - 1.) + d__1 * d__1 - 
		    1.;
	    g[i__] += fa;
	    g[i__ + 1] += (x[i__ + 2] - 1.) * 3. * fa;
	    g[i__ + 2] += x[i__ + 1] * 3. * fa;
	    g[i__ + 3] += x[i__ + 3] * 2. * fa;
	} else if (ka % 6 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1];
/* Computing 2nd power */
	    d__2 = x[i__ + 2] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2 - x[i__ + 3] - 3.;
	    g[i__] += (x[i__] + x[i__ + 1]) * 2. * fa;
	    g[i__ + 1] += (x[i__] + x[i__ + 1]) * 2. * fa;
	    g[i__ + 2] += (x[i__ + 2] - 1.) * 2. * fa;
	    g[i__ + 3] -= fa;
	} else if (ka % 6 == 3) {
	    fa = x[i__] * x[i__ + 1] - x[i__ + 2] * x[i__ + 3];
	    g[i__] += x[i__ + 1] * fa;
	    g[i__ + 1] += x[i__] * fa;
	    g[i__ + 2] -= x[i__ + 3] * fa;
	    g[i__ + 3] -= x[i__ + 2] * fa;
	} else if (ka % 6 == 4) {
	    fa = x[i__] * 2. * x[i__ + 2] + x[i__ + 1] * x[i__ + 3] - 3.;
	    g[i__] += x[i__ + 2] * 2. * fa;
	    g[i__ + 1] += x[i__ + 3] * fa;
	    g[i__ + 2] += x[i__] * 2. * fa;
	    g[i__ + 3] += x[i__ + 1] * fa;
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3];
/* Computing 2nd power */
	    d__2 = x[i__] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2;
	    g[i__] += ((x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 2. + 
		    (x[i__] - 1.) * 2.) * fa;
	    g[i__ + 1] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	    g[i__ + 2] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	    g[i__ + 3] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = x[i__] * x[i__ + 1] * x[i__ + 2] * x[i__ + 3] + d__1 * d__1 
		    - 1.;
	    g[i__] += x[i__ + 1] * x[i__ + 2] * x[i__ + 3] * fa;
	    g[i__ + 1] += x[i__] * x[i__ + 2] * x[i__ + 3] * fa;
	    g[i__ + 2] += x[i__] * x[i__ + 1] * x[i__ + 3] * fa;
	    g[i__ + 3] += (x[i__] * x[i__ + 1] * x[i__ + 2] + (x[i__ + 3] - 
		    1.) * 2.) * fa;
	}
/* L491: */
    }
    return 0;
L500:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	j = ka % 2;
	if (j == 0) {
	    fa = 6. - exp(x[i__] * 2.) - exp(x[i__ + 1] * 2.);
	    g[i__] -= exp(x[i__] * 2.) * 2. * fa;
	    g[i__ + 1] -= exp(x[i__ + 1] * 2.) * 2. * fa;
	} else if (i__ == 1) {
	    fa = 4. - exp(x[i__]) - exp(x[i__ + 1]);
	    g[i__] -= exp(x[i__]) * fa;
	    g[i__ + 1] -= exp(x[i__ + 1]) * fa;
	} else if (i__ == *n) {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.);
	    g[i__ - 1] -= exp(x[i__ - 1] * 3.) * 3. * fa;
	    g[i__] -= exp(x[i__] * 3.) * 3. * fa;
	} else {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.) + 4. - exp(x[
		    i__]) - exp(x[i__ + 1]);
	    g[i__ - 1] -= exp(x[i__ - 1] * 3.) * 3. * fa;
	    g[i__] -= (exp(x[i__] * 3.) * 3. + exp(x[i__])) * fa;
	    g[i__ + 1] -= exp(x[i__ + 1]) * fa;
	}
/* L501: */
    }
    return 0;
L510:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (x[i__] * 2. / (d__1 * d__1 + 1.) - x[i__ + 1]) * 10.;
/* Computing 2nd power */
	    d__1 = x[i__];
/* Computing 2nd power */
	    d__3 = x[i__];
/* Computing 2nd power */
	    d__2 = d__3 * d__3 + 1.;
	    g[i__] += (1. - d__1 * d__1) * 20. / (d__2 * d__2) * fa;
	    g[i__ + 1] -= fa * 10.;
	} else {
	    fa = x[i__] - 1.;
	    g[i__] += fa;
	}
/* L511: */
    }
    return 0;
L520:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 5) / 6 * 3 - 2;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 6 == 2) {
	    fa = x[i__ + 2] - 1.;
	    g[i__ + 2] += fa;
	} else if (ka % 6 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = d__1 * d__1;
	    g[i__ + 3] += (x[i__ + 3] - 1.) * 2. * fa;
	} else if (ka % 6 == 4) {
/* Computing 3rd power */
	    d__1 = x[i__ + 4] - 1.;
	    fa = d__1 * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = x[i__ + 4] - 1.;
	    g[i__ + 4] += d__1 * d__1 * 3. * fa;
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = d__1 * d__1 * x[i__ + 3] + sin(x[i__ + 3] - x[i__ + 4]) - 
		    10.;
	    g[i__] += x[i__] * 2. * x[i__ + 3] * fa;
/* Computing 2nd power */
	    d__1 = x[i__];
	    g[i__ + 3] += (d__1 * d__1 + cos(x[i__ + 3] - x[i__ + 4])) * fa;
	    g[i__ + 4] -= cos(x[i__ + 3] - x[i__ + 4]) * fa;
	} else {
/* Computing 2nd power */
	    d__2 = x[i__ + 2];
/* Computing 2nd power */
	    d__1 = d__2 * d__2 * x[i__ + 3];
	    fa = x[i__ + 1] + d__1 * d__1 - 20.;
	    g[i__ + 1] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 2] * x[i__ + 3];
	    g[i__ + 2] += x[i__ + 2] * 4. * (d__1 * d__1) * fa;
/* Computing 4th power */
	    d__1 = x[i__ + 2], d__1 *= d__1;
	    g[i__ + 3] += d__1 * d__1 * 2. * x[i__ + 3] * fa;
	}
/* L521: */
    }
    return 0;
L530:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 7 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 2]) * 10.;
	    g[i__ + 1] += x[i__ + 1] * 20. * fa;
	    g[i__ + 2] -= fa * 10.;
	} else if (ka % 7 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2] - x[i__ + 3];
	    fa = d__1 * d__1;
	    g[i__ + 2] += (x[i__ + 2] - x[i__ + 3]) * 2. * fa;
	    g[i__ + 3] -= (x[i__ + 2] - x[i__ + 3]) * 2. * fa;
	} else if (ka % 7 == 4) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - x[i__ + 4];
	    fa = d__1 * d__1;
	    g[i__ + 3] += (x[i__ + 3] - x[i__ + 4]) * 2. * fa;
	    g[i__ + 4] -= (x[i__ + 3] - x[i__ + 4]) * 2. * fa;
	} else if (ka % 7 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = x[i__] + d__1 * d__1 + x[i__ + 2] - 30.;
	    g[i__] += fa;
	    g[i__ + 1] += x[i__ + 1] * 2. * fa;
	    g[i__ + 2] += fa;
	} else if (ka % 7 == 6) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2];
	    fa = x[i__ + 1] - d__1 * d__1 + x[i__ + 3] - 10.;
	    g[i__ + 1] += fa;
	    g[i__ + 2] -= x[i__ + 2] * 2. * fa;
	    g[i__ + 3] += fa;
	} else {
	    fa = x[i__] * x[i__ + 4] - 10.;
	    g[i__] += x[i__ + 4] * fa;
	    g[i__ + 4] += x[i__] * fa;
	}
/* L531: */
    }
    return 0;
L540:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		if (x[i__ + j] == 0.) {
		    x[i__ + j] = 1e-16;
		}
		d__2 = (d__1 = x[i__ + j], abs(d__1));
		d__3 = (doublereal) j / (doublereal) (k * l);
		a = a * d_sign(&c_b347, &x[i__ + j]) * pow_dd(&d__2, &d__3);
/* L541: */
	    }
	    fa += a;
/* L542: */
	}
	for (k = 1; k <= 3; ++k) {
	    a = (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		d__2 = (d__1 = x[i__ + j], abs(d__1));
		d__3 = (doublereal) j / (doublereal) (k * l);
		a = a * d_sign(&c_b347, &x[i__ + j]) * pow_dd(&d__2, &d__3);
/* L543: */
	    }
	    for (j = 1; j <= 4; ++j) {
		g[i__ + j] += (doublereal) j / (doublereal) (k * l) * a / x[
			i__ + j] * fa;
/* L544: */
	    }
/* L545: */
	}
/* L546: */
    }
    return 0;
L550:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = 0.;
	    for (j = 1; j <= 4; ++j) {
		a += x[i__ + j] * ((doublereal) j / (doublereal) (k * l));
/* L551: */
	    }
	    fa += exp(a) * (doublereal) (k * k) / (doublereal) l;
/* L552: */
	}
	for (k = 1; k <= 3; ++k) {
	    a = 0.;
	    for (j = 1; j <= 4; ++j) {
		a += x[i__ + j] * ((doublereal) j / (doublereal) (k * l));
/* L553: */
	    }
	    a = exp(a) * (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		g[i__ + j] += a * ((doublereal) j / (doublereal) (k * l)) * 
			fa;
/* L554: */
	    }
/* L555: */
	}
/* L556: */
    }
    return 0;
L560:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (j = 1; j <= 4; ++j) {
	    fa = fa + (doublereal) ((1 - (j % 2 << 1)) * l * j * j) * sin(x[
		    i__ + j]) + (doublereal) (l * l * j) * cos(x[i__ + j]);
/* L561: */
	}
	for (j = 1; j <= 4; ++j) {
	    g[i__ + j] += ((doublereal) ((1 - (j % 2 << 1)) * l * j * j) * 
		    cos(x[i__ + j]) - (doublereal) (l * l * j) * sin(x[i__ + 
		    j])) * fa;
/* L562: */
	}
/* L563: */
    }
    return 0;
L570:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alfa = .5;
	if (ka == 1) {
	    fa = alfa - (1. - alfa) * x[3] - x[1] * (x[2] * 4. + 1.);
	    g[1] -= (x[2] * 4. + 1.) * fa;
	    g[2] -= x[1] * 4. * fa;
	    g[3] += (alfa - 1.) * fa;
	} else if (ka == 2) {
	    fa = -(2. - alfa) * x[4] - x[2] * (x[1] * 4. + 1.);
	    g[1] -= x[2] * 4. * fa;
	    g[2] -= (x[1] * 4. + 1.) * fa;
	    g[4] += (alfa - 2.) * fa;
	} else if (ka == *n - 1) {
	    fa = alfa * x[*n - 3] - x[*n - 1] * (x[*n] * 4. + 1.);
	    g[*n - 3] += alfa * fa;
	    g[*n - 1] -= (x[*n] * 4. + 1.) * fa;
	    g[*n] -= x[*n - 1] * 4. * fa;
	} else if (ka == *n) {
	    fa = alfa * x[*n - 2] - (2. - alfa) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	    g[*n - 2] += alfa * fa;
	    g[*n - 1] -= x[*n] * 4. * fa;
	    g[*n] -= (x[*n - 1] * 4. + 1.) * fa;
	} else if (ka % 2 == 1) {
	    fa = alfa * x[ka - 2] - (1. - alfa) * x[ka + 2] - x[ka] * (x[ka + 
		    1] * 4. + 1.);
	    g[ka - 2] += alfa * fa;
	    g[ka] -= (x[ka + 1] * 4. + 1.) * fa;
	    g[ka + 1] -= x[ka] * 4. * fa;
	    g[ka + 2] += (alfa - 1.) * fa;
	} else {
	    fa = alfa * x[ka - 2] - (2. - alfa) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	    g[ka - 2] += alfa * fa;
	    g[ka - 1] -= x[ka] * 4. * fa;
	    g[ka] -= (x[ka - 1] * 4. + 1.) * fa;
	    g[ka + 2] += (alfa - 2.) * fa;
	}
/* L571: */
    }
    return 0;
L580:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = (x[ka] - d__1 * d__1) * 4.;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4.;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2.;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* L581: */
    }
    return 0;
L590:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka + 1] * 2. + x[*n - 4] *
		     3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] += fa * .5;
	    g[*n] -= fa;
	    g[ka] -= (x[ka] * 4. - 3.) * fa;
	    g[ka + 1] -= fa * 2.;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka - 1] - x[ka + 1] * 2. 
		    + x[*n - 4] * 3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 
		    - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] += fa * .5;
	    g[*n] -= fa;
	    g[ka - 1] -= fa;
	    g[ka] -= (x[ka] * 4. - 3.) * fa;
	    g[ka + 1] -= fa * 2.;
	} else {
/* Computing 2nd power */
	    d__1 = x[*n];
	    fa = d__1 * d__1 * -2. + x[*n] * 3. - x[*n - 1] + x[*n - 4] * 3. 
		    - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] -= fa * .5;
	    g[*n] -= (x[*n] * 4. - 2.) * fa;
	}
/* L591: */
    }
    return 0;
L600:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
/* Computing 3rd power */
	d__1 = x[ka] + v + 1.;
	fa = x[ka] * 2. + u * .5 * u * (d__1 * (d__1 * d__1)) + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1];
	}
/* Computing 2nd power */
	d__1 = u;
/* Computing 2nd power */
	d__2 = x[ka] + v + 1.;
	g[ka] += (d__1 * d__1 * 1.5 * (d__2 * d__2) + 2.) * fa;
	if (ka > 1) {
	    g[ka - 1] -= fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= fa;
	}
/* L601: */
    }
    return 0;
L610:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 7 == 2) {
	    fa = x[i__ + 1] + x[i__ + 2] - 2.;
	    g[i__ + 1] += fa;
	    g[i__ + 2] += fa;
	} else if (ka % 7 == 3) {
	    fa = x[i__ + 3] - 1.;
	    g[i__ + 3] += fa;
	} else if (ka % 7 == 4) {
	    fa = x[i__ + 4] - 1.;
	    g[i__ + 4] += fa;
	} else if (ka % 7 == 5) {
	    fa = x[i__] + x[i__ + 1] * 3.;
	    g[i__] += fa;
	    g[i__ + 1] += fa * 3.;
	} else if (ka % 7 == 6) {
	    fa = x[i__ + 2] + x[i__ + 3] - x[i__ + 4] * 2.;
	    g[i__ + 2] += fa;
	    g[i__ + 3] += fa;
	    g[i__ + 4] -= fa * 2.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 4]) * 10.;
	    g[i__ + 1] += x[i__ + 1] * 20. * fa;
	    g[i__ + 4] -= fa * 10.;
	}
/* L611: */
    }
    return 0;
L620:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka / 2;
	if (ka == 1) {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	} else if (ka % 2 == 0) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__] - x[i__ + 1];
	    a = exp(-(d__1 * d__1)) * 2.;
/* Computing 2nd power */
	    d__1 = x[i__ + 1] - x[i__ + 2];
	    b = exp(d__1 * d__1 * -2.);
	    fa = a + b;
	    g[i__] -= (x[i__] - x[i__ + 1]) * 2. * a * fa;
	    g[i__ + 1] += ((x[i__] - x[i__ + 1]) * 2. * a - (x[i__ + 1] - x[
		    i__ + 2]) * 4. * b) * fa;
	    g[i__ + 2] += (x[i__ + 1] - x[i__ + 2]) * 4. * b * fa;
	}
/* L621: */
    }
    return 0;
L630:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing MIN */
/* Computing MAX */
	i__2 = ka % 13 - 2;
	i__4 = max(i__2,1);
	ia = min(i__4,7);
	ib = (ka + 12) / 13;
	i__ = ia + ib - 1;
	if (ia == 7) {
	    j = ib;
	} else {
	    j = ia + ib;
	}
	c__ = (doublereal) ia * 3. / 10.;
	a = 0.;
	b = exp(sin(c__) * x[j]);
	d__ = x[j] - sin(x[i__]) - 1. + empr28_1.y[0];
	e = cos(c__) + 1.;
	for (l = 0; l <= 6; ++l) {
	    if (ib + l != i__ && ib + l != j) {
		a = a + sin(x[ib + l]) - empr28_1.y[0];
	    }
/* L631: */
	}
/* Computing 2nd power */
	d__1 = d__;
	fa = e * (d__1 * d__1) + (x[i__] - 1.) * 5. * b + a * .5;
	g[i__] -= (d__ * 2. * e * cos(x[i__]) - b * 5.) * fa;
	g[j] += (d__ * 2. * e + (x[i__] - 1.) * 5. * b * sin(c__)) * fa;
	for (l = 0; l <= 6; ++l) {
	    if (ib + l != i__ && ib + l != j) {
		g[ib + l] += cos(x[ib + l]) * .5 * fa;
	    }
/* L632: */
	}
/* L633: */
    }
    return 0;
L720:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a1 = .414214;
	if (ka == 1) {
	    fa = x[1] - (1. - x[1]) * x[3] - a1 * (x[2] * 4. + 1.);
	    g[1] += (x[3] + 1.) * fa;
	    g[2] -= a1 * 4. * fa;
	    g[3] -= (1. - x[1]) * fa;
	} else if (ka == 2) {
	    fa = -(1. - x[1]) * x[4] - a1 * (x[2] * 4. + 1.);
	    g[1] += x[4] * fa;
	    g[2] -= a1 * 4. * fa;
	    g[4] -= (1. - x[1]) * fa;
	} else if (ka == 3) {
	    fa = a1 * x[1] - (1. - x[1]) * x[5] - x[3] * (x[2] * 4. + 1.);
	    g[1] += (a1 + x[5]) * fa;
	    g[2] -= x[3] * 4. * fa;
	    g[3] -= (x[2] * 4. + 1.) * fa;
	    g[5] -= (1. - x[1]) * fa;
	} else if (ka <= *n - 2) {
	    fa = x[1] * x[ka - 2] - (1. - x[1]) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	    g[1] += (x[ka - 2] + x[ka + 2]) * fa;
	    g[ka - 2] += x[1] * fa;
	    g[ka - 1] -= x[ka] * 4. * fa;
	    g[ka] -= (x[ka - 1] * 4. + 1.) * fa;
	    g[ka + 2] -= (1. - x[1]) * fa;
	} else if (ka == *n - 1) {
	    fa = x[1] * x[*n - 3] - x[*n - 1] * (x[*n - 2] * 4. + 1.);
	    g[1] += x[*n - 3] * fa;
	    g[*n - 3] += x[1] * fa;
	    g[*n - 2] -= x[*n - 1] * 4. * fa;
	    g[*n - 1] -= (x[*n - 2] * 4. + 1.) * fa;
	} else {
	    fa = x[1] * x[*n - 2] - (1. - x[1]) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	    g[1] += (x[*n - 2] + 1.) * fa;
	    g[*n - 2] += x[1] * fa;
	    g[*n - 1] -= x[*n] * 4. * fa;
	    g[*n] -= (x[*n - 1] * 4. + 1.) * fa;
	}
/* L721: */
    }
    return 0;
L740:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d1s = cos(x[ka] - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d2s = sin(x[ka] - x[ka + 1]) * cos(x[ka] + x[ka + 1]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 9. + d1s + d2s) * fa;
	    g[ka + 1] += (2. - d1s + d2s) * fa;
	} else if (ka < *n) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]) + x[ka] * 4. - x[ka 
		    - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	    d1s = cos(x[ka] - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d2s = sin(x[ka] - x[ka + 1]) * cos(x[ka] + x[ka + 1]);
	    ex = exp(x[ka - 1] - x[ka]);
	    g[ka - 1] -= (ex + x[ka - 1] * ex) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 9. + d1s + d2s + 4. + x[ka - 1] * ex) * 
		    fa;
	    g[ka + 1] += (2. - d1s + d2s) * fa;
	} else {
	    fa = x[ka] * 4. - x[ka - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	    ex = exp(x[ka - 1] - x[ka]);
	    g[ka - 1] -= (ex + x[ka - 1] * ex) * fa;
	    g[ka] += (x[ka - 1] * ex + 4.) * fa;
	}
/* L741: */
    }
    return 0;
L750:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = 0.;
	    if (ka != 1) {
/* Computing 3rd power */
		d__1 = x[ka - 2] - x[ka];
		fa = fa - d__1 * (d__1 * d__1) * 6. + 10. - x[ka - 1] * 4. - 
			sin(x[ka - 2] - x[ka - 1] - x[ka]) * 2. * sin(x[ka - 
			2] + x[ka - 1] - x[ka]);
	    }
	    if (ka != *n) {
/* Computing 3rd power */
		d__1 = x[ka] - x[ka + 2];
		fa = fa + d__1 * (d__1 * d__1) * 3. - 5. + x[ka + 1] * 2. + 
			sin(x[ka] - x[ka + 1] - x[ka + 2]) * sin(x[ka] + x[ka 
			+ 1] - x[ka + 2]);
	    }
	    if (ka != 1) {
		d1s = cos(x[ka - 2] - x[ka - 1] - x[ka]) * sin(x[ka - 2] + x[
			ka - 1] - x[ka]);
		d2s = sin(x[ka - 2] - x[ka - 1] - x[ka]) * cos(x[ka - 2] + x[
			ka - 1] - x[ka]);
/* Computing 2nd power */
		d__1 = x[ka - 2] - x[ka];
		g[ka - 2] -= (d__1 * d__1 * 18. + (d1s + d2s) * 2.) * fa;
		g[ka - 1] -= (4. - (d1s - d2s) * 2.) * fa;
/* Computing 2nd power */
		d__1 = x[ka - 2] - x[ka];
		g[ka] += (d__1 * d__1 * 18. + (d1s + d2s) * 2.) * fa;
	    }
	    if (ka != *n) {
		d1s = cos(x[ka] - x[ka + 1] - x[ka + 2]) * sin(x[ka] + x[ka + 
			1] - x[ka + 2]);
		d2s = sin(x[ka] - x[ka + 1] - x[ka + 2]) * cos(x[ka] + x[ka + 
			1] - x[ka + 2]);
/* Computing 2nd power */
		d__1 = x[ka] - x[ka + 2];
		g[ka] += (d__1 * d__1 * 9. + d1s + d2s) * fa;
		g[ka + 1] += (2. - d1s + d2s) * fa;
/* Computing 2nd power */
		d__1 = x[ka] - x[ka + 2];
		g[ka + 2] -= (d__1 * d__1 * 9. + d1s + d2s) * fa;
	    }
	} else {
	    ex = exp(x[ka - 1] - x[ka] - x[ka + 1]);
	    fa = x[ka] * 4. - (x[ka - 1] - x[ka + 1]) * ex - 3.;
	    w = x[ka - 1] - x[ka + 1];
	    g[ka - 1] -= (ex + w * ex) * fa;
	    g[ka] += (w * ex + 4.) * fa;
	    g[ka + 1] += (ex + w * ex) * fa;
	}
/* L751: */
    }
    return 0;
L760:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.;
	    fa = d__1 * d__1;
	    g[1] += ((3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.) * 2. * (3. - 
		    h__ * 2. * x[1]) * fa;
	    g[2] -= ((3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.) * 4. * fa;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] * 2. + 
		    1.;
	    fa = d__1 * d__1;
	    g[ka - 1] -= ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] *
		     2. + 1.) * 2. * fa;
	    g[ka] += ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] * 2. 
		    + 1.) * 2. * (3. - h__ * 2. * x[ka]) * fa;
	    g[ka + 1] -= ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] *
		     2. + 1.) * 4. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.;
	    fa = d__1 * d__1;
	    g[*n - 1] -= ((3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.) * 2. * 
		    fa;
	    g[*n] += ((3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.) * 2. * (3. 
		    - h__ * 2. * x[*n]) * fa;
	}
/* L761: */
    }
    return 0;
L780:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + x[ka + 1] - d__3 * d__3;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4;
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2];
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2];
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* L781: */
    }
    return 0;
L790:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
/* Computing 2nd power */
	    d__3 = x[ka + 3];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2 + x[ka 
		    + 2] - d__3 * d__3;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 + x[ka + 1] - 
		    d__4 * d__4 + x[ka + 2] - d__5 * d__5;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < 4) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 + x[ka + 2] - d__6 * 
		    d__6;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < *n - 2) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2] - d__6 * d__6;
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] + d__4 * d__4 - x[ka - 3];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
/* Computing 2nd power */
	    d__3 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2] + d__3 * d__3 - x[ka - 3];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* L791: */
    }
    return 0;
L810:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] + ((5. - x[ka + 1]) * x[ka + 1] - 2.) * x[ka + 1] - 
		    13.;
	    g[ka] += fa;
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    g[ka + 1] += (x[ka + 1] * 10. - d__1 * d__1 * 3. - 2.) * fa;
	} else {
	    fa = x[ka - 1] + ((x[ka] + 1.) * x[ka] - 14.) * x[ka] - 29.;
	    g[ka - 1] += fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 3. + x[ka] * 2. - 14.) * fa;
	}
/* L811: */
    }
    return 0;
L830:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    a = exp(x[ka]) - x[ka + 1];
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	    g[ka] += a * 2. * exp(x[ka]) * fa;
	    g[ka + 1] -= a * 2. * fa;
	} else if (ka % 4 == 2) {
/* Computing 3rd power */
	    d__1 = x[ka] - x[ka + 1];
	    fa = d__1 * (d__1 * d__1) * 10.;
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka + 1];
	    a = d__1 * d__1 * 30. * fa;
	    g[ka] += a;
	    g[ka + 1] -= a;
	} else if (ka % 4 == 3) {
	    a = x[ka] - x[ka + 1];
/* Computing 2nd power */
	    d__1 = sin(a) / cos(a);
	    fa = d__1 * d__1;
/* Computing 3rd power */
	    d__1 = cos(a);
	    b = sin(a) * 2. / (d__1 * (d__1 * d__1)) * fa;
	    g[ka] += b;
	    g[ka + 1] -= b;
	} else {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	}
/* L831: */
    }
    return 0;
L840:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    fa = x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	    g[ka] += (x[ka] - 3.) * fa;
	    g[ka + 1] += fa * 2.;
	} else if (ka < *n) {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	    g[ka - 1] += fa;
	    g[ka] += (x[ka] - 3.) * fa;
	    g[ka + 1] += fa * 2.;
	} else {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1.;
	    g[ka - 1] += fa;
	    g[ka] += (x[ka] - 3.) * fa;
	}
/* L841: */
    }
    return 0;
L860:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] * 1e4 * x[ka + 1] - 1.;
	    g[ka] += x[ka + 1] * 1e4 * fa;
	    g[ka + 1] += x[ka] * 1e4 * fa;
	} else {
	    fa = exp(-x[ka - 1]) + exp(-x[ka]) - 1.0001;
	    g[ka - 1] -= exp(-x[ka - 1]) * fa;
	    g[ka] -= exp(-x[ka]) * fa;
	}
/* L861: */
    }
    return 0;
L870:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -200. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] -= ((x[ka + 1] - d__1 * d__1 * 3.) * 200. - 1.) * fa;
	    g[ka + 1] -= x[ka] * 200. * fa;
	} else if (ka % 4 == 2) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 200. + (x[ka] - 1.) * 20.2 + (x[ka + 
		    2] - 1.) * 19.8;
	    g[ka - 1] -= x[ka - 1] * 400. * fa;
	    g[ka] += fa * 220.2;
	    g[ka + 2] += fa * 19.8;
	} else if (ka % 4 == 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -180. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] -= ((x[ka + 1] - d__1 * d__1 * 3.) * 180. - 1.) * fa;
	    g[ka + 1] -= x[ka] * 180. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 180. + (x[ka] - 1.) * 20.2 + (x[ka - 
		    2] - 1.) * 19.8;
	    g[ka - 2] += fa * 19.8;
	    g[ka - 1] -= x[ka - 1] * 360. * fa;
	    g[ka] += fa * 200.2;
	}
/* L871: */
    }
    return 0;
L880:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    a = exp(cos((doublereal) ka * (x[ka] + x[ka + 1])));
	    b = a * (doublereal) ka * sin((doublereal) ka * (x[ka] + x[ka + 1]
		    ));
	    fa = x[ka] - a;
	    g[ka + 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	} else if (ka < *n) {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])) * (
		    doublereal) ka;
	    fa = x[ka] - a;
	    g[ka - 1] += b * fa;
	    g[ka + 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	} else {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka])) * (doublereal) 
		    ka;
	    fa = x[ka] - a;
	    g[ka - 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	}
/* L881: */
    }
    return 0;
L900:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2.) + d__1 * d__1 * .25;
	    g[ka] += (x[ka + 1] - x[ka] * 4.) * 3. * fa;
	    g[ka + 1] += (x[ka] * 3. + x[ka + 1] * .5) * fa;
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = 20. - x[ka - 1];
	    fa = x[ka] * 3. * (20. - x[ka] * 2. + x[ka - 1]) + d__1 * d__1 * 
		    .25;
	    g[ka - 1] += (x[ka] * 3. - (20. - x[ka - 1]) * .5) * fa;
	    g[ka] += (20. - x[ka] * 4. + x[ka - 1]) * 3. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2. + x[ka - 1]) + d__1 * 
		    d__1 * .25;
	    g[ka - 1] += (x[ka] * 3. - (x[ka + 1] - x[ka - 1]) * .5) * fa;
	    g[ka] += (x[ka + 1] - x[ka] * 4. + x[ka - 1]) * 3. * fa;
	    g[ka + 1] += (x[ka] * 3. + (x[ka + 1] - x[ka - 1]) * .5) * fa;
	}
/* L901: */
    }
    return 0;
L910:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka + 1];
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - x[ka + 1];
	    g[ka - 1] -= fa;
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka + 1] -= fa;
	} else {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - 1.;
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka - 1] -= fa;
	}
/* L911: */
    }
    return 0;
L920:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 6.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	if (ka > 1) {
	    fa -= x[ka - 1] * 4.;
	    a1 -= x[ka - 1];
	    a2 += x[ka - 1];
	    a3 += x[ka - 1] * 2.;
	}
	if (ka > 2) {
	    fa += x[ka - 2];
	    a3 -= x[ka - 2];
	}
	if (ka < *n - 1) {
	    fa += x[ka + 2];
	    a3 += x[ka + 2];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 4.;
	    a1 += x[ka + 1];
	    a2 += x[ka + 1];
	    a3 -= x[ka + 1] * 2.;
	}
	if (ka >= *n - 1) {
	    fa += 1.;
	    a3 += 1.;
	}
	if (ka >= *n) {
	    fa += -4.;
	    a1 += 1.;
	    a2 += 1.;
	    a3 += -2.;
	}
	fa -= empr28_1.par * .5 * (a1 * a2 - x[ka] * a3);
	g[ka] += fa * 6.;
	ga1[0] = 0.;
	ga1[1] = 0.;
	ga2[0] = 0.;
	ga2[1] = 0.;
	if (ka > 1) {
	    g[ka - 1] -= (4. - empr28_1.par * x[ka]) * fa;
	    ga1[0] = -1.;
	    ga2[0] = 1.;
	}
	if (ka > 2) {
	    g[ka - 2] += (1. - empr28_1.par * .5 * x[ka]) * fa;
	}
	if (ka < *n - 1) {
	    g[ka + 2] += (empr28_1.par * .5 * x[ka] + 1.) * fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= (empr28_1.par * x[ka] + 4.) * fa;
	    ga1[1] = 1.;
	    ga2[1] = 1.;
	}
	g[ka] += empr28_1.par * .5 * a3 * fa;
	if (ka > 1) {
	    g[ka - 1] -= empr28_1.par * .5 * (ga1[0] * a2 + a1 * ga2[0]) * fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= empr28_1.par * .5 * (ga1[1] * a2 + a1 * ga2[1]) * fa;
	}
/* L921: */
    }
    return 0;
L930:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (empr28_1.m + 1);
	if (ka <= empr28_1.m) {
	    j = ka + empr28_1.m;
	    fa = x[ka] * 6.;
	    a1 = 0.;
	    a2 = 0.;
	    if (ka == 1) {
		a1 += 1.;
	    }
	    if (ka > 1) {
		fa -= x[ka - 1] * 4.;
		a1 -= x[j - 1];
		a2 += x[ka - 1] * 2.;
	    }
	    if (ka > 2) {
		fa += x[ka - 2];
		a2 -= x[ka - 2];
	    }
	    if (ka < empr28_1.m - 1) {
		fa += x[ka + 2];
		a2 += x[ka + 2];
	    }
	    if (ka < empr28_1.m) {
		fa -= x[ka + 1] * 4.;
		a1 += x[j + 1];
		a2 -= x[ka + 1] * 2.;
	    }
	    if (ka == empr28_1.m) {
		a1 += 1.;
	    }
/* Computing 2nd power */
	    d__1 = h__;
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a2 + x[j] * a1 * (d__1 * 
		    d__1));
	} else {
	    j = ka - empr28_1.m;
	    fa = x[ka] * -2.;
	    a1 = 0.;
	    a2 = 0.;
	    if (j == 1) {
		a2 += 1.;
	    }
	    if (j > 1) {
		fa += x[ka - 1];
		a1 -= x[j - 1];
		a2 -= x[ka - 1];
	    }
	    if (j < empr28_1.m) {
		fa += x[ka + 1];
		a1 += x[j + 1];
		a2 += x[ka + 1];
	    }
	    if (j == empr28_1.m) {
		a2 += 1.;
	    }
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a1 + x[j] * a2);
	}
	if (ka <= empr28_1.m) {
	    g[ka] += fa * 6.;
	    if (ka > 1) {
		g[ka - 1] -= (4. - empr28_1.par * h__ * x[ka]) * fa;
/* Computing 3rd power */
		d__1 = h__;
		g[j - 1] -= empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * x[j] 
			* fa;
	    }
	    if (ka > 2) {
		g[ka - 2] += (1. - empr28_1.par * .5 * h__ * x[ka]) * fa;
	    }
	    if (ka < empr28_1.m - 1) {
		g[ka + 2] += (empr28_1.par * .5 * h__ * x[ka] + 1.) * fa;
	    }
	    if (ka < empr28_1.m) {
		g[ka + 1] -= (empr28_1.par * h__ * x[ka] + 4.) * fa;
/* Computing 3rd power */
		d__1 = h__;
		g[j + 1] += empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * x[j] 
			* fa;
	    }
	    g[ka] += empr28_1.par * .5 * h__ * a2 * fa;
/* Computing 3rd power */
	    d__1 = h__;
	    g[j] += empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * a1 * fa;
	} else {
	    g[ka] -= fa * 2.;
	    if (j > 1) {
		g[ka - 1] += (1. - empr28_1.par * .5 * h__ * x[j]) * fa;
		g[j - 1] -= empr28_1.par * .5 * h__ * x[ka] * fa;
	    }
	    if (j < empr28_1.m) {
		g[ka + 1] += (empr28_1.par * .5 * h__ * x[j] + 1.) * fa;
		g[j + 1] += empr28_1.par * .5 * h__ * x[ka] * fa;
	    }
	    g[ka] += empr28_1.par * .5 * h__ * a1 * fa;
	    g[j] += empr28_1.par * .5 * h__ * a2 * fa;
	}
/* L931: */
    }
    return 0;
L940:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 4. - empr28_1.par * exp(x[ka]);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
	g[ka] += (4. - empr28_1.par * exp(x[ka])) * fa;
	if (j > 1) {
	    g[ka - empr28_1.m] -= fa;
	}
	if (i__ > 1) {
	    g[ka - 1] -= fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= fa;
	}
/* L941: */
    }
    return 0;
L950:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 3rd power */
	d__1 = x[ka];
/* Computing 2nd power */
	d__2 = (doublereal) i__;
/* Computing 2nd power */
	d__3 = (doublereal) j;
	fa = x[ka] * 4. + empr28_1.par * (d__1 * (d__1 * d__1)) / (
		empr28_1.par * (d__2 * d__2) + 1. + empr28_1.par * (d__3 * 
		d__3));
	if (i__ == 1) {
	    fa += -1.;
	}
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (i__ == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) j / (doublereal) (empr28_1.m + 1))
		    ;
	}
	if (j == 1) {
	    fa += -1.;
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) i__ / (doublereal) (empr28_1.m + 
		    1));
	}
/* Computing 2nd power */
	d__1 = x[ka];
/* Computing 2nd power */
	d__2 = (doublereal) i__;
/* Computing 2nd power */
	d__3 = (doublereal) j;
	g[ka] += (empr28_1.par * 3. * (d__1 * d__1) / (empr28_1.par * (d__2 * 
		d__2) + 1. + empr28_1.par * (d__3 * d__3)) + 4.) * fa;
	if (j > 1) {
	    g[ka - empr28_1.m] -= fa;
	}
	if (i__ > 1) {
	    g[ka - 1] -= fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= fa;
	}
/* L951: */
    }
    return 0;
L960:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = (doublereal) i__ / (doublereal) (empr28_1.m + 1);
	a2 = (doublereal) j / (doublereal) (empr28_1.m + 1);
/* Computing 2nd power */
	d__1 = a1 - .25;
/* Computing 2nd power */
	d__2 = a2 - .75;
	fa = x[ka] * 4. - empr28_1.par * sin(pi * 2. * x[ka]) - (d__1 * d__1 
		+ d__2 * d__2) * 1e4 * empr28_1.par;
	if (i__ == 1) {
	    fa = fa - x[ka + 1] - empr28_1.par * sin(pi * x[ka + 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    fa = fa - x[ka + 1] - x[ka - 1] - empr28_1.par * sin(pi * (x[ka + 
		    1] - x[ka - 1]) * (doublereal) (empr28_1.m + 1));
	}
	if (i__ == empr28_1.m) {
	    fa = fa - x[ka - 1] + empr28_1.par * sin(pi * x[ka - 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == 1) {
	    fa = fa - x[ka + empr28_1.m] - empr28_1.par * sin(pi * x[ka + 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
	if (j > 1 && j < empr28_1.m) {
	    fa = fa - x[ka + empr28_1.m] - x[ka - empr28_1.m] - empr28_1.par *
		     sin(pi * (x[ka + empr28_1.m] - x[ka - empr28_1.m]) * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == empr28_1.m) {
	    fa = fa - x[ka - empr28_1.m] + empr28_1.par * sin(pi * x[ka - 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
	g[ka] += (4. - pi * 2. * empr28_1.par * cos(pi * 2. * x[ka])) * fa;
	if (i__ == 1) {
	    g[ka + 1] -= (pi * (doublereal) (empr28_1.m + 1) * empr28_1.par * 
		    cos(pi * x[ka + 1] * (doublereal) (empr28_1.m + 1)) + 1.) 
		    * fa;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    g[ka - 1] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + 1] - x[ka - 1]) * (
		    doublereal) (empr28_1.m + 1))) * fa;
	    g[ka + 1] -= (pi * (doublereal) (empr28_1.m + 1) * empr28_1.par * 
		    cos(pi * (x[ka + 1] - x[ka - 1]) * (doublereal) (
		    empr28_1.m + 1)) + 1.) * fa;
	}
	if (i__ == empr28_1.m) {
	    g[ka - 1] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka - 1] * (doublereal) (
		    empr28_1.m + 1))) * fa;
	}
	if (j == 1) {
	    g[ka + empr28_1.m] -= (pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka + empr28_1.m] * (doublereal) 
		    (empr28_1.m + 1)) + 1.) * fa;
	}
	if (j > 1 && j < empr28_1.m) {
	    g[ka - empr28_1.m] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m]) * (doublereal) (empr28_1.m + 1))) * fa;
	    g[ka + empr28_1.m] -= (pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m]) * (doublereal) (empr28_1.m + 1)) + 1.) * fa;
	}
	if (j == empr28_1.m) {
	    g[ka - empr28_1.m] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka - empr28_1.m] * (doublereal) 
		    (empr28_1.m + 1))) * fa;
	}
/* L961: */
    }
    return 0;
L970:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 2nd power */
	d__1 = x[ka];
	fa = d__1 * d__1 * 8.;
	if (i__ == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - 1.;
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + 1.) - d__1 * d__1 * .5 - d__2 
		    * d__2 * 1.5 * (x[ka + 1] - 1.) * empr28_1.par;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + x[ka - 1]) - d__1 * d__1 * .5 
		    - d__2 * d__2 * 1.5 * (x[ka + 1] - x[ka - 1]) * 
		    empr28_1.par;
	}
	if (i__ == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * x[ka - 1] - d__1 * d__1 * .5 + d__2 * d__2 
		    * 1.5 * x[ka - 1] * empr28_1.par;
	}
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - 1.;
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + 1.) - d__1 * d__1 * 
		    .5;
	}
	if (j > 1 && j < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + x[ka - empr28_1.m]) 
		    - d__1 * d__1 * .5;
	}
	if (j == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * x[ka - empr28_1.m] - d__1 * d__1 * .5;
	}
	if (i__ == 1 && j == 1) {
	    fa -= empr28_1.par / (doublereal) (empr28_1.m + 1);
	}
	g[ka] += x[ka] * 16. * fa;
	if (i__ == 1) {
	    g[ka] -= ((x[ka + 1] + 1.) * 2. + x[ka] * 3. * (x[ka + 1] - 1.) * 
		    empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka + 1] -= (x[ka] * 2. + (x[ka + 1] - 1.) + d__1 * d__1 * 1.5 * 
		    empr28_1.par) * fa;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    g[ka] -= ((x[ka + 1] + x[ka - 1]) * 2. + x[ka] * 3. * (x[ka + 1] 
		    - x[ka - 1]) * empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka - 1] -= (x[ka] * 2. - (x[ka + 1] - x[ka - 1]) - d__1 * d__1 *
		     1.5 * empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka + 1] -= (x[ka] * 2. + (x[ka + 1] - x[ka - 1]) + d__1 * d__1 *
		     1.5 * empr28_1.par) * fa;
	}
	if (i__ == empr28_1.m) {
	    g[ka] -= (x[ka - 1] * 2. - x[ka] * 3. * x[ka - 1] * empr28_1.par) 
		    * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka - 1] -= (x[ka] * 2. + x[ka - 1] - d__1 * d__1 * 1.5 * 
		    empr28_1.par) * fa;
	}
	if (j == 1) {
	    g[ka] -= (x[ka + empr28_1.m] + 1.) * 2. * fa;
	    g[ka + empr28_1.m] -= (x[ka] * 2. + (x[ka + empr28_1.m] - 1.)) * 
		    fa;
	}
	if (j > 1 && j < empr28_1.m) {
	    g[ka] -= (x[ka + empr28_1.m] + x[ka - empr28_1.m]) * 2. * fa;
	    g[ka - empr28_1.m] -= (x[ka] * 2. - (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m])) * fa;
	    g[ka + empr28_1.m] -= (x[ka] * 2. + (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m])) * fa;
	}
	if (j == empr28_1.m) {
	    g[ka] -= x[ka - empr28_1.m] * 2. * fa;
	    g[ka - empr28_1.m] -= (x[ka] * 2. + x[ka - empr28_1.m]) * fa;
	}
/* L971: */
    }
    return 0;
L980:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a3 = 0.;
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = empr28_1.par * (doublereal) i__;
	a2 = empr28_1.par * (doublereal) j;
/* Computing 2nd power */
	d__1 = empr28_1.par;
	fa = x[ka] * 4. - a1 * 2e3 * a2 * (1. - a1) * (1. - a2) * (d__1 * 
		d__1);
	if (i__ > 1) {
	    fa -= x[ka - 1];
	    a3 -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	    a3 += x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	    a3 -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	    a3 += x[ka + empr28_1.m];
	}
	fa += empr28_1.par * 20. * a3 * x[ka];
	g[ka] += fa * 4.;
	if (i__ > 1) {
	    g[ka - 1] -= (empr28_1.par * 20. * x[ka] + 1.) * fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= (1. - empr28_1.par * 20. * x[ka]) * fa;
	}
	if (j > 1) {
	    g[ka - empr28_1.m] -= (empr28_1.par * 20. * x[ka] + 1.) * fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= (1. - empr28_1.par * 20. * x[ka]) * fa;
	}
	g[ka] += empr28_1.par * 20. * a3 * fa;
/* L981: */
    }
    return 0;
L990:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing MAX */
	d__1 = 0., d__2 = x[ka];
	d__3 = (doublereal) i__ / (doublereal) (empr28_1.m + 2) - .5;
	fa = x[ka] * 20. - empr28_1.par * max(d__1,d__2) - d_sign(&
		empr28_1.par, &d__3);
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	}
	g[ka] += (20. - empr28_1.par) * fa;
	if (j > 2) {
	    g[ka - empr28_1.m - empr28_1.m] += fa;
	}
	if (j > 1) {
	    if (i__ > 1) {
		g[ka - empr28_1.m - 1] += fa * 2.;
	    }
	    g[ka - empr28_1.m] -= fa * 8.;
	    if (i__ < empr28_1.m) {
		g[ka - empr28_1.m + 1] += fa * 2.;
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		g[ka - 2] += fa;
	    }
	    g[ka - 1] -= fa * 8.;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa * 8.;
	    if (i__ < empr28_1.m - 1) {
		g[ka + 2] += fa;
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		g[ka + empr28_1.m - 1] += fa * 2.;
	    }
	    g[ka + empr28_1.m] -= fa * 8.;
	    if (i__ < empr28_1.m) {
		g[ka + empr28_1.m + 1] += fa * 2.;
	    }
	}
	if (j < empr28_1.m - 1) {
	    g[ka + empr28_1.m + empr28_1.m] += fa;
	}
/* L991: */
    }
    return 0;
L800:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = .5 / (doublereal) (empr28_1.m + 2);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	fa = x[ka] * 20.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	a4 = 0.;
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	    a4 += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
		a3 += x[ka - empr28_1.m - 1];
		a4 += x[ka - empr28_1.m - 1];
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    a1 -= x[ka - empr28_1.m];
	    a4 -= x[ka - empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
		a3 -= x[ka - empr28_1.m + 1];
		a4 += x[ka - empr28_1.m + 1];
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
		a3 += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	    a2 -= x[ka - 1];
	    a3 -= x[ka - 1] * 4.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    a2 += x[ka + 1];
	    a3 += x[ka + 1] * 4.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
		a3 -= x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
		a3 += x[ka + empr28_1.m - 1];
		a4 -= x[ka + empr28_1.m - 1];
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    a1 += x[ka + empr28_1.m];
	    a4 += x[ka + empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
		a3 -= x[ka + empr28_1.m + 1];
		a4 -= x[ka + empr28_1.m + 1];
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	    a4 -= x[ka + empr28_1.m + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    if (i__ > 1) {
		fa = fa - h__ - h__;
		a3 -= h__;
		a4 += h__;
	    }
	    fa += h__ * 8.;
	    a1 -= h__;
	    a4 -= h__ * 4.;
	    if (i__ < empr28_1.m) {
		fa -= h__ * 2.;
		a3 += h__;
		a4 += h__;
	    }
	    fa += h__;
	    a4 -= h__;
	}
	if (j == empr28_1.m - 1) {
	    fa -= h__;
	    a4 += h__;
	}
	fa += empr28_1.par * .25 * (a1 * a3 - a2 * a4);
	g[ka] += fa * 20.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	a4 = 0.;
	ga1[0] = 0.;
	ga1[1] = 0.;
	ga2[0] = 0.;
	ga2[1] = 0.;
	for (k = 1; k <= 6; ++k) {
	    ga3[k - 1] = 0.;
	    ga4[k - 1] = 0.;
/* L801: */
	}
	if (j > 2) {
	    g[ka - empr28_1.m - empr28_1.m] += fa;
	    ga4[0] += 1.;
	    a4 += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		g[ka - empr28_1.m - 1] += fa * 2.;
		ga3[0] += 1.;
		ga4[1] += 1.;
		a3 += x[ka - empr28_1.m - 1];
		a4 += x[ka - empr28_1.m - 1];
	    }
	    g[ka - empr28_1.m] -= fa * 8.;
	    ga1[0] += -1.;
	    a1 -= x[ka - empr28_1.m];
	    if (i__ < empr28_1.m) {
		g[ka - empr28_1.m + 1] += fa * 2.;
		ga3[1] += -1.;
		ga4[2] += 1.;
		a3 -= x[ka - empr28_1.m + 1];
		a4 += x[ka - empr28_1.m + 1];
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		g[ka - 2] += fa;
		ga3[2] += 1.;
		a3 += x[ka - 2];
	    }
	    g[ka - 1] -= fa * 8.;
	    ga2[0] += -1.;
	    a2 -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa * 8.;
	    ga2[1] += 1.;
	    a2 += x[ka + 1];
	    if (i__ < empr28_1.m - 1) {
		g[ka + 2] += fa;
		ga3[3] += -1.;
		a3 -= x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		g[ka + empr28_1.m - 1] += fa * 2.;
		ga3[4] += 1.;
		ga4[3] += -1.;
		a3 += x[ka + empr28_1.m - 1];
		a4 -= x[ka + empr28_1.m - 1];
	    }
	    g[ka + empr28_1.m] -= fa * 8.;
	    ga1[1] += 1.;
	    a1 += x[ka + empr28_1.m];
	    if (i__ < empr28_1.m) {
		g[ka + empr28_1.m + 1] += fa * 2.;
		ga3[5] += -1.;
		ga4[4] += -1.;
		a3 -= x[ka + empr28_1.m + 1];
		a4 -= x[ka + empr28_1.m + 1];
	    }
	}
	if (j < empr28_1.m - 1) {
	    g[ka + empr28_1.m + empr28_1.m] += fa;
	    ga4[5] += -1.;
	    a4 -= x[ka + empr28_1.m + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    if (i__ > 1) {
		a3 -= h__;
		a4 += h__;
	    }
	    a1 -= h__;
	    if (i__ < empr28_1.m) {
		a3 += h__;
		a4 += h__;
	    }
	    a4 -= h__;
	}
	if (j == empr28_1.m - 1) {
	    a4 += h__;
	}
	if (ka > empr28_1.m + empr28_1.m) {
	    g[ka - empr28_1.m - empr28_1.m] += empr28_1.par * .25 * (-a2 * 
		    ga4[0]) * fa;
	}
	if (ka > empr28_1.m + 1) {
	    g[ka - empr28_1.m - 1] += empr28_1.par * .25 * (a1 * ga3[0] - a2 *
		     ga4[1]) * fa;
	}
	if (ka > empr28_1.m) {
	    g[ka - empr28_1.m] += empr28_1.par * .25 * (ga1[0] * a3) * fa;
	}
	if (ka > empr28_1.m - 1) {
	    g[ka - empr28_1.m + 1] += empr28_1.par * .25 * (a1 * ga3[1] - a2 *
		     ga4[2]) * fa;
	}
	if (ka > 2) {
	    g[ka - 2] += empr28_1.par * .25 * (a1 * ga3[2]) * fa;
	}
	if (ka > 1) {
	    g[ka - 1] += empr28_1.par * .25 * (-ga2[0] * a4) * fa;
	}
	if (ka <= *n - 1) {
	    g[ka + 1] += empr28_1.par * .25 * (-ga2[1] * a4) * fa;
	}
	if (ka <= *n - 2) {
	    g[ka + 2] += empr28_1.par * .25 * (a1 * ga3[3]) * fa;
	}
	if (ka <= *n - empr28_1.m + 1) {
	    g[ka + empr28_1.m - 1] += empr28_1.par * .25 * (a1 * ga3[4] - a2 *
		     ga4[3]) * fa;
	}
	if (ka <= *n - empr28_1.m) {
	    g[ka + empr28_1.m] += empr28_1.par * .25 * (ga1[1] * a3) * fa;
	}
	if (ka <= *n - empr28_1.m - 1) {
	    g[ka + empr28_1.m + 1] += empr28_1.par * .25 * (a1 * ga3[5] - a2 *
		     ga4[4]) * fa;
	}
	if (ka <= *n - empr28_1.m - empr28_1.m) {
	    g[ka + empr28_1.m + empr28_1.m] += empr28_1.par * .25 * (-a2 * 
		    ga4[5]) * fa;
	}
/* L802: */
    }
    return 0;
L240:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	w = 0.;
	i__4 = *n - 1;
	for (i__ = 1; i__ <= i__4; ++i__) {
	    w += (doublereal) ka / (doublereal) (ka + i__) * x[i__];
/* L241: */
	}
	fa = x[ka] - (.4 / (doublereal) (*n) * x[ka] * (w + .5 + (doublereal) 
		ka / (doublereal) (ka + *n) * .5 * x[*n]) + 1.);
	w = w + .5 + (doublereal) ka * .5 / (doublereal) (ka + *n) * x[*n];
	i__4 = *n - 1;
	for (i__ = 1; i__ <= i__4; ++i__) {
	    g[i__] -= .4 / (doublereal) (*n) * x[ka] * (doublereal) ka / (
		    doublereal) (ka + i__) * fa;
/* L242: */
	}
	g[*n] -= .2 / (doublereal) (*n) * x[ka] * (doublereal) ka / (
		doublereal) (ka + *n) * fa;
	g[ka] += (1. - w * .4 / (doublereal) (*n)) * fa;
/* L243: */
    }
    return 0;
L410:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = 1. - x[1];
	    g[1] -= fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka - 1];
	    fa = (doublereal) (ka - 1) * 10. * (d__1 * d__1);
	    g[ka] += (doublereal) (ka - 1) * 20. * (x[ka] - x[ka - 1]) * fa;
	    g[ka - 1] -= (doublereal) (ka - 1) * 20. * (x[ka] - x[ka - 1]) * 
		    fa;
	}
/* L411: */
    }
    return 0;
L420:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = x[ka] - d__1 * d__1 * .1;
	    g[1] -= x[1] * .2 * fa;
	    g[*n] += fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] - d__1 * d__1 * .1;
	    g[ka] += fa;
	    g[ka + 1] -= x[ka + 1] * .2 * fa;
	}
/* L421: */
    }
    return 0;
L650:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = 0.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
/* Computing 3rd power */
	    d__1 = x[j];
	    s += d__1 * (d__1 * d__1);
/* L651: */
	}
	fa = x[ka] - 1. / (doublereal) (*n << 1) * (s + (doublereal) ka);
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j == ka) {
/* Computing 2nd power */
		d__1 = x[j];
		g[j] += (1. - d__1 * d__1 * 3. / ((doublereal) (*n) * 2.)) * 
			fa;
	    } else {
/* Computing 2nd power */
		d__1 = x[j];
		g[j] -= d__1 * d__1 * 3. / ((doublereal) (*n) * 2.) * fa;
	    }
/* L652: */
	}
/* L653: */
    }
    return 0;
L660:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1 * exp(x[ka]);
	if (*n == 1) {
	    fa = x[ka] * -2. - s;
	    g[ka] -= (s + 2.) * fa;
	} else if (ka == 1) {
	    fa = x[ka] * -2. + x[ka + 1] - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka + 1] += fa;
	} else if (ka == *n) {
	    fa = x[ka - 1] - x[ka] * 2. - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka - 1] += fa;
	} else {
	    fa = x[ka - 1] - x[ka] * 2. + x[ka + 1] - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka - 1] += fa;
	    g[ka + 1] += fa;
	}
/* L661: */
    }
    return 0;
L670:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .1;
	if (*n == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	} else if (ka == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka + 1] * 2.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka + 1] -= fa * 2.;
	} else if (ka == *n) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1];
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka - 1] -= fa;
	} else {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1] - x[ka + 1] * 2.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa * 2.;
	}
/* L671: */
    }
    return 0;
L680:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s1 = 1.;
	s2 = 1.;
	s3 = 1.;
	j1 = 3;
	j2 = 3;
	if (ka - j1 > 1) {
	    i1 = ka - j1;
	} else {
	    i1 = 1;
	}
	if (ka + j2 < *n) {
	    i2 = ka + j2;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		s = s + x[j] + d__1 * d__1;
	    }
/* L681: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = (s1 + s2 * (d__1 * d__1)) * x[ka] + 1. - s3 * s;
/* Computing 2nd power */
	d__1 = x[ka];
	g[ka] += (s1 + s2 * 3. * (d__1 * d__1)) * fa;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    g[j] -= s3 * (x[j] * 2. + 1.) * fa;
/* L682: */
	}
/* L683: */
    }
    return 0;
L690:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = d__1 * d__1 - 1.;
	    g[1] += x[1] * 2. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = d__1 * d__1 + log(x[ka]) - 1.;
	    g[ka - 1] += x[ka - 1] * 2. * fa;
	    g[ka] += 1. / x[ka] * fa;
	}
/* L691: */
    }
    return 0;
L340:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = x[1];
	    g[1] += fa;
	} else {
	    fa = cos(x[ka - 1]) + x[ka] - 1.;
	    g[ka] += fa;
	    g[ka - 1] -= sin(x[ka - 1]) * fa;
	}
/* L341: */
    }
    return 0;
L360:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1;
	if (*n == 1) {
	    fa = x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	} else if (ka == 1) {
	    fa = x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka == *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka - 1] -= fa;
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka])
		    );
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
/* L361: */
    }
    return 0;
L380:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka - 5 > 1) {
	    i1 = ka - 5;
	} else {
	    i1 = 1;
	}
	if (ka + 1 < *n) {
	    i2 = ka + 1;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
		s += x[j] * (x[j] + 1.);
	    }
/* L381: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = x[ka] * (d__1 * d__1 * 5. + 2.) + 1. - s;
/* Computing 2nd power */
	d__1 = x[ka];
	g[ka] += (d__1 * d__1 * 15. + 2.) * fa;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
		g[j] -= (x[j] * 2. + 1.) * fa;
	    }
/* L382: */
	}
/* L383: */
    }
    return 0;
L430:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alf = 5.;
	bet = 14.;
	gam = 3.;
	d__1 = (doublereal) ka - (doublereal) (*n) / 2.;
	fa = bet * *n * x[ka] + pow_dd(&d__1, &gam);
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) ka / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		fa += t * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf));
	    }
/* L431: */
	}
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) ka / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		d__3 = sin(s1);
		d__4 = alf - 1;
		d__5 = cos(s1);
		d__6 = alf - 1;
		g[j] += x[j] * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf) + 
			alf * pow_dd(&d__3, &d__4) * cos(s1) - alf * sin(s1) *
			 pow_dd(&d__5, &d__6)) / t * fa;
	    } else {
		g[j] += bet * *n * fa;
	    }
/* L432: */
	}
/* L433: */
    }
    return 0;
L440:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	c__ = .5;
	h__ = 1. / (doublereal) (*n);
	fa = 1. - c__ * h__ / 4.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    s = c__ * h__ * (doublereal) ka / (doublereal) (ka + j << 1);
	    if (j == *n) {
		s /= 2.;
	    }
	    fa -= s * x[j];
/* L441: */
	}
	fa = x[ka] * fa - 1.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    sx[j - 1] = c__ * h__ * (doublereal) ka / (doublereal) (ka + j << 
		    1);
/* L442: */
	}
	sx[*n - 1] *= .5;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (ka != j) {
		g[j] -= sx[j - 1] * x[ka] * fa;
	    } else {
		t = 1. - c__ * h__ / 4.;
		i__2 = *n;
		for (l = 1; l <= i__2; ++l) {
		    if (l == ka) {
			t -= sx[ka - 1] * 2. * x[ka];
		    } else {
			t -= sx[l - 1] * x[l];
		    }
/* L443: */
		}
		g[j] += t * fa;
	    }
/* L444: */
	}
/* L445: */
    }
    return 0;
L270:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 * 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - h__ * x[ka + 1];
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka + 1] -= (h__ + 1.) * fa;
	} else if (1 < ka && ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - 
		    h__ * (x[ka + 1] - x[ka - 1]);
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka - 1] -= (1. - h__) * fa;
	    g[ka + 1] -= (h__ + 1.) * fa;
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - .5 - t * (d__1 * d__1) - h__ * (.5 
		    - x[ka - 1]);
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka - 1] -= (1. - h__) * fa;
	}
/* L271: */
    }
    return 0;
L280:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .5;
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 / s;
	t1 = h__ * 2.;
	al = 0.;
	be = .5;
	s1 = 0.;
	i__4 = ka;
	for (j = 1; j <= i__4; ++j) {
	    if (j == 1) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - al) / t1);
	    }
	    if (1 < j && j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - x[j - 1]) / 
			t1);
	    }
	    if (j == *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (be - x[j - 1]) / t1);
	    }
/* L281: */
	}
	s1 = (1. - (doublereal) ka * h__) * s1;
	if (ka == *n) {
	    goto L283;
	}
	s2 = 0.;
	i__4 = *n;
	for (j = ka + 1; j <= i__4; ++j) {
	    if (j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (x[j + 1] 
			- x[j - 1]) / t1);
	    } else {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (be - x[j 
			- 1]) / t1);
	    }
/* L282: */
	}
	s1 += (doublereal) ka * s2;
L283:
	fa = x[ka] - (doublereal) ka * .5 * h__ - t * s1;
/* Computing 2nd power */
	d__1 = h__;
	s1 = d__1 * d__1 / s;
	s2 = 1. - (doublereal) ka * h__;
	i__4 = ka;
	for (j = 1; j <= i__4; ++j) {
	    sx[j - 1] = (doublereal) j * s2;
/* L284: */
	}
	if (ka == *n) {
	    goto L286;
	}
	i__4 = *n;
	for (j = ka + 1; j <= i__4; ++j) {
	    sx[j - 1] = (doublereal) ka * (1. - (doublereal) j * h__);
/* L285: */
	}
L286:
	g[1] -= s1 * (sx[0] * 2. * x[1] - sx[1] / t1) * fa;
	g[*n] -= s1 * (sx[*n - 2] / t1 + sx[*n - 1] * 2. * x[*n]) * fa;
	i__4 = *n - 1;
	for (j = 2; j <= i__4; ++j) {
	    g[j] -= s1 * ((sx[j - 2] - sx[j]) / t1 + sx[j - 1] * 2. * x[j]) * 
		    fa;
/* L287: */
	}
	g[ka] += fa;
	return 0;
/* L288: */
    }
    return 0;
L290:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a = -.009;
	b = .001;
	al = 0.;
	be = 25.;
	ga = 20.;
	ca = .3;
	cb = .3;
	h__ = (b - a) / (doublereal) (*n + 1);
	t = a + (doublereal) ka * h__;
/* Computing 2nd power */
	d__1 = h__;
	h__ = d__1 * d__1;
	s = (doublereal) ka / (doublereal) (*n + 1);
	u = al * (1. - s) + be * s + x[ka];
	ff = cb * exp(ga * (u - be)) - ca * exp(ga * (al - u));
	fg = ga * (cb * exp(ga * (u - be)) + ca * exp(ga * (al - u)));
	if (t <= 0.) {
	    ff += ca;
	} else {
	    ff -= cb;
	}
	if (*n == 1) {
	    fa = -al + x[ka] * 2. - be + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	} else if (ka == 1) {
	    fa = -al + x[ka] * 2. - x[ka + 1] + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka < *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - be + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka - 1] -= fa;
	}
/* L291: */
    }
    return 0;
L300:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	al1 = 0.;
	al2 = 0.;
	be1 = 0.;
	be2 = 0.;
	n1 = *n / 2;
	h__ = 1. / (doublereal) (n1 + 1);
	t = (doublereal) ka * h__;
	if (ka == 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al1;
	} else if (ka == n1 + 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al2;
	} else if (ka == n1) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be1;
	} else if (ka == *n) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be2;
	} else {
	    s1 = -x[ka - 1] + x[ka] * 2. - x[ka + 1];
	    b = 0.;
	}
	if (ka <= n1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[n1 + ka];
	    s2 = d__1 * d__1 + x[ka] + d__2 * d__2 * .1 - 1.2;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - n1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    s2 = d__1 * d__1 * .2 + d__2 * d__2 + x[ka] * 2. - .6;
	}
/* Computing 2nd power */
	d__1 = h__;
	fa = s1 + d__1 * d__1 * s2 - b;
/* Computing 2nd power */
	d__1 = (doublereal) (n1 + 1);
	h__ = 1. / (d__1 * d__1);
	if (ka == 1) {
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else if (ka == n1 + 1) {
	    g[1] += h__ * .4 * x[1] * fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka == n1) {
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else if (ka == *n) {
	    g[n1] += h__ * .4 * x[n1] * fa;
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	} else if (ka < n1) {
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else {
	    g[ka - n1] += h__ * .4 * x[ka - n1] * fa;
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	    g[ka + 1] -= fa;
	}
/* L301: */
    }
    return 0;
L710:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	la = 1;
	h__ = 1. / (doublereal) (nd + 1);
	h2 = la * h__ * h__;
	if (l == 1 && k == 1) {
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * exp(x[1]);
	}
	if (1 < l && l < nd && k == 1) {
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * exp(x[l]);
	}
	if (l == nd && k == 1) {
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * exp(x[nd]);
	}
	if (l == 1 && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == 1 && k == nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * exp(x[
		    ka]);
	}
	if (l == nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * exp(x[ka]);
	}
	if (l == 1 && k == 1) {
	    g[1] += (h2 * exp(x[1]) + 4.) * fa;
	    g[2] -= fa;
	    g[nd + 1] -= fa;
	}
	if (1 < l && l < nd && k == 1) {
	    g[l] += (h2 * exp(x[l]) + 4.) * fa;
	    g[l - 1] -= fa;
	    g[l + 1] -= fa;
	    g[l + nd] -= fa;
	}
	if (l == nd && k == 1) {
	    g[nd] += (h2 * exp(x[nd]) + 4.) * fa;
	    g[nd - 1] -= fa;
	    g[nd + nd] -= fa;
	}
	if (l == 1 && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == nd && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == 1 && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	}
	if (1 < l && l < nd && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
	if (l == nd && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
/* L711: */
    }
    return 0;
L820:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	h__ = 1. / (doublereal) (nd + 1);
	h2 = h__ * h__;
	if (l == 1 && k == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
/* Computing 2nd power */
	    d__2 = h__ + 1.;
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * (d__1 * d__1) - 24. / (
		    d__2 * d__2);
	}
	if (1 < l && l < nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[l];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 1.;
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * (d__1 * 
		    d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[nd];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (l == 1 && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 1.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == 1 && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (1 < l && l < nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * (d__1 
		    * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    24. / (d__2 * d__2);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * (d__1 * d__1);
	}
	if (l == 1 && k == 1) {
	    g[1] += (h2 * x[1] * 2. + 4.) * fa;
	    g[2] -= fa;
	    g[nd + 1] -= fa;
	}
	if (1 < l && l < nd && k == 1) {
	    g[l] += (h2 * x[l] * 2. + 4.) * fa;
	    g[l - 1] -= fa;
	    g[l + 1] -= fa;
	    g[l + nd] -= fa;
	}
	if (l == nd && k == 1) {
	    g[nd] += (h2 * x[nd] * 2. + 4.) * fa;
	    g[nd - 1] -= fa;
	    g[nd + nd] -= fa;
	}
	if (l == 1 && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == nd && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == 1 && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	}
	if (1 < l && l < nd && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
	if (l == nd && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
/* L821: */
    }
    return 0;
} /* tfgu28_ */

/* SUBROUTINE TFBU28                ALL SYSTEMS                92/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 92/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  VALUES AND GRADIENTS OF MODEL FUNCTIONS FOR UNCONSTRAINED */
/*  MINIMIZATION. UNIVERSAL VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  F  VALUE OF THE MODEL FUNCTION. */
/*  RI  G(N)  GRADIENG OF THE MODEL FUNCTION. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tfbu28_(integer *n, doublereal *x, doublereal *f, 
	doublereal *g, integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */
    double exp(doublereal), cos(doublereal), sin(doublereal), pow_dd(
	    doublereal *, doublereal *), d_sign(doublereal *, doublereal *), 
	    log(doublereal), atan(doublereal), sqrt(doublereal), pow_di(
	    doublereal *, integer *), sinh(doublereal), cosh(doublereal);

    /* Local variables */
    static doublereal a, b, c__, d__, e, h__;
    static integer i__, j, k, l;
    static doublereal p, q, r__, s, t, u, v, w, a1, a2, a3, a4, h2;
    static integer i1, i2, j1, j2, n1;
    static doublereal s1, s2, s3, t1, ca, cb, fa, be, ga;
    static integer ia, ib;
    static doublereal ff, al, fg;
    static integer ic, ka, la, nd;
    static doublereal pi, ex, sx[1000], be1, ga1[2], ga2[2], ga3[6], ga4[6], 
	    be2, al1, al2, d1s, d2s, alf, gam, bet, alfa;
    extern /* Subroutine */ int uxvset_(integer *, doublereal *, doublereal *)
	    ;

    /* Parameter adjustments */
    --g;
    --x;

    /* Function Body */
    pi = 3.14159265358979323846;
    *f = 0.;
    uxvset_(n, &c_b413, &g[1]);
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
	case 11:  goto L110;
	case 12:  goto L120;
	case 13:  goto L130;
	case 14:  goto L140;
	case 15:  goto L150;
	case 16:  goto L160;
	case 17:  goto L170;
	case 18:  goto L180;
	case 19:  goto L190;
	case 20:  goto L200;
	case 21:  goto L210;
	case 22:  goto L220;
	case 23:  goto L230;
	case 24:  goto L250;
	case 25:  goto L310;
	case 26:  goto L320;
	case 27:  goto L330;
	case 28:  goto L350;
	case 29:  goto L370;
	case 30:  goto L390;
	case 31:  goto L400;
	case 32:  goto L450;
	case 33:  goto L460;
	case 34:  goto L470;
	case 35:  goto L480;
	case 36:  goto L490;
	case 37:  goto L500;
	case 38:  goto L510;
	case 39:  goto L520;
	case 40:  goto L530;
	case 41:  goto L540;
	case 42:  goto L550;
	case 43:  goto L560;
	case 44:  goto L570;
	case 45:  goto L580;
	case 46:  goto L590;
	case 47:  goto L600;
	case 48:  goto L610;
	case 49:  goto L620;
	case 50:  goto L630;
	case 51:  goto L720;
	case 52:  goto L740;
	case 53:  goto L750;
	case 54:  goto L760;
	case 55:  goto L780;
	case 56:  goto L790;
	case 57:  goto L810;
	case 58:  goto L830;
	case 59:  goto L840;
	case 60:  goto L860;
	case 61:  goto L870;
	case 62:  goto L880;
	case 63:  goto L900;
	case 64:  goto L910;
	case 65:  goto L920;
	case 66:  goto L930;
	case 67:  goto L940;
	case 68:  goto L950;
	case 69:  goto L960;
	case 70:  goto L970;
	case 71:  goto L980;
	case 72:  goto L990;
	case 73:  goto L800;
	case 74:  goto L240;
	case 75:  goto L410;
	case 76:  goto L420;
	case 77:  goto L650;
	case 78:  goto L660;
	case 79:  goto L670;
	case 80:  goto L680;
	case 81:  goto L690;
	case 82:  goto L340;
	case 83:  goto L360;
	case 84:  goto L380;
	case 85:  goto L430;
	case 86:  goto L440;
	case 87:  goto L270;
	case 88:  goto L280;
	case 89:  goto L290;
	case 90:  goto L300;
	case 91:  goto L710;
	case 92:  goto L820;
    }
L10:
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
	*f = *f + d__1 * d__1 * 100. + d__2 * d__2;
	g[j - 1] = g[j - 1] + x[j - 1] * 400. * a + b * 2.;
	g[j] -= a * 200.;
/* L11: */
    }
    return 0;
L20:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j - 1];
	a = d__1 * d__1 - x[j];
	b = x[j - 1] - 1.;
/* Computing 2nd power */
	d__1 = x[j + 1];
	c__ = d__1 * d__1 - x[j + 2];
	d__ = x[j + 1] - 1.;
	u = x[j] + x[j + 2] - 2.;
	v = x[j] - x[j + 2];
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
/* Computing 2nd power */
	d__3 = c__;
/* Computing 2nd power */
	d__4 = d__;
/* Computing 2nd power */
	d__5 = u;
/* Computing 2nd power */
	d__6 = v;
	*f = *f + d__1 * d__1 * 100. + d__2 * d__2 + d__3 * d__3 * 90. + d__4 
		* d__4 + d__5 * d__5 * 10. + d__6 * d__6 * .1;
	g[j - 1] = g[j - 1] + x[j - 1] * 400. * a + b * 2.;
	g[j] = g[j] - a * 200. + u * 20. + v * .2;
	g[j + 1] = g[j + 1] + x[j + 1] * 360. * c__ + d__ * 2.;
	g[j + 2] = g[j + 2] - c__ * 180. + u * 20. - v * .2;
/* L21: */
    }
    return 0;
L30:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] + x[j] * 10.;
	b = x[j + 1] - x[j + 2];
	c__ = x[j] - x[j + 1] * 2.;
	d__ = x[j - 1] - x[j + 2];
/* Computing 2nd power */
	d__1 = a;
/* Computing 2nd power */
	d__2 = b;
/* Computing 4th power */
	d__3 = c__, d__3 *= d__3;
/* Computing 4th power */
	d__4 = d__, d__4 *= d__4;
	*f = *f + d__1 * d__1 + d__2 * d__2 * 5. + d__3 * d__3 + d__4 * d__4 *
		 10.;
/* Computing 3rd power */
	d__1 = d__;
	g[j - 1] = g[j - 1] + a * 2. + d__1 * (d__1 * d__1) * 40.;
/* Computing 3rd power */
	d__1 = c__;
	g[j] = g[j] + a * 20. + d__1 * (d__1 * d__1) * 4.;
/* Computing 3rd power */
	d__1 = c__;
	g[j + 1] = g[j + 1] - d__1 * (d__1 * d__1) * 8. + b * 10.;
/* Computing 3rd power */
	d__1 = d__;
	g[j + 2] = g[j + 2] - d__1 * (d__1 * d__1) * 40. - b * 10.;
/* L31: */
    }
    return 0;
L40:
    i__1 = *n - 2;
    for (j = 2; j <= i__1; j += 2) {
	a = exp(x[j - 1]);
	b = a - x[j];
	d__ = x[j] - x[j + 1];
	p = x[j + 1] - x[j + 2];
	c__ = cos(p);
	q = sin(p) / cos(p);
	u = x[j - 1];
	v = x[j + 2] - 1.;
/* Computing 4th power */
	d__1 = b, d__1 *= d__1;
/* Computing 6th power */
	d__2 = d__, d__2 *= d__2;
/* Computing 4th power */
	d__3 = q, d__3 *= d__3;
/* Computing 8th power */
	d__4 = u, d__4 *= d__4, d__4 *= d__4;
/* Computing 2nd power */
	d__5 = v;
	*f = *f + d__1 * d__1 + d__2 * (d__2 * d__2) * 100. + d__3 * d__3 + 
		d__4 * d__4 + d__5 * d__5;
/* Computing 3rd power */
	d__1 = b;
	b = d__1 * (d__1 * d__1) * 4.;
/* Computing 5th power */
	d__1 = d__, d__2 = d__1, d__1 *= d__1;
	d__ = d__2 * (d__1 * d__1) * 600.;
/* Computing 3rd power */
	d__1 = q;
/* Computing 2nd power */
	d__2 = c__;
	q = d__1 * (d__1 * d__1) * 4. / (d__2 * d__2);
/* Computing 7th power */
	d__1 = u, d__2 = d__1, d__1 *= d__1, d__2 *= d__1;
	g[j - 1] = g[j - 1] + a * b + d__2 * (d__1 * d__1) * 8.;
	g[j] = g[j] + d__ - b;
	g[j + 1] = g[j + 1] + q - d__;
	g[j + 2] = g[j + 2] + v * 2. - q;
/* L41: */
    }
    return 0;
L50:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	g[j] += b * (3. - x[j] * 4.);
	if (j > 1) {
	    g[j - 1] -= b;
	}
	if (j < *n) {
	    g[j + 1] -= b;
	}
/* L51: */
    }
    return 0;
L60:
    p = 2.3333333333333335;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = (d__1 * d__1 * 5. + 2.) * x[j] + 1.;
/* Computing MAX */
	i__2 = 1, i__3 = j - 5;
/* Computing MIN */
	i__5 = *n, i__6 = j + 1;
	i__4 = min(i__5,i__6);
	for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
	    if (i__ != j) {
		a += x[i__] * (x[i__] + 1.);
	    }
/* L61: */
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
/* Computing 2nd power */
	d__1 = x[j];
	g[j] += b * (d__1 * d__1 * 15. + 2.);
/* Computing MAX */
	i__4 = 1, i__2 = j - 5;
/* Computing MIN */
	i__5 = *n, i__6 = j + 1;
	i__3 = min(i__5,i__6);
	for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
	    if (i__ != j) {
		g[i__] += b * (x[i__] * 2. + 1.);
	    }
/* L62: */
	}
/* L63: */
    }
    return 0;
L70:
    p = 2.3333333333333335;
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (3. - x[j] * 2.) * x[j] + 1.;
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
	d__1 = abs(a);
	*f += pow_dd(&d__1, &p);
	d__1 = abs(a);
	d__2 = p - 1.;
	b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	g[j] += b * (3. - x[j] * 4.);
	if (j > 1) {
	    g[j - 1] -= b;
	}
	if (j < *n) {
	    g[j + 1] -= b;
	}
	if (j <= k) {
	    a = x[j] + x[j + k];
	    d__1 = abs(a);
	    *f += pow_dd(&d__1, &p);
	    d__1 = abs(a);
	    d__2 = p - 1.;
	    b = p * pow_dd(&d__1, &d__2) * d_sign(&c_b347, &a);
	    g[j] += b;
	    g[j + k] += b;
	}
/* L71: */
    }
    return 0;
L80:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L81;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
L81:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	}
/* Computing 2nd power */
	d__1 = (doublereal) (*n + j) - p;
	*f += d__1 * d__1 / (doublereal) (*n);
	p = ((doublereal) (*n + j) - p) * 2. / (doublereal) (*n);
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L82;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
L82:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    g[i__] -= p * (a * cos(x[i__]) - b * sin(x[i__]));
	}
/* L83: */
    }
    return 0;
L90:
    k = *n / 2;
    q = 1. / (doublereal) (*n);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L91;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
L91:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) (i__ + j) / 10.;
	    p = p + a * sin(x[i__]) + b * cos(x[i__]);
	    g[i__] += q * (a * cos(x[i__]) - b * sin(x[i__]));
	}
	*f += (p + (doublereal) j * (1. - cos(x[j]))) * q;
	g[j] += q * (doublereal) j * sin(x[j]);
/* L92: */
    }
    return 0;
L100:
    k = *n / 2;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	p = 0.;
	q = (doublereal) j / 10. + 1.;
	i__3 = j + 2;
	for (i__ = j - 2; i__ <= i__3; ++i__) {
	    if (i__ < 1 || i__ > *n) {
		goto L101;
	    }
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
L101:
	    ;
	}
	if (j > k) {
	    i__ = j - k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
	} else {
	    i__ = j + k;
	    a = (i__ % 5 + 1. + j % 5) * 5.;
	    b = (doublereal) i__ / 10. + 1.;
	    c__ = (doublereal) (i__ + j) / 10.;
	    p += a * sin(q * x[j] + b * x[i__] + c__);
	    r__ = a * cos(q * x[j] + b * x[i__] + c__) / (doublereal) (*n);
	    g[j] += r__ * q;
	    g[i__] += r__ * b;
	}
	*f += p;
/* L102: */
    }
    *f /= (doublereal) (*n);
    return 0;
L110:
    p = -.002008;
    q = -.0019;
    r__ = -2.61e-4;
    i__1 = *n - 5;
    for (i__ = 0; i__ <= i__1; i__ += 5) {
	a = 1.;
	b = 0.;
	for (j = 1; j <= 5; ++j) {
	    a *= x[i__ + j];
/* Computing 2nd power */
	    d__1 = x[i__ + j];
	    b += d__1 * d__1;
/* L111: */
	}
	w = exp(a);
	a *= w;
	b = b - 10. - p;
	c__ = x[i__ + 2] * x[i__ + 3] - x[i__ + 4] * 5. * x[i__ + 5] - q;
/* Computing 3rd power */
	d__1 = x[i__ + 1];
/* Computing 3rd power */
	d__2 = x[i__ + 2];
	d__ = d__1 * (d__1 * d__1) + d__2 * (d__2 * d__2) + 1. - r__;
/* Computing 2nd power */
	d__1 = b;
/* Computing 2nd power */
	d__2 = c__;
/* Computing 2nd power */
	d__3 = d__;
	*f = *f + w + (d__1 * d__1 + d__2 * d__2 + d__3 * d__3) * 10.;
/* Computing 2nd power */
	d__1 = x[i__ + 1];
	g[i__ + 1] = g[i__ + 1] + a / x[i__ + 1] + (b * 2. * x[i__ + 1] + d__ 
		* 3. * (d__1 * d__1)) * 20.;
/* Computing 2nd power */
	d__1 = x[i__ + 2];
	g[i__ + 2] = g[i__ + 2] + a / x[i__ + 2] + (b * 2. * x[i__ + 2] + c__ 
		* x[i__ + 3] + d__ * 3. * (d__1 * d__1)) * 20.;
	g[i__ + 3] = g[i__ + 3] + a / x[i__ + 3] + (b * 2. * x[i__ + 3] + c__ 
		* x[i__ + 2]) * 20.;
	g[i__ + 4] = g[i__ + 4] + a / x[i__ + 4] + (b * 2. * x[i__ + 4] - c__ 
		* 5. * x[i__ + 5]) * 20.;
	g[i__ + 5] = g[i__ + 5] + a / x[i__ + 5] + (b * 2. * x[i__ + 5] - c__ 
		* 5. * x[i__ + 4]) * 20.;
/* L112: */
    }
    return 0;
L120:
    c__ = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
	a = x[j - 1] - 3.;
	b = x[j - 1] - x[j];
	c__ += a;
/* Computing 2nd power */
	d__1 = a;
	*f = *f + d__1 * d__1 * 1e-4 - b + exp(b * 20.);
	g[j - 1] = g[j - 1] + a * 2e-4 - 1. + exp(b * 20.) * 20.;
	g[j] = g[j] + 1. - exp(b * 20.) * 20.;
/* L121: */
    }
/* Computing 2nd power */
    d__1 = c__;
    *f += d__1 * d__1;
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
	g[j - 1] += c__ * 2.;
/* L122: */
    }
    return 0;
L130:
    i__1 = *n;
    for (j = 2; j <= i__1; j += 2) {
/* Computing 2nd power */
	d__1 = x[j];
	a = d__1 * d__1;
	if (a == 0.) {
	    a = 1e-60;
	}
/* Computing 2nd power */
	d__1 = x[j - 1];
	b = d__1 * d__1;
	if (b == 0.) {
	    b = 1e-60;
	}
	c__ = a + 1.;
	d__ = b + 1.;
	*f = *f + pow_dd(&b, &c__) + pow_dd(&a, &d__);
	p = 0.;
	if (a > p) {
	    p = log(a);
	}
	q = 0.;
	if (b > q) {
	    q = log(b);
	}
	g[j - 1] += x[j - 1] * 2. * (c__ * pow_dd(&b, &a) + p * pow_dd(&a, &
		d__));
	g[j] += x[j] * 2. * (d__ * pow_dd(&a, &b) + q * pow_dd(&b, &c__));
/* L131: */
    }
    return 0;
L140:
    p = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
    d__1 = p;
    q = d__1 * d__1 * .5;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 3rd power */
	d__1 = x[j] + (doublereal) j * p + 1.;
	a = x[j] * 2. + q * (d__1 * (d__1 * d__1));
	if (j > 1) {
	    a -= x[j - 1];
	}
	if (j < *n) {
	    a -= x[j + 1];
	}
/* Computing 2nd power */
	d__1 = a;
	*f += d__1 * d__1;
	d__1 = x[j] + (doublereal) j * p + 1.;
	g[j] += a * (q * 6. * pow_dd(&d__1, &c_b532) + 4.);
	if (j > 1) {
	    g[j - 1] -= a * 2.;
	}
	if (j < *n) {
	    g[j + 1] -= a * 2.;
	}
/* L141: */
    }
    return 0;
L150:
    p = 1. / (doublereal) (*n + 1);
    q = 2. / p;
    r__ = p * 2.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	a = x[j - 1] - x[j];
	*f += q * x[j - 1] * a;
	g[j - 1] += q * (x[j - 1] * 2. - x[j]);
	g[j] -= q * x[j - 1];
	if (abs(a) <= 1e-6) {
	    *f += r__ * exp(x[j]) * (a / 2. * (a / 3. * (a / 4. + 1.) + 1.) + 
		    1.);
	    g[j - 1] += r__ * exp(x[j]) * (a * (a / 8. + .33333333333333331) 
		    + .5);
	    g[j] += r__ * exp(x[j]) * (a * (a / 24. + .16666666666666666) + 
		    .5);
	} else {
	    b = exp(x[j - 1]) - exp(x[j]);
	    *f += r__ * b / a;
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] += r__ * (exp(x[j - 1]) * a - b) / (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = a;
	    g[j] -= r__ * (exp(x[j]) * a - b) / (d__1 * d__1);
	}
/* L151: */
    }
/* Computing 2nd power */
    d__1 = x[*n];
    *f = *f + q * (d__1 * d__1) + r__ * (exp(x[1]) - 1.) / x[1] + r__ * (exp(
	    x[*n]) - 1.) / x[*n];
/* Computing 2nd power */
    d__1 = x[1];
    g[1] += r__ * (exp(x[1]) * (x[1] - 1.) + 1.) / (d__1 * d__1);
/* Computing 2nd power */
    d__1 = x[*n];
    g[*n] = g[*n] + q * 2. * x[*n] + r__ * (exp(x[*n]) * (x[*n] - 1.) + 1.) / 
	    (d__1 * d__1);
    return 0;
L160:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	a = (doublereal) j * (1. - cos(x[j]));
	if (j > 1) {
	    a += (doublereal) j * sin(x[j - 1]);
	}
	if (j < *n) {
	    a -= (doublereal) j * sin(x[j + 1]);
	}
	*f += a;
	a = (doublereal) j * sin(x[j]);
	g[j] += a;
	if (j > 1) {
	    g[j - 1] += (doublereal) j * cos(x[j - 1]);
	}
	if (j < *n) {
	    g[j + 1] -= (doublereal) j * cos(x[j + 1]);
	}
/* L161: */
    }
    return 0;
L170:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
	    *f = *f + d__1 * d__1 * .25 / p + d__2 * d__2 * .125 / p + p * (
		    exp(x[j]) - 1.);
	    g[j] = g[j] + x[j] * .5 / p + p * exp(x[j]);
	    g[j + 1] += x[j + 1] * .25 / p;
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
	    *f = *f + d__1 * d__1 * .25 / p + d__2 * d__2 * .125 / p + p * (
		    exp(x[j]) - 1.);
	    g[j] = g[j] + x[j] * .5 / p + p * exp(x[j]);
	    g[j - 1] += x[j - 1] * .25 / p;
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
	    *f = *f + d__1 * d__1 * .125 / p + p * (exp(x[j]) - 1.);
	    a = (x[j + 1] - x[j - 1]) * .25 / p;
	    g[j] += p * exp(x[j]);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L171: */
    }
    return 0;
L180:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = (doublereal) j * p;
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
/* Computing 2nd power */
	    d__3 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * .25 / p - p * (
		    d__3 * d__3 + x[j] * 2. * q);
	    g[j] = g[j] + x[j] / p - p * 2. * (x[j] + q);
	    g[j + 1] += x[j + 1] * .5 / p;
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
/* Computing 2nd power */
	    d__3 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * .25 / p - p * (
		    d__3 * d__3 + x[j] * 2. * q);
	    g[j] = g[j] + x[j] / p - p * 2. * (x[j] + q);
	    g[j - 1] += x[j - 1] * .5 / p;
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
/* Computing 2nd power */
	    d__2 = x[j];
	    *f = *f + d__1 * d__1 * .25 / p - p * (d__2 * d__2 + x[j] * 2. * 
		    q);
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    g[j] -= p * 2. * (x[j] + q);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L181: */
    }
    return 0;
L190:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	q = exp((doublereal) j * 2. * p);
	if (j == 1) {
	    r__ = .33333333333333331;
/* Computing 2nd power */
	    d__1 = x[j] - r__;
/* Computing 2nd power */
	    d__2 = r__;
/* Computing 2nd power */
	    d__3 = x[j + 1] - r__;
/* Computing 2nd power */
	    d__4 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * 7. + d__3 * d__3 * 
		    .25 / p + p * (d__4 * d__4 + x[j] * 2. * q);
	    a = (x[j + 1] - r__) * .5 / p;
	    g[j] = g[j] + p * 2. * (x[j] + q) + (x[j] - r__) / p;
	    g[j + 1] += a;
	} else if (j == *n) {
	    r__ = exp(2.) / 3.;
/* Computing 2nd power */
	    d__1 = x[j] - r__;
/* Computing 2nd power */
	    d__2 = r__;
/* Computing 2nd power */
	    d__3 = x[j - 1] - r__;
/* Computing 2nd power */
	    d__4 = x[j];
	    *f = *f + d__1 * d__1 * .5 / p + d__2 * d__2 * 7. + d__3 * d__3 * 
		    .25 / p + p * (d__4 * d__4 + x[j] * 2. * q);
	    a = (x[j - 1] - r__) * .5 / p;
	    g[j] = g[j] + p * 2. * (x[j] + q) + (x[j] - r__) / p;
	    g[j - 1] += a;
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
/* Computing 2nd power */
	    d__2 = x[j];
	    *f = *f + d__1 * d__1 * .25 / p + p * (d__2 * d__2 + x[j] * 2. * 
		    q);
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    g[j] += p * 2. * (x[j] + q);
	    g[j - 1] -= a;
	    g[j + 1] += a;
	}
/* L191: */
    }
    return 0;
L200:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a = exp(d__1 * d__1 * -2.);
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j + 1];
	    *f = *f + (d__1 * d__1 * .5 / p - p) + (d__2 * d__2 * .25 / p - p)
		     * a;
	    b = x[j + 1] * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] = g[j] + x[j] / p - x[j] * 4. * a * p * (d__1 * d__1 - 1.);
	    g[j + 1] += a * b;
	} else if (j == *n) {
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = x[j - 1];
	    *f = *f + (d__1 * d__1 * .5 / p - p) * exp(-2.) + (d__2 * d__2 * 
		    .25 / p - p) * a;
	    b = x[j - 1] * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] = g[j] + x[j] / p * exp(-2.) - x[j] * 4. * a * p * (d__1 * 
		    d__1 - 1.);
	    g[j - 1] += a * b;
	} else {
/* Computing 2nd power */
	    d__1 = x[j + 1] - x[j - 1];
	    *f += (d__1 * d__1 * .25 / p - p) * a;
	    b = (x[j + 1] - x[j - 1]) * .5 / p;
/* Computing 2nd power */
	    d__1 = b;
	    g[j] -= x[j] * 4. * a * p * (d__1 * d__1 - 1.);
	    g[j - 1] -= a * b;
	    g[j + 1] += a * b;
	}
/* L201: */
    }
    return 0;
L210:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = (x[j + 1] - 1.) * .5 / p;
	    b = (x[j] - 1.) / p;
	    u = atan(a);
	    v = atan(b);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__3 = b;
	    *f = *f + p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.))) 
		    + p * .5 * (b * v + 1. - log(sqrt(d__3 * d__3 + 1.)));
	    g[j] = g[j] + p * 2. * x[j] + v * .5;
	    g[j + 1] += u * .5;
	} else if (j == *n) {
	    a = (2. - x[j - 1]) * .5 / p;
	    b = (2. - x[j]) / p;
	    u = atan(a);
	    v = atan(b);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__3 = b;
	    *f = *f + p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.))) 
		    + p * .5 * (b * v + 4. - log(sqrt(d__3 * d__3 + 1.)));
	    g[j] = g[j] + p * 2. * x[j] - v * .5;
	    g[j - 1] -= u * .5;
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
	    u = atan(a);
/* Computing 2nd power */
	    d__1 = x[j];
/* Computing 2nd power */
	    d__2 = a;
	    *f += p * (d__1 * d__1 + a * u - log(sqrt(d__2 * d__2 + 1.)));
	    g[j] += p * 2. * x[j];
	    g[j - 1] -= u * .5;
	    g[j + 1] += u * .5;
	}
/* L211: */
    }
    return 0;
L220:
    p = 1. / (doublereal) (*n + 1);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    a = x[j + 1] * .5 / p;
	    b = x[j] / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
/* Computing 4th power */
	    d__4 = b, d__4 *= d__4;
/* Computing 2nd power */
	    d__5 = 1. - b;
	    *f = *f + p * (d__1 * d__1 * 100. + d__3 * d__3) + p * .5 * (d__4 
		    * d__4 * 100. + d__5 * d__5);
/* Computing 2nd power */
	    d__1 = a;
/* Computing 3rd power */
	    d__2 = b;
	    g[j] = g[j] + p * 200. * (x[j] - d__1 * d__1) + d__2 * (d__2 * 
		    d__2) * 200. - (1. - b);
/* Computing 2nd power */
	    d__1 = a;
	    g[j + 1] = g[j + 1] - (x[j] - d__1 * d__1) * 200. * a - (1. - a);
	} else if (j == *n) {
	    a = x[j - 1] * -.5 / p;
	    b = -x[j] / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
/* Computing 4th power */
	    d__4 = b, d__4 *= d__4;
/* Computing 2nd power */
	    d__5 = 1. - b;
	    *f = *f + p * (d__1 * d__1 * 100. + d__3 * d__3) + p * .5 * (d__4 
		    * d__4 * 100. + d__5 * d__5);
/* Computing 2nd power */
	    d__1 = a;
/* Computing 3rd power */
	    d__2 = b;
	    g[j] = g[j] + p * 200. * (x[j] - d__1 * d__1) - d__2 * (d__2 * 
		    d__2) * 200. + (1. - b);
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] = g[j - 1] + (x[j] - d__1 * d__1) * 200. * a + (1. - a);
	} else {
	    a = (x[j + 1] - x[j - 1]) * .5 / p;
/* Computing 2nd power */
	    d__2 = a;
/* Computing 2nd power */
	    d__1 = x[j] - d__2 * d__2;
/* Computing 2nd power */
	    d__3 = 1. - a;
	    *f += p * (d__1 * d__1 * 100. + d__3 * d__3);
/* Computing 2nd power */
	    d__1 = a;
	    g[j] += p * 200. * (x[j] - d__1 * d__1);
/* Computing 2nd power */
	    d__1 = a;
	    g[j - 1] = g[j - 1] + (x[j] - d__1 * d__1) * 200. * a + (1. - a);
/* Computing 2nd power */
	    d__1 = a;
	    g[j + 1] = g[j + 1] - (x[j] - d__1 * d__1) * 200. * a - (1. - a);
	}
/* L221: */
    }
    return 0;
L230:
    a1 = -1.;
    a2 = 0.;
    a3 = 0.;
    d1s = exp(.01);
    d2s = 1.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = x[j];
	a1 += (doublereal) (*n - j + 1) * (d__1 * d__1);
/* L231: */
    }
    a = a1 * 4.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	g[j] = a * (doublereal) (*n - j + 1) * x[j];
	s1 = exp(x[j] / 100.);
	if (j > 1) {
	    s3 = s1 + s2 - d2s * (d1s - 1.);
/* Computing 2nd power */
	    d__1 = s3;
	    a2 += d__1 * d__1;
/* Computing 2nd power */
	    d__1 = s1 - 1. / d1s;
	    a3 += d__1 * d__1;
	    g[j] += s1 * 1e-5 * (s3 + s1 - 1. / d1s) / 50.;
	    g[j - 1] += s2 * 1e-5 * s3 / 50.;
	}
	s2 = s1;
	d2s = d1s * d2s;
/* L232: */
    }
/* Computing 2nd power */
    d__1 = a1;
/* Computing 2nd power */
    d__2 = x[1] - .2;
    *f = (a2 + a3) * 1e-5 + d__1 * d__1 + d__2 * d__2;
    g[1] += (x[1] - .2) * 2.;
    return 0;
L250:
    a = 1.;
    b = 0.;
    c__ = 0.;
    d__ = 0.;
    *f = 0.;
    u = exp(x[*n]);
    v = exp(x[*n - 1]);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *n / 2) {
/* Computing 2nd power */
	    d__1 = x[j] - 1.;
	    *f += d__1 * d__1;
	}
	if (j <= *n - 2) {
/* Computing 2nd power */
	    d__1 = x[j] + x[j + 1] * 2. + x[j + 2] * 10. - 1.;
	    b += d__1 * d__1;
/* Computing 2nd power */
	    d__1 = x[j] * 2. + x[j + 1] - 3.;
	    c__ += d__1 * d__1;
	}
/* Computing 2nd power */
	d__1 = x[j];
	d__ = d__ + d__1 * d__1 - (doublereal) (*n);
/* L251: */
    }
/* Computing 2nd power */
    d__1 = d__;
    *f = *f + a * (u * b + 1. + b * c__ + v * c__) + d__1 * d__1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *n / 2) {
	    g[j] += (x[j] - 1.) * 2.;
	}
	if (j <= *n - 2) {
	    p = a * (u + c__) * (x[j] + x[j + 1] * 2. + x[j + 2] * 10. - 1.);
	    q = a * (v + b) * (x[j] * 2. + x[j + 1] - 3.);
	    g[j] = g[j] + p * 2. + q * 4.;
	    g[j + 1] = g[j + 1] + p * 4. + q * 2.;
	    g[j + 2] += p * 20.;
	}
	g[j] += d__ * 4. * x[j];
/* L252: */
    }
    g[*n - 1] += a * v * c__;
    g[*n] += a * u * b;
    return 0;
L310:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = (x[ka + 1] - d__1 * d__1) * 10.;
	    g[ka] -= x[ka] * 20. * fa;
	    g[ka + 1] += fa * 10.;
	} else {
	    fa = 1. - x[ka - 1];
	    g[ka - 1] -= fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L311: */
    }
    *f *= .5;
    return 0;
L320:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    fa = x[ka] + x[ka + 1] * 10.;
	    g[ka] += fa;
	    g[ka + 1] += fa * 10.;
	} else if (ka % 4 == 2) {
	    fa = (x[ka + 1] - x[ka + 2]) * 2.23606797749979;
	    g[ka + 1] += fa * 2.23606797749979;
	    g[ka + 2] -= fa * 2.23606797749979;
	} else if (ka % 4 == 3) {
	    a = x[ka - 1] - x[ka] * 2.;
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	    g[ka - 1] += a * 2. * fa;
	    g[ka] -= a * 4. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 3] - x[ka];
	    fa = d__1 * d__1 * 3.16227766016838;
	    a = (x[ka - 3] - x[ka]) * 2.;
	    g[ka - 3] += a * 3.16227766016838 * fa;
	    g[ka] -= a * 3.16227766016838 * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L321: */
    }
    *f *= .5;
    return 0;
L330:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = (x[ka] - 1.) / 316.22776601683825;
	    g[ka] += fa * .0031622776601683764;
	} else {
	    fa = -.25;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
/* Computing 2nd power */
		d__1 = x[j];
		fa += d__1 * d__1;
/* L331: */
	    }
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		g[j] += x[j] * 2. * fa;
/* L332: */
	    }
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L333: */
    }
    *f *= .5;
    return 0;
L350:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka <= *n) {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	} else {
	    fa = 0.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		fa += (doublereal) j * (x[j] - 1.);
/* L351: */
	    }
	    if (ka == *n + 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    g[j] += (doublereal) j * fa;
/* L352: */
		}
	    } else if (ka == *n + 2) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
/* Computing 3rd power */
		    d__1 = fa;
		    g[j] += (doublereal) j * 2. * (d__1 * (d__1 * d__1));
/* L353: */
		}
/* Computing 2nd power */
		d__1 = fa;
		fa = d__1 * d__1;
	    }
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L354: */
    }
    *f *= .5;
    return 0;
L370:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < *n) {
	    a = 0.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		a += x[j];
/* L371: */
	    }
	    fa = x[ka] + a - (doublereal) (*n + 1);
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		g[j] += fa;
/* L372: */
	    }
	    g[ka] += fa;
	} else {
	    a = 1.;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		a *= x[j];
/* L373: */
	    }
	    fa = a - 1.;
	    i__ = 0;
	    i__3 = *n;
	    for (j = 1; j <= i__3; ++j) {
		b = x[j];
		if (b == 0. && i__ == 0) {
		    i__ = j;
		}
		if (i__ != j) {
		    a *= b;
		}
/* L374: */
	    }
	    if (i__ == 0) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    g[j] += a / x[j] * fa;
/* L375: */
		}
	    } else {
		g[i__] += a * fa;
	    }
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L376: */
    }
    *f *= .5;
    return 0;
L390:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
	a = 0.;
	b = 0.;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    w = (doublereal) j * u;
	    if (j <= ka) {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		a += w * (d__1 * (d__1 * d__1));
	    } else {
/* Computing 3rd power */
		d__1 = x[j] + w + 1.;
		b += (1. - w) * (d__1 * (d__1 * d__1));
	    }
/* L391: */
	}
	fa = x[ka] + u * ((1. - v) * a + v * b) / 2.;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	i__3 = *n;
	for (j = 1; j <= i__3; ++j) {
	    w = (doublereal) j * u;
/* Computing 2nd power */
	    d__1 = x[j] + w + 1.;
	    a = d__1 * d__1;
	    if (j <= ka) {
		g[j] += u * 1.5 * (1. - v) * w * a * fa;
	    } else {
		g[j] += u * 1.5 * (1. - w) * v * a * fa;
	    }
/* L392: */
	}
	g[ka] += fa;
/* L393: */
    }
    *f *= .5;
    return 0;
L400:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = (3. - x[ka] * 2.) * x[ka] + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 2.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += (3. - x[ka] * 4.) * fa;
	if (ka > 1) {
	    g[ka - 1] -= fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= fa * 2.;
	}
/* L401: */
    }
    *f *= .5;
    return 0;
L450:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
	fa = (3. - x[i__] * 2.) * x[i__] + 1.;
	if (i__ > 1) {
	    fa -= x[i__ - 1];
	}
	if (i__ < *n) {
	    fa -= x[i__ + 1];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[i__] += (3. - x[i__] * 4.) * fa;
	if (i__ > 1) {
	    g[i__ - 1] -= fa;
	}
	if (i__ < *n) {
	    g[i__ + 1] -= fa;
	}
/* L451: */
    }
    *f *= .5;
    return 0;
L460:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka;
/* Computing 2nd power */
	d__1 = x[i__];
	fa = (d__1 * d__1 * 5. + 2.) * x[i__] + 1.;
/* Computing MAX */
	i__3 = 1, i__4 = i__ - 5;
/* Computing MIN */
	i__5 = *n, i__6 = i__ + 1;
	i__2 = min(i__5,i__6);
	for (j = max(i__3,i__4); j <= i__2; ++j) {
	    if (i__ != j) {
		fa += x[j] * (x[j] + 1.);
	    }
/* L461: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing MAX */
	i__2 = 1, i__3 = i__ - 5;
/* Computing MIN */
	i__5 = *n, i__6 = i__ + 1;
	i__4 = min(i__5,i__6);
	for (j = max(i__2,i__3); j <= i__4; ++j) {
	    if (i__ != j) {
		g[j] += (x[j] * 2. + 1.) * fa;
	    }
/* L462: */
	}
/* Computing 2nd power */
	d__1 = x[i__];
	g[i__] += (d__1 * d__1 * 15. + 2.) * fa;
/* L463: */
    }
    *f *= .5;
    return 0;
L470:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
	    fa = x[i__] + x[i__ + 1] * ((5. - x[i__ + 1]) * x[i__ + 1] - 2.) 
		    - 13.;
	    g[i__] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    g[i__ + 1] += (x[i__ + 1] * 10. - d__1 * d__1 * 3. - 2.) * fa;
	} else {
	    fa = x[i__] + x[i__ + 1] * ((x[i__ + 1] + 1.) * x[i__ + 1] - 14.) 
		    - 29.;
	    g[i__] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    g[i__ + 1] += (x[i__ + 1] * 2. + d__1 * d__1 * 3. - 14.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L471: */
    }
    *f *= .5;
    return 0;
L480:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka % (*n / 2) + 1;
	j = i__ + *n / 2;
	empr28_1.m = *n * 5;
	if (ka <= empr28_1.m / 2) {
	    ia = 1;
	} else {
	    ia = 2;
	}
	ib = 5 - ka / (empr28_1.m / 4);
	ic = ka % 5 + 1;
	d__1 = pow_di(&x[i__], &ia) - pow_di(&x[j], &ib);
	fa = pow_di(&d__1, &ic);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	a = (doublereal) ia;
	b = (doublereal) ib;
	c__ = (doublereal) ic;
	d__ = pow_di(&x[i__], &ia) - pow_di(&x[j], &ib);
	if (d__ != 0.) {
	    i__4 = ic - 1;
	    e = c__ * pow_di(&d__, &i__4);
	    if (x[i__] == 0. && ia <= 1) {
	    } else {
		i__4 = ia - 1;
		g[i__] += e * a * pow_di(&x[i__], &i__4) * fa;
	    }
	    if (x[j] == 0. && ib <= 1) {
	    } else {
		i__4 = ib - 1;
		g[j] -= e * b * pow_di(&x[j], &i__4) * fa;
	    }
	}
/* L481: */
    }
    *f *= .5;
    return 0;
L490:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 5) / 6 << 1) - 1;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3];
	    fa = x[i__] + x[i__ + 1] * 3. * (x[i__ + 2] - 1.) + d__1 * d__1 - 
		    1.;
	    g[i__] += fa;
	    g[i__ + 1] += (x[i__ + 2] - 1.) * 3. * fa;
	    g[i__ + 2] += x[i__ + 1] * 3. * fa;
	    g[i__ + 3] += x[i__ + 3] * 2. * fa;
	} else if (ka % 6 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1];
/* Computing 2nd power */
	    d__2 = x[i__ + 2] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2 - x[i__ + 3] - 3.;
	    g[i__] += (x[i__] + x[i__ + 1]) * 2. * fa;
	    g[i__ + 1] += (x[i__] + x[i__ + 1]) * 2. * fa;
	    g[i__ + 2] += (x[i__ + 2] - 1.) * 2. * fa;
	    g[i__ + 3] -= fa;
	} else if (ka % 6 == 3) {
	    fa = x[i__] * x[i__ + 1] - x[i__ + 2] * x[i__ + 3];
	    g[i__] += x[i__ + 1] * fa;
	    g[i__ + 1] += x[i__] * fa;
	    g[i__ + 2] -= x[i__ + 3] * fa;
	    g[i__ + 3] -= x[i__ + 2] * fa;
	} else if (ka % 6 == 4) {
	    fa = x[i__] * 2. * x[i__ + 2] + x[i__ + 1] * x[i__ + 3] - 3.;
	    g[i__] += x[i__ + 2] * 2. * fa;
	    g[i__ + 1] += x[i__ + 3] * fa;
	    g[i__ + 2] += x[i__] * 2. * fa;
	    g[i__ + 3] += x[i__ + 1] * fa;
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3];
/* Computing 2nd power */
	    d__2 = x[i__] - 1.;
	    fa = d__1 * d__1 + d__2 * d__2;
	    g[i__] += ((x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 2. + 
		    (x[i__] - 1.) * 2.) * fa;
	    g[i__ + 1] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	    g[i__ + 2] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	    g[i__ + 3] += (x[i__] + x[i__ + 1] + x[i__ + 2] + x[i__ + 3]) * 
		    2. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = x[i__] * x[i__ + 1] * x[i__ + 2] * x[i__ + 3] + d__1 * d__1 
		    - 1.;
	    g[i__] += x[i__ + 1] * x[i__ + 2] * x[i__ + 3] * fa;
	    g[i__ + 1] += x[i__] * x[i__ + 2] * x[i__ + 3] * fa;
	    g[i__ + 2] += x[i__] * x[i__ + 1] * x[i__ + 3] * fa;
	    g[i__ + 3] += (x[i__] * x[i__ + 1] * x[i__ + 2] + (x[i__ + 3] - 
		    1.) * 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L491: */
    }
    *f *= .5;
    return 0;
L500:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	j = ka % 2;
	if (j == 0) {
	    fa = 6. - exp(x[i__] * 2.) - exp(x[i__ + 1] * 2.);
	    g[i__] -= exp(x[i__] * 2.) * 2. * fa;
	    g[i__ + 1] -= exp(x[i__ + 1] * 2.) * 2. * fa;
	} else if (i__ == 1) {
	    fa = 4. - exp(x[i__]) - exp(x[i__ + 1]);
	    g[i__] -= exp(x[i__]) * fa;
	    g[i__ + 1] -= exp(x[i__ + 1]) * fa;
	} else if (i__ == *n) {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.);
	    g[i__ - 1] -= exp(x[i__ - 1] * 3.) * 3. * fa;
	    g[i__] -= exp(x[i__] * 3.) * 3. * fa;
	} else {
	    fa = 8. - exp(x[i__ - 1] * 3.) - exp(x[i__] * 3.) + 4. - exp(x[
		    i__]) - exp(x[i__ + 1]);
	    g[i__ - 1] -= exp(x[i__ - 1] * 3.) * 3. * fa;
	    g[i__] -= (exp(x[i__] * 3.) * 3. + exp(x[i__])) * fa;
	    g[i__ + 1] -= exp(x[i__ + 1]) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L501: */
    }
    *f *= .5;
    return 0;
L510:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 1) / 2;
	if (ka % 2 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (x[i__] * 2. / (d__1 * d__1 + 1.) - x[i__ + 1]) * 10.;
/* Computing 2nd power */
	    d__1 = x[i__];
/* Computing 2nd power */
	    d__3 = x[i__];
/* Computing 2nd power */
	    d__2 = d__3 * d__3 + 1.;
	    g[i__] += (1. - d__1 * d__1) * 20. / (d__2 * d__2) * fa;
	    g[i__ + 1] -= fa * 10.;
	} else {
	    fa = x[i__] - 1.;
	    g[i__] += fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L511: */
    }
    *f *= .5;
    return 0;
L520:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 5) / 6 * 3 - 2;
	if (ka % 6 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 6 == 2) {
	    fa = x[i__ + 2] - 1.;
	    g[i__ + 2] += fa;
	} else if (ka % 6 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - 1.;
	    fa = d__1 * d__1;
	    g[i__ + 3] += (x[i__ + 3] - 1.) * 2. * fa;
	} else if (ka % 6 == 4) {
/* Computing 3rd power */
	    d__1 = x[i__ + 4] - 1.;
	    fa = d__1 * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = x[i__ + 4] - 1.;
	    g[i__ + 4] += d__1 * d__1 * 3. * fa;
	} else if (ka % 6 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = d__1 * d__1 * x[i__ + 3] + sin(x[i__ + 3] - x[i__ + 4]) - 
		    10.;
	    g[i__] += x[i__] * 2. * x[i__ + 3] * fa;
/* Computing 2nd power */
	    d__1 = x[i__];
	    g[i__ + 3] += (d__1 * d__1 + cos(x[i__ + 3] - x[i__ + 4])) * fa;
	    g[i__ + 4] -= cos(x[i__ + 3] - x[i__ + 4]) * fa;
	} else {
/* Computing 2nd power */
	    d__2 = x[i__ + 2];
/* Computing 2nd power */
	    d__1 = d__2 * d__2 * x[i__ + 3];
	    fa = x[i__ + 1] + d__1 * d__1 - 20.;
	    g[i__ + 1] += fa;
/* Computing 2nd power */
	    d__1 = x[i__ + 2] * x[i__ + 3];
	    g[i__ + 2] += x[i__ + 2] * 4. * (d__1 * d__1) * fa;
/* Computing 4th power */
	    d__1 = x[i__ + 2], d__1 *= d__1;
	    g[i__ + 3] += d__1 * d__1 * 2. * x[i__ + 3] * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L521: */
    }
    *f *= .5;
    return 0;
L530:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 7 == 2) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 2]) * 10.;
	    g[i__ + 1] += x[i__ + 1] * 20. * fa;
	    g[i__ + 2] -= fa * 10.;
	} else if (ka % 7 == 3) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2] - x[i__ + 3];
	    fa = d__1 * d__1;
	    g[i__ + 2] += (x[i__ + 2] - x[i__ + 3]) * 2. * fa;
	    g[i__ + 3] -= (x[i__ + 2] - x[i__ + 3]) * 2. * fa;
	} else if (ka % 7 == 4) {
/* Computing 2nd power */
	    d__1 = x[i__ + 3] - x[i__ + 4];
	    fa = d__1 * d__1;
	    g[i__ + 3] += (x[i__ + 3] - x[i__ + 4]) * 2. * fa;
	    g[i__ + 4] -= (x[i__ + 3] - x[i__ + 4]) * 2. * fa;
	} else if (ka % 7 == 5) {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = x[i__] + d__1 * d__1 + x[i__ + 2] - 30.;
	    g[i__] += fa;
	    g[i__ + 1] += x[i__ + 1] * 2. * fa;
	    g[i__ + 2] += fa;
	} else if (ka % 7 == 6) {
/* Computing 2nd power */
	    d__1 = x[i__ + 2];
	    fa = x[i__ + 1] - d__1 * d__1 + x[i__ + 3] - 10.;
	    g[i__ + 1] += fa;
	    g[i__ + 2] -= x[i__ + 2] * 2. * fa;
	    g[i__ + 3] += fa;
	} else {
	    fa = x[i__] * x[i__ + 4] - 10.;
	    g[i__] += x[i__ + 4] * fa;
	    g[i__ + 4] += x[i__] * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L531: */
    }
    *f *= .5;
    return 0;
L540:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		if (x[i__ + j] == 0.) {
		    x[i__ + j] = 1e-16;
		}
		d__2 = (d__1 = x[i__ + j], abs(d__1));
		d__3 = (doublereal) j / (doublereal) (k * l);
		a = a * d_sign(&c_b347, &x[i__ + j]) * pow_dd(&d__2, &d__3);
/* L541: */
	    }
	    fa += a;
/* L542: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	for (k = 1; k <= 3; ++k) {
	    a = (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		d__2 = (d__1 = x[i__ + j], abs(d__1));
		d__3 = (doublereal) j / (doublereal) (k * l);
		a = a * d_sign(&c_b347, &x[i__ + j]) * pow_dd(&d__2, &d__3);
/* L543: */
	    }
	    for (j = 1; j <= 4; ++j) {
		g[i__ + j] += (doublereal) j / (doublereal) (k * l) * a / x[
			i__ + j] * fa;
/* L544: */
	    }
/* L545: */
	}
/* L546: */
    }
    *f *= .5;
    return 0;
L550:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (k = 1; k <= 3; ++k) {
	    a = 0.;
	    for (j = 1; j <= 4; ++j) {
		a += x[i__ + j] * ((doublereal) j / (doublereal) (k * l));
/* L551: */
	    }
	    fa += exp(a) * (doublereal) (k * k) / (doublereal) l;
/* L552: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	for (k = 1; k <= 3; ++k) {
	    a = 0.;
	    for (j = 1; j <= 4; ++j) {
		a += x[i__ + j] * ((doublereal) j / (doublereal) (k * l));
/* L553: */
	    }
	    a = exp(a) * (doublereal) (k * k) / (doublereal) l;
	    for (j = 1; j <= 4; ++j) {
		g[i__ + j] += a * ((doublereal) j / (doublereal) (k * l)) * 
			fa;
/* L554: */
	    }
/* L555: */
	}
/* L556: */
    }
    *f *= .5;
    return 0;
L560:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ((ka + 3) / 4 << 1) - 2;
	l = (ka - 1) % 4 + 1;
	fa = -empr28_1.y[l - 1];
	for (j = 1; j <= 4; ++j) {
	    fa = fa + (doublereal) ((1 - (j % 2 << 1)) * l * j * j) * sin(x[
		    i__ + j]) + (doublereal) (l * l * j) * cos(x[i__ + j]);
/* L561: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	for (j = 1; j <= 4; ++j) {
	    g[i__ + j] += ((doublereal) ((1 - (j % 2 << 1)) * l * j * j) * 
		    cos(x[i__ + j]) - (doublereal) (l * l * j) * sin(x[i__ + 
		    j])) * fa;
/* L562: */
	}
/* L563: */
    }
    *f *= .5;
    return 0;
L570:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alfa = .5;
	if (ka == 1) {
	    fa = alfa - (1. - alfa) * x[3] - x[1] * (x[2] * 4. + 1.);
	    g[1] -= (x[2] * 4. + 1.) * fa;
	    g[2] -= x[1] * 4. * fa;
	    g[3] += (alfa - 1.) * fa;
	} else if (ka == 2) {
	    fa = -(2. - alfa) * x[4] - x[2] * (x[1] * 4. + 1.);
	    g[1] -= x[2] * 4. * fa;
	    g[2] -= (x[1] * 4. + 1.) * fa;
	    g[4] += (alfa - 2.) * fa;
	} else if (ka == *n - 1) {
	    fa = alfa * x[*n - 3] - x[*n - 1] * (x[*n] * 4. + 1.);
	    g[*n - 3] += alfa * fa;
	    g[*n - 1] -= (x[*n] * 4. + 1.) * fa;
	    g[*n] -= x[*n - 1] * 4. * fa;
	} else if (ka == *n) {
	    fa = alfa * x[*n - 2] - (2. - alfa) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	    g[*n - 2] += alfa * fa;
	    g[*n - 1] -= x[*n] * 4. * fa;
	    g[*n] -= (x[*n - 1] * 4. + 1.) * fa;
	} else if (ka % 2 == 1) {
	    fa = alfa * x[ka - 2] - (1. - alfa) * x[ka + 2] - x[ka] * (x[ka + 
		    1] * 4. + 1.);
	    g[ka - 2] += alfa * fa;
	    g[ka] -= (x[ka + 1] * 4. + 1.) * fa;
	    g[ka + 1] -= x[ka] * 4. * fa;
	    g[ka + 2] += (alfa - 1.) * fa;
	} else {
	    fa = alfa * x[ka - 2] - (2. - alfa) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	    g[ka - 2] += alfa * fa;
	    g[ka - 1] -= x[ka] * 4. * fa;
	    g[ka] -= (x[ka - 1] * 4. + 1.) * fa;
	    g[ka + 2] += (alfa - 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L571: */
    }
    *f *= .5;
    return 0;
L580:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = (x[ka] - d__1 * d__1) * 4.;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4.;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2.;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L581: */
    }
    *f *= .5;
    return 0;
L590:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka + 1] * 2. + x[*n - 4] *
		     3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] += fa * .5;
	    g[*n] -= fa;
	    g[ka] -= (x[ka] * 4. - 3.) * fa;
	    g[ka + 1] -= fa * 2.;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = d__1 * d__1 * -2. + x[ka] * 3. - x[ka - 1] - x[ka + 1] * 2. 
		    + x[*n - 4] * 3. - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 
		    - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] += fa * .5;
	    g[*n] -= fa;
	    g[ka - 1] -= fa;
	    g[ka] -= (x[ka] * 4. - 3.) * fa;
	    g[ka + 1] -= fa * 2.;
	} else {
/* Computing 2nd power */
	    d__1 = x[*n];
	    fa = d__1 * d__1 * -2. + x[*n] * 3. - x[*n - 1] + x[*n - 4] * 3. 
		    - x[*n - 3] - x[*n - 2] + x[*n - 1] * .5 - x[*n] + 1.;
	    g[*n - 4] += fa * 3.;
	    g[*n - 3] -= fa;
	    g[*n - 2] -= fa;
	    g[*n - 1] -= fa * .5;
	    g[*n] -= (x[*n] * 4. - 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L591: */
    }
    *f *= .5;
    return 0;
L600:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	u = 1. / (doublereal) (*n + 1);
	v = (doublereal) ka * u;
/* Computing 3rd power */
	d__1 = x[ka] + v + 1.;
	fa = x[ka] * 2. + u * .5 * u * (d__1 * (d__1 * d__1)) + 1.;
	if (ka > 1) {
	    fa -= x[ka - 1];
	}
	if (ka < *n) {
	    fa -= x[ka + 1];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing 2nd power */
	d__1 = u;
/* Computing 2nd power */
	d__2 = x[ka] + v + 1.;
	g[ka] += (d__1 * d__1 * 1.5 * (d__2 * d__2) + 2.) * fa;
	if (ka > 1) {
	    g[ka - 1] -= fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= fa;
	}
/* L601: */
    }
    *f *= .5;
    return 0;
L610:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = (ka + 6) / 7 * 3 - 2;
	if (ka % 7 == 1) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else if (ka % 7 == 2) {
	    fa = x[i__ + 1] + x[i__ + 2] - 2.;
	    g[i__ + 1] += fa;
	    g[i__ + 2] += fa;
	} else if (ka % 7 == 3) {
	    fa = x[i__ + 3] - 1.;
	    g[i__ + 3] += fa;
	} else if (ka % 7 == 4) {
	    fa = x[i__ + 4] - 1.;
	    g[i__ + 4] += fa;
	} else if (ka % 7 == 5) {
	    fa = x[i__] + x[i__ + 1] * 3.;
	    g[i__] += fa;
	    g[i__ + 1] += fa * 3.;
	} else if (ka % 7 == 6) {
	    fa = x[i__ + 2] + x[i__ + 3] - x[i__ + 4] * 2.;
	    g[i__ + 2] += fa;
	    g[i__ + 3] += fa;
	    g[i__ + 4] -= fa * 2.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__ + 1];
	    fa = (d__1 * d__1 - x[i__ + 4]) * 10.;
	    g[i__ + 1] += x[i__ + 1] * 20. * fa;
	    g[i__ + 4] -= fa * 10.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L611: */
    }
    *f *= .5;
    return 0;
L620:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	i__ = ka / 2;
	if (ka == 1) {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	} else if (ka % 2 == 0) {
/* Computing 2nd power */
	    d__1 = x[i__];
	    fa = (d__1 * d__1 - x[i__ + 1]) * 10.;
	    g[i__] += x[i__] * 20. * fa;
	    g[i__ + 1] -= fa * 10.;
	} else {
/* Computing 2nd power */
	    d__1 = x[i__] - x[i__ + 1];
	    a = exp(-(d__1 * d__1)) * 2.;
/* Computing 2nd power */
	    d__1 = x[i__ + 1] - x[i__ + 2];
	    b = exp(d__1 * d__1 * -2.);
	    fa = a + b;
	    g[i__] -= (x[i__] - x[i__ + 1]) * 2. * a * fa;
	    g[i__ + 1] += ((x[i__] - x[i__ + 1]) * 2. * a - (x[i__ + 1] - x[
		    i__ + 2]) * 4. * b) * fa;
	    g[i__ + 2] += (x[i__ + 1] - x[i__ + 2]) * 4. * b * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L621: */
    }
    *f *= .5;
    return 0;
L630:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing MIN */
/* Computing MAX */
	i__2 = ka % 13 - 2;
	i__4 = max(i__2,1);
	ia = min(i__4,7);
	ib = (ka + 12) / 13;
	i__ = ia + ib - 1;
	if (ia == 7) {
	    j = ib;
	} else {
	    j = ia + ib;
	}
	c__ = (doublereal) ia * 3. / 10.;
	a = 0.;
	b = exp(sin(c__) * x[j]);
	d__ = x[j] - sin(x[i__]) - 1. + empr28_1.y[0];
	e = cos(c__) + 1.;
	for (l = 0; l <= 6; ++l) {
	    if (ib + l != i__ && ib + l != j) {
		a = a + sin(x[ib + l]) - empr28_1.y[0];
	    }
/* L631: */
	}
/* Computing 2nd power */
	d__1 = d__;
	fa = e * (d__1 * d__1) + (x[i__] - 1.) * 5. * b + a * .5;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[i__] -= (d__ * 2. * e * cos(x[i__]) - b * 5.) * fa;
	g[j] += (d__ * 2. * e + (x[i__] - 1.) * 5. * b * sin(c__)) * fa;
	for (l = 0; l <= 6; ++l) {
	    if (ib + l != i__ && ib + l != j) {
		g[ib + l] += cos(x[ib + l]) * .5 * fa;
	    }
/* L632: */
	}
/* L633: */
    }
    *f *= .5;
    return 0;
L720:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a1 = .414214;
	if (ka == 1) {
	    fa = x[1] - (1. - x[1]) * x[3] - a1 * (x[2] * 4. + 1.);
	    g[1] += (x[3] + 1.) * fa;
	    g[2] -= a1 * 4. * fa;
	    g[3] -= (1. - x[1]) * fa;
	} else if (ka == 2) {
	    fa = -(1. - x[1]) * x[4] - a1 * (x[2] * 4. + 1.);
	    g[1] += x[4] * fa;
	    g[2] -= a1 * 4. * fa;
	    g[4] -= (1. - x[1]) * fa;
	} else if (ka == 3) {
	    fa = a1 * x[1] - (1. - x[1]) * x[5] - x[3] * (x[2] * 4. + 1.);
	    g[1] += (a1 + x[5]) * fa;
	    g[2] -= x[3] * 4. * fa;
	    g[3] -= (x[2] * 4. + 1.) * fa;
	    g[5] -= (1. - x[1]) * fa;
	} else if (ka <= *n - 2) {
	    fa = x[1] * x[ka - 2] - (1. - x[1]) * x[ka + 2] - x[ka] * (x[ka - 
		    1] * 4. + 1.);
	    g[1] += (x[ka - 2] + x[ka + 2]) * fa;
	    g[ka - 2] += x[1] * fa;
	    g[ka - 1] -= x[ka] * 4. * fa;
	    g[ka] -= (x[ka - 1] * 4. + 1.) * fa;
	    g[ka + 2] -= (1. - x[1]) * fa;
	} else if (ka == *n - 1) {
	    fa = x[1] * x[*n - 3] - x[*n - 1] * (x[*n - 2] * 4. + 1.);
	    g[1] += x[*n - 3] * fa;
	    g[*n - 3] += x[1] * fa;
	    g[*n - 2] -= x[*n - 1] * 4. * fa;
	    g[*n - 1] -= (x[*n - 2] * 4. + 1.) * fa;
	} else {
	    fa = x[1] * x[*n - 2] - (1. - x[1]) - x[*n] * (x[*n - 1] * 4. + 
		    1.);
	    g[1] += (x[*n - 2] + 1.) * fa;
	    g[*n - 2] += x[1] * fa;
	    g[*n - 1] -= x[*n] * 4. * fa;
	    g[*n] -= (x[*n - 1] * 4. + 1.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L721: */
    }
    *f *= .5;
    return 0;
L740:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d1s = cos(x[ka] - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d2s = sin(x[ka] - x[ka + 1]) * cos(x[ka] + x[ka + 1]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 9. + d1s + d2s) * fa;
	    g[ka + 1] += (2. - d1s + d2s) * fa;
	} else if (ka < *n) {
/* Computing 3rd power */
	    d__1 = x[ka];
	    fa = d__1 * (d__1 * d__1) * 3. + x[ka + 1] * 2. - 5. + sin(x[ka] 
		    - x[ka + 1]) * sin(x[ka] + x[ka + 1]) + x[ka] * 4. - x[ka 
		    - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	    d1s = cos(x[ka] - x[ka + 1]) * sin(x[ka] + x[ka + 1]);
	    d2s = sin(x[ka] - x[ka + 1]) * cos(x[ka] + x[ka + 1]);
	    ex = exp(x[ka - 1] - x[ka]);
	    g[ka - 1] -= (ex + x[ka - 1] * ex) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 9. + d1s + d2s + 4. + x[ka - 1] * ex) * 
		    fa;
	    g[ka + 1] += (2. - d1s + d2s) * fa;
	} else {
	    fa = x[ka] * 4. - x[ka - 1] * exp(x[ka - 1] - x[ka]) - 3.;
	    ex = exp(x[ka - 1] - x[ka]);
	    g[ka - 1] -= (ex + x[ka - 1] * ex) * fa;
	    g[ka] += (x[ka - 1] * ex + 4.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L741: */
    }
    *f *= .5;
    return 0;
L750:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = 0.;
	    if (ka != 1) {
/* Computing 3rd power */
		d__1 = x[ka - 2] - x[ka];
		fa = fa - d__1 * (d__1 * d__1) * 6. + 10. - x[ka - 1] * 4. - 
			sin(x[ka - 2] - x[ka - 1] - x[ka]) * 2. * sin(x[ka - 
			2] + x[ka - 1] - x[ka]);
	    }
	    if (ka != *n) {
/* Computing 3rd power */
		d__1 = x[ka] - x[ka + 2];
		fa = fa + d__1 * (d__1 * d__1) * 3. - 5. + x[ka + 1] * 2. + 
			sin(x[ka] - x[ka + 1] - x[ka + 2]) * sin(x[ka] + x[ka 
			+ 1] - x[ka + 2]);
	    }
	    if (ka != 1) {
		d1s = cos(x[ka - 2] - x[ka - 1] - x[ka]) * sin(x[ka - 2] + x[
			ka - 1] - x[ka]);
		d2s = sin(x[ka - 2] - x[ka - 1] - x[ka]) * cos(x[ka - 2] + x[
			ka - 1] - x[ka]);
/* Computing 2nd power */
		d__1 = x[ka - 2] - x[ka];
		g[ka - 2] -= (d__1 * d__1 * 18. + (d1s + d2s) * 2.) * fa;
		g[ka - 1] -= (4. - (d1s - d2s) * 2.) * fa;
/* Computing 2nd power */
		d__1 = x[ka - 2] - x[ka];
		g[ka] += (d__1 * d__1 * 18. + (d1s + d2s) * 2.) * fa;
	    }
	    if (ka != *n) {
		d1s = cos(x[ka] - x[ka + 1] - x[ka + 2]) * sin(x[ka] + x[ka + 
			1] - x[ka + 2]);
		d2s = sin(x[ka] - x[ka + 1] - x[ka + 2]) * cos(x[ka] + x[ka + 
			1] - x[ka + 2]);
/* Computing 2nd power */
		d__1 = x[ka] - x[ka + 2];
		g[ka] += (d__1 * d__1 * 9. + d1s + d2s) * fa;
		g[ka + 1] += (2. - d1s + d2s) * fa;
/* Computing 2nd power */
		d__1 = x[ka] - x[ka + 2];
		g[ka + 2] -= (d__1 * d__1 * 9. + d1s + d2s) * fa;
	    }
	} else {
	    ex = exp(x[ka - 1] - x[ka] - x[ka + 1]);
	    fa = x[ka] * 4. - (x[ka - 1] - x[ka + 1]) * ex - 3.;
	    w = x[ka - 1] - x[ka + 1];
	    g[ka - 1] -= (ex + w * ex) * fa;
	    g[ka] += (w * ex + 4.) * fa;
	    g[ka + 1] += (ex + w * ex) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L751: */
    }
    *f *= .5;
    return 0;
L760:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.;
	    fa = d__1 * d__1;
	    g[1] += ((3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.) * 2. * (3. - 
		    h__ * 2. * x[1]) * fa;
	    g[2] -= ((3. - h__ * x[1]) * x[1] - x[2] * 2. + 1.) * 4. * fa;
	} else if (ka <= *n - 1) {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] * 2. + 
		    1.;
	    fa = d__1 * d__1;
	    g[ka - 1] -= ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] *
		     2. + 1.) * 2. * fa;
	    g[ka] += ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] * 2. 
		    + 1.) * 2. * (3. - h__ * 2. * x[ka]) * fa;
	    g[ka + 1] -= ((3. - h__ * x[ka]) * x[ka] - x[ka - 1] - x[ka + 1] *
		     2. + 1.) * 4. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = (3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.;
	    fa = d__1 * d__1;
	    g[*n - 1] -= ((3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.) * 2. * 
		    fa;
	    g[*n] += ((3. - h__ * x[*n]) * x[*n] - x[*n - 1] + 1.) * 2. * (3. 
		    - h__ * 2. * x[*n]) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L761: */
    }
    *f *= .5;
    return 0;
L780:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + x[ka + 1] - d__3 * d__3;
	    g[ka - 1] -= x[ka] * 8. * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4;
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= x[ka + 2] * 2. * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2];
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= x[ka + 1] * 8. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2];
	    g[ka - 2] -= fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L781: */
    }
    *f *= .5;
    return 0;
L790:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
/* Computing 2nd power */
	    d__2 = x[ka + 2];
/* Computing 2nd power */
	    d__3 = x[ka + 3];
	    fa = (x[ka] - d__1 * d__1) * 4. + x[ka + 1] - d__2 * d__2 + x[ka 
		    + 2] - d__3 * d__3;
	    g[ka] += fa * 4.;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 + x[ka + 1] - 
		    d__4 * d__4 + x[ka + 2] - d__5 * d__5;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < 4) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 + x[ka + 2] - d__6 * 
		    d__6;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < *n - 2) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
/* Computing 2nd power */
	    d__6 = x[ka + 3];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2] - d__6 * d__6;
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	    g[ka + 3] -= x[ka + 3] * 2. * fa;
	} else if (ka < *n - 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka + 2];
/* Computing 2nd power */
	    d__5 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] - d__4 * d__4 + d__5 * d__5 - x[ka - 3] + x[ka + 
		    2];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	    g[ka + 2] -= (x[ka + 2] * 2. - 1.) * fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka + 1];
/* Computing 2nd power */
	    d__3 = x[ka - 1];
/* Computing 2nd power */
	    d__4 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    (x[ka] - d__2 * d__2) * 4. + d__3 * d__3 - x[ka - 2] + x[
		    ka + 1] + d__4 * d__4 - x[ka - 3];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 6.) * fa;
	    g[ka + 1] -= (x[ka + 1] * 8. - 1.) * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[ka - 1];
/* Computing 2nd power */
	    d__3 = x[ka - 2];
	    fa = x[ka] * 8. * (d__1 * d__1 - x[ka - 1]) - (1. - x[ka]) * 2. + 
		    d__2 * d__2 - x[ka - 2] + d__3 * d__3 - x[ka - 3];
	    g[ka - 3] -= fa;
	    g[ka - 2] += (x[ka - 2] * 2. - 1.) * fa;
	    g[ka - 1] -= (x[ka] * 8. - x[ka - 1] * 2.) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 24. - x[ka - 1] * 8. + 2.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L791: */
    }
    *f *= .5;
    return 0;
L810:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] + ((5. - x[ka + 1]) * x[ka + 1] - 2.) * x[ka + 1] - 
		    13.;
	    g[ka] += fa;
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    g[ka + 1] += (x[ka + 1] * 10. - d__1 * d__1 * 3. - 2.) * fa;
	} else {
	    fa = x[ka - 1] + ((x[ka] + 1.) * x[ka] - 14.) * x[ka] - 29.;
	    g[ka - 1] += fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] += (d__1 * d__1 * 3. + x[ka] * 2. - 14.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L811: */
    }
    *f *= .5;
    return 0;
L830:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
	    a = exp(x[ka]) - x[ka + 1];
/* Computing 2nd power */
	    d__1 = a;
	    fa = d__1 * d__1;
	    g[ka] += a * 2. * exp(x[ka]) * fa;
	    g[ka + 1] -= a * 2. * fa;
	} else if (ka % 4 == 2) {
/* Computing 3rd power */
	    d__1 = x[ka] - x[ka + 1];
	    fa = d__1 * (d__1 * d__1) * 10.;
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka + 1];
	    a = d__1 * d__1 * 30. * fa;
	    g[ka] += a;
	    g[ka + 1] -= a;
	} else if (ka % 4 == 3) {
	    a = x[ka] - x[ka + 1];
/* Computing 2nd power */
	    d__1 = sin(a) / cos(a);
	    fa = d__1 * d__1;
/* Computing 3rd power */
	    d__1 = cos(a);
	    b = sin(a) * 2. / (d__1 * (d__1 * d__1)) * fa;
	    g[ka] += b;
	    g[ka + 1] -= b;
	} else {
	    fa = x[ka] - 1.;
	    g[ka] += fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L831: */
    }
    *f *= .5;
    return 0;
L840:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    fa = x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	    g[ka] += (x[ka] - 3.) * fa;
	    g[ka + 1] += fa * 2.;
	} else if (ka < *n) {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1. + x[ka + 1] * 2.;
	    g[ka - 1] += fa;
	    g[ka] += (x[ka] - 3.) * fa;
	    g[ka + 1] += fa * 2.;
	} else {
	    fa = x[ka - 1] + x[ka] * (x[ka] * .5 - 3.) - 1.;
	    g[ka - 1] += fa;
	    g[ka] += (x[ka] - 3.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L841: */
    }
    *f *= .5;
    return 0;
L860:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 2 == 1) {
	    fa = x[ka] * 1e4 * x[ka + 1] - 1.;
	    g[ka] += x[ka + 1] * 1e4 * fa;
	    g[ka + 1] += x[ka] * 1e4 * fa;
	} else {
	    fa = exp(-x[ka - 1]) + exp(-x[ka]) - 1.0001;
	    g[ka - 1] -= exp(-x[ka - 1]) * fa;
	    g[ka] -= exp(-x[ka]) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L861: */
    }
    *f *= .5;
    return 0;
L870:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka % 4 == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -200. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] -= ((x[ka + 1] - d__1 * d__1 * 3.) * 200. - 1.) * fa;
	    g[ka + 1] -= x[ka] * 200. * fa;
	} else if (ka % 4 == 2) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 200. + (x[ka] - 1.) * 20.2 + (x[ka + 
		    2] - 1.) * 19.8;
	    g[ka - 1] -= x[ka - 1] * 400. * fa;
	    g[ka] += fa * 220.2;
	    g[ka + 2] += fa * 19.8;
	} else if (ka % 4 == 3) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * -180. * (x[ka + 1] - d__1 * d__1) - (1. - x[ka]);
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka] -= ((x[ka + 1] - d__1 * d__1 * 3.) * 180. - 1.) * fa;
	    g[ka + 1] -= x[ka] * 180. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = (x[ka] - d__1 * d__1) * 180. + (x[ka] - 1.) * 20.2 + (x[ka - 
		    2] - 1.) * 19.8;
	    g[ka - 2] += fa * 19.8;
	    g[ka - 1] -= x[ka - 1] * 360. * fa;
	    g[ka] += fa * 200.2;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L871: */
    }
    *f *= .5;
    return 0;
L880:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka < 2) {
	    a = exp(cos((doublereal) ka * (x[ka] + x[ka + 1])));
	    b = a * (doublereal) ka * sin((doublereal) ka * (x[ka] + x[ka + 1]
		    ));
	    fa = x[ka] - a;
	    g[ka + 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	} else if (ka < *n) {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka] + x[ka + 1])) * (
		    doublereal) ka;
	    fa = x[ka] - a;
	    g[ka - 1] += b * fa;
	    g[ka + 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	} else {
	    a = exp(cos((doublereal) ka * (x[ka - 1] + x[ka])));
	    b = a * sin((doublereal) ka * (x[ka - 1] + x[ka])) * (doublereal) 
		    ka;
	    fa = x[ka] - a;
	    g[ka - 1] += b * fa;
	    g[ka] += (b + 1.) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L881: */
    }
    *f *= .5;
    return 0;
L900:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2.) + d__1 * d__1 * .25;
	    g[ka] += (x[ka + 1] - x[ka] * 4.) * 3. * fa;
	    g[ka + 1] += (x[ka] * 3. + x[ka + 1] * .5) * fa;
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = 20. - x[ka - 1];
	    fa = x[ka] * 3. * (20. - x[ka] * 2. + x[ka - 1]) + d__1 * d__1 * 
		    .25;
	    g[ka - 1] += (x[ka] * 3. - (20. - x[ka - 1]) * .5) * fa;
	    g[ka] += (20. - x[ka] * 4. + x[ka - 1]) * 3. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
	    fa = x[ka] * 3. * (x[ka + 1] - x[ka] * 2. + x[ka - 1]) + d__1 * 
		    d__1 * .25;
	    g[ka - 1] += (x[ka] * 3. - (x[ka + 1] - x[ka - 1]) * .5) * fa;
	    g[ka] += (x[ka + 1] - x[ka] * 4. + x[ka - 1]) * 3. * fa;
	    g[ka + 1] += (x[ka] * 3. + (x[ka + 1] - x[ka - 1]) * .5) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L901: */
    }
    *f *= .5;
    return 0;
L910:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
	if (ka < 2) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka + 1];
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka < *n) {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - x[ka + 1];
	    g[ka - 1] -= fa;
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka + 1] -= fa;
	} else {
/* Computing 2nd power */
	    d__1 = h__;
	    fa = x[ka] * 2. + empr28_1.par * (d__1 * d__1) * sinh(
		    empr28_1.par * x[ka]) - x[ka - 1] - 1.;
/* Computing 2nd power */
	    d__1 = empr28_1.par;
/* Computing 2nd power */
	    d__2 = h__;
	    g[ka] += (d__1 * d__1 * (d__2 * d__2) * cosh(empr28_1.par * x[ka])
		     + 2.) * fa;
	    g[ka - 1] -= fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L911: */
    }
    *f *= .5;
    return 0;
L920:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 6.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	if (ka > 1) {
	    fa -= x[ka - 1] * 4.;
	    a1 -= x[ka - 1];
	    a2 += x[ka - 1];
	    a3 += x[ka - 1] * 2.;
	}
	if (ka > 2) {
	    fa += x[ka - 2];
	    a3 -= x[ka - 2];
	}
	if (ka < *n - 1) {
	    fa += x[ka + 2];
	    a3 += x[ka + 2];
	}
	if (ka < *n) {
	    fa -= x[ka + 1] * 4.;
	    a1 += x[ka + 1];
	    a2 += x[ka + 1];
	    a3 -= x[ka + 1] * 2.;
	}
	if (ka >= *n - 1) {
	    fa += 1.;
	    a3 += 1.;
	}
	if (ka >= *n) {
	    fa += -4.;
	    a1 += 1.;
	    a2 += 1.;
	    a3 += -2.;
	}
	fa -= empr28_1.par * .5 * (a1 * a2 - x[ka] * a3);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += fa * 6.;
	ga1[0] = 0.;
	ga1[1] = 0.;
	ga2[0] = 0.;
	ga2[1] = 0.;
	if (ka > 1) {
	    g[ka - 1] -= (4. - empr28_1.par * x[ka]) * fa;
	    ga1[0] = -1.;
	    ga2[0] = 1.;
	}
	if (ka > 2) {
	    g[ka - 2] += (1. - empr28_1.par * .5 * x[ka]) * fa;
	}
	if (ka < *n - 1) {
	    g[ka + 2] += (empr28_1.par * .5 * x[ka] + 1.) * fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= (empr28_1.par * x[ka] + 4.) * fa;
	    ga1[1] = 1.;
	    ga2[1] = 1.;
	}
	g[ka] += empr28_1.par * .5 * a3 * fa;
	if (ka > 1) {
	    g[ka - 1] -= empr28_1.par * .5 * (ga1[0] * a2 + a1 * ga2[0]) * fa;
	}
	if (ka < *n) {
	    g[ka + 1] -= empr28_1.par * .5 * (ga1[1] * a2 + a1 * ga2[1]) * fa;
	}
/* L921: */
    }
    *f *= .5;
    return 0;
L930:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (empr28_1.m + 1);
	if (ka <= empr28_1.m) {
	    j = ka + empr28_1.m;
	    fa = x[ka] * 6.;
	    a1 = 0.;
	    a2 = 0.;
	    if (ka == 1) {
		a1 += 1.;
	    }
	    if (ka > 1) {
		fa -= x[ka - 1] * 4.;
		a1 -= x[j - 1];
		a2 += x[ka - 1] * 2.;
	    }
	    if (ka > 2) {
		fa += x[ka - 2];
		a2 -= x[ka - 2];
	    }
	    if (ka < empr28_1.m - 1) {
		fa += x[ka + 2];
		a2 += x[ka + 2];
	    }
	    if (ka < empr28_1.m) {
		fa -= x[ka + 1] * 4.;
		a1 += x[j + 1];
		a2 -= x[ka + 1] * 2.;
	    }
	    if (ka == empr28_1.m) {
		a1 += 1.;
	    }
/* Computing 2nd power */
	    d__1 = h__;
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a2 + x[j] * a1 * (d__1 * 
		    d__1));
	} else {
	    j = ka - empr28_1.m;
	    fa = x[ka] * -2.;
	    a1 = 0.;
	    a2 = 0.;
	    if (j == 1) {
		a2 += 1.;
	    }
	    if (j > 1) {
		fa += x[ka - 1];
		a1 -= x[j - 1];
		a2 -= x[ka - 1];
	    }
	    if (j < empr28_1.m) {
		fa += x[ka + 1];
		a1 += x[j + 1];
		a2 += x[ka + 1];
	    }
	    if (j == empr28_1.m) {
		a2 += 1.;
	    }
	    fa += empr28_1.par * .5 * h__ * (x[ka] * a1 + x[j] * a2);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	if (ka <= empr28_1.m) {
	    g[ka] += fa * 6.;
	    if (ka > 1) {
		g[ka - 1] -= (4. - empr28_1.par * h__ * x[ka]) * fa;
/* Computing 3rd power */
		d__1 = h__;
		g[j - 1] -= empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * x[j] 
			* fa;
	    }
	    if (ka > 2) {
		g[ka - 2] += (1. - empr28_1.par * .5 * h__ * x[ka]) * fa;
	    }
	    if (ka < empr28_1.m - 1) {
		g[ka + 2] += (empr28_1.par * .5 * h__ * x[ka] + 1.) * fa;
	    }
	    if (ka < empr28_1.m) {
		g[ka + 1] -= (empr28_1.par * h__ * x[ka] + 4.) * fa;
/* Computing 3rd power */
		d__1 = h__;
		g[j + 1] += empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * x[j] 
			* fa;
	    }
	    g[ka] += empr28_1.par * .5 * h__ * a2 * fa;
/* Computing 3rd power */
	    d__1 = h__;
	    g[j] += empr28_1.par * .5 * (d__1 * (d__1 * d__1)) * a1 * fa;
	} else {
	    g[ka] -= fa * 2.;
	    if (j > 1) {
		g[ka - 1] += (1. - empr28_1.par * .5 * h__ * x[j]) * fa;
		g[j - 1] -= empr28_1.par * .5 * h__ * x[ka] * fa;
	    }
	    if (j < empr28_1.m) {
		g[ka + 1] += (empr28_1.par * .5 * h__ * x[j] + 1.) * fa;
		g[j + 1] += empr28_1.par * .5 * h__ * x[ka] * fa;
	    }
	    g[ka] += empr28_1.par * .5 * h__ * a1 * fa;
	    g[j] += empr28_1.par * .5 * h__ * a2 * fa;
	}
/* L931: */
    }
    *f *= .5;
    return 0;
L940:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	fa = x[ka] * 4. - empr28_1.par * exp(x[ka]);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += (4. - empr28_1.par * exp(x[ka])) * fa;
	if (j > 1) {
	    g[ka - empr28_1.m] -= fa;
	}
	if (i__ > 1) {
	    g[ka - 1] -= fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= fa;
	}
/* L941: */
    }
    *f *= .5;
    return 0;
L950:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 3rd power */
	d__1 = x[ka];
/* Computing 2nd power */
	d__2 = (doublereal) i__;
/* Computing 2nd power */
	d__3 = (doublereal) j;
	fa = x[ka] * 4. + empr28_1.par * (d__1 * (d__1 * d__1)) / (
		empr28_1.par * (d__2 * d__2) + 1. + empr28_1.par * (d__3 * 
		d__3));
	if (i__ == 1) {
	    fa += -1.;
	}
	if (i__ > 1) {
	    fa -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	}
	if (i__ == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) j / (doublereal) (empr28_1.m + 1))
		    ;
	}
	if (j == 1) {
	    fa += -1.;
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    fa = fa - 2. + exp((doublereal) i__ / (doublereal) (empr28_1.m + 
		    1));
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing 2nd power */
	d__1 = x[ka];
/* Computing 2nd power */
	d__2 = (doublereal) i__;
/* Computing 2nd power */
	d__3 = (doublereal) j;
	g[ka] += (empr28_1.par * 3. * (d__1 * d__1) / (empr28_1.par * (d__2 * 
		d__2) + 1. + empr28_1.par * (d__3 * d__3)) + 4.) * fa;
	if (j > 1) {
	    g[ka - empr28_1.m] -= fa;
	}
	if (i__ > 1) {
	    g[ka - 1] -= fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= fa;
	}
/* L951: */
    }
    *f *= .5;
    return 0;
L960:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = (doublereal) i__ / (doublereal) (empr28_1.m + 1);
	a2 = (doublereal) j / (doublereal) (empr28_1.m + 1);
/* Computing 2nd power */
	d__1 = a1 - .25;
/* Computing 2nd power */
	d__2 = a2 - .75;
	fa = x[ka] * 4. - empr28_1.par * sin(pi * 2. * x[ka]) - (d__1 * d__1 
		+ d__2 * d__2) * 1e4 * empr28_1.par;
	if (i__ == 1) {
	    fa = fa - x[ka + 1] - empr28_1.par * sin(pi * x[ka + 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    fa = fa - x[ka + 1] - x[ka - 1] - empr28_1.par * sin(pi * (x[ka + 
		    1] - x[ka - 1]) * (doublereal) (empr28_1.m + 1));
	}
	if (i__ == empr28_1.m) {
	    fa = fa - x[ka - 1] + empr28_1.par * sin(pi * x[ka - 1] * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == 1) {
	    fa = fa - x[ka + empr28_1.m] - empr28_1.par * sin(pi * x[ka + 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
	if (j > 1 && j < empr28_1.m) {
	    fa = fa - x[ka + empr28_1.m] - x[ka - empr28_1.m] - empr28_1.par *
		     sin(pi * (x[ka + empr28_1.m] - x[ka - empr28_1.m]) * (
		    doublereal) (empr28_1.m + 1));
	}
	if (j == empr28_1.m) {
	    fa = fa - x[ka - empr28_1.m] + empr28_1.par * sin(pi * x[ka - 
		    empr28_1.m] * (doublereal) (empr28_1.m + 1));
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += (4. - pi * 2. * empr28_1.par * cos(pi * 2. * x[ka])) * fa;
	if (i__ == 1) {
	    g[ka + 1] -= (pi * (doublereal) (empr28_1.m + 1) * empr28_1.par * 
		    cos(pi * x[ka + 1] * (doublereal) (empr28_1.m + 1)) + 1.) 
		    * fa;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    g[ka - 1] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + 1] - x[ka - 1]) * (
		    doublereal) (empr28_1.m + 1))) * fa;
	    g[ka + 1] -= (pi * (doublereal) (empr28_1.m + 1) * empr28_1.par * 
		    cos(pi * (x[ka + 1] - x[ka - 1]) * (doublereal) (
		    empr28_1.m + 1)) + 1.) * fa;
	}
	if (i__ == empr28_1.m) {
	    g[ka - 1] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka - 1] * (doublereal) (
		    empr28_1.m + 1))) * fa;
	}
	if (j == 1) {
	    g[ka + empr28_1.m] -= (pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka + empr28_1.m] * (doublereal) 
		    (empr28_1.m + 1)) + 1.) * fa;
	}
	if (j > 1 && j < empr28_1.m) {
	    g[ka - empr28_1.m] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m]) * (doublereal) (empr28_1.m + 1))) * fa;
	    g[ka + empr28_1.m] -= (pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m]) * (doublereal) (empr28_1.m + 1)) + 1.) * fa;
	}
	if (j == empr28_1.m) {
	    g[ka - empr28_1.m] -= (1. - pi * (doublereal) (empr28_1.m + 1) * 
		    empr28_1.par * cos(pi * x[ka - empr28_1.m] * (doublereal) 
		    (empr28_1.m + 1))) * fa;
	}
/* L961: */
    }
    *f *= .5;
    return 0;
L970:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing 2nd power */
	d__1 = x[ka];
	fa = d__1 * d__1 * 8.;
	if (i__ == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - 1.;
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + 1.) - d__1 * d__1 * .5 - d__2 
		    * d__2 * 1.5 * (x[ka + 1] - 1.) * empr28_1.par;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + 1] - x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * (x[ka + 1] + x[ka - 1]) - d__1 * d__1 * .5 
		    - d__2 * d__2 * 1.5 * (x[ka + 1] - x[ka - 1]) * 
		    empr28_1.par;
	}
	if (i__ == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    fa = fa - x[ka] * 2. * x[ka - 1] - d__1 * d__1 * .5 + d__2 * d__2 
		    * 1.5 * x[ka - 1] * empr28_1.par;
	}
	if (j == 1) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - 1.;
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + 1.) - d__1 * d__1 * 
		    .5;
	}
	if (j > 1 && j < empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka + empr28_1.m] - x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * (x[ka + empr28_1.m] + x[ka - empr28_1.m]) 
		    - d__1 * d__1 * .5;
	}
	if (j == empr28_1.m) {
/* Computing 2nd power */
	    d__1 = x[ka - empr28_1.m];
	    fa = fa - x[ka] * 2. * x[ka - empr28_1.m] - d__1 * d__1 * .5;
	}
	if (i__ == 1 && j == 1) {
	    fa -= empr28_1.par / (doublereal) (empr28_1.m + 1);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += x[ka] * 16. * fa;
	if (i__ == 1) {
	    g[ka] -= ((x[ka + 1] + 1.) * 2. + x[ka] * 3. * (x[ka + 1] - 1.) * 
		    empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka + 1] -= (x[ka] * 2. + (x[ka + 1] - 1.) + d__1 * d__1 * 1.5 * 
		    empr28_1.par) * fa;
	}
	if (i__ > 1 && i__ < empr28_1.m) {
	    g[ka] -= ((x[ka + 1] + x[ka - 1]) * 2. + x[ka] * 3. * (x[ka + 1] 
		    - x[ka - 1]) * empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka - 1] -= (x[ka] * 2. - (x[ka + 1] - x[ka - 1]) - d__1 * d__1 *
		     1.5 * empr28_1.par) * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka + 1] -= (x[ka] * 2. + (x[ka + 1] - x[ka - 1]) + d__1 * d__1 *
		     1.5 * empr28_1.par) * fa;
	}
	if (i__ == empr28_1.m) {
	    g[ka] -= (x[ka - 1] * 2. - x[ka] * 3. * x[ka - 1] * empr28_1.par) 
		    * fa;
/* Computing 2nd power */
	    d__1 = x[ka];
	    g[ka - 1] -= (x[ka] * 2. + x[ka - 1] - d__1 * d__1 * 1.5 * 
		    empr28_1.par) * fa;
	}
	if (j == 1) {
	    g[ka] -= (x[ka + empr28_1.m] + 1.) * 2. * fa;
	    g[ka + empr28_1.m] -= (x[ka] * 2. + (x[ka + empr28_1.m] - 1.)) * 
		    fa;
	}
	if (j > 1 && j < empr28_1.m) {
	    g[ka] -= (x[ka + empr28_1.m] + x[ka - empr28_1.m]) * 2. * fa;
	    g[ka - empr28_1.m] -= (x[ka] * 2. - (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m])) * fa;
	    g[ka + empr28_1.m] -= (x[ka] * 2. + (x[ka + empr28_1.m] - x[ka - 
		    empr28_1.m])) * fa;
	}
	if (j == empr28_1.m) {
	    g[ka] -= x[ka - empr28_1.m] * 2. * fa;
	    g[ka - empr28_1.m] -= (x[ka] * 2. + x[ka - empr28_1.m]) * fa;
	}
/* L971: */
    }
    *f *= .5;
    return 0;
L980:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a3 = 0.;
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	a1 = empr28_1.par * (doublereal) i__;
	a2 = empr28_1.par * (doublereal) j;
/* Computing 2nd power */
	d__1 = empr28_1.par;
	fa = x[ka] * 4. - a1 * 2e3 * a2 * (1. - a1) * (1. - a2) * (d__1 * 
		d__1);
	if (i__ > 1) {
	    fa -= x[ka - 1];
	    a3 -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1];
	    a3 += x[ka + 1];
	}
	if (j > 1) {
	    fa -= x[ka - empr28_1.m];
	    a3 -= x[ka - empr28_1.m];
	}
	if (j < empr28_1.m) {
	    fa -= x[ka + empr28_1.m];
	    a3 += x[ka + empr28_1.m];
	}
	fa += empr28_1.par * 20. * a3 * x[ka];
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += fa * 4.;
	if (i__ > 1) {
	    g[ka - 1] -= (empr28_1.par * 20. * x[ka] + 1.) * fa;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= (1. - empr28_1.par * 20. * x[ka]) * fa;
	}
	if (j > 1) {
	    g[ka - empr28_1.m] -= (empr28_1.par * 20. * x[ka] + 1.) * fa;
	}
	if (j < empr28_1.m) {
	    g[ka + empr28_1.m] -= (1. - empr28_1.par * 20. * x[ka]) * fa;
	}
	g[ka] += empr28_1.par * 20. * a3 * fa;
/* L981: */
    }
    *f *= .5;
    return 0;
L990:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
/* Computing MAX */
	d__1 = 0., d__2 = x[ka];
	d__3 = (doublereal) i__ / (doublereal) (empr28_1.m + 2) - .5;
	fa = x[ka] * 20. - empr28_1.par * max(d__1,d__2) - d_sign(&
		empr28_1.par, &d__3);
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += (20. - empr28_1.par) * fa;
	if (j > 2) {
	    g[ka - empr28_1.m - empr28_1.m] += fa;
	}
	if (j > 1) {
	    if (i__ > 1) {
		g[ka - empr28_1.m - 1] += fa * 2.;
	    }
	    g[ka - empr28_1.m] -= fa * 8.;
	    if (i__ < empr28_1.m) {
		g[ka - empr28_1.m + 1] += fa * 2.;
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		g[ka - 2] += fa;
	    }
	    g[ka - 1] -= fa * 8.;
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa * 8.;
	    if (i__ < empr28_1.m - 1) {
		g[ka + 2] += fa;
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		g[ka + empr28_1.m - 1] += fa * 2.;
	    }
	    g[ka + empr28_1.m] -= fa * 8.;
	    if (i__ < empr28_1.m) {
		g[ka + empr28_1.m + 1] += fa * 2.;
	    }
	}
	if (j < empr28_1.m - 1) {
	    g[ka + empr28_1.m + empr28_1.m] += fa;
	}
/* L991: */
    }
    *f *= .5;
    return 0;
L800:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = .5 / (doublereal) (empr28_1.m + 2);
	j = (ka - 1) / empr28_1.m + 1;
	i__ = ka - (j - 1) * empr28_1.m;
	fa = x[ka] * 20.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	a4 = 0.;
	if (j > 2) {
	    fa += x[ka - empr28_1.m - empr28_1.m];
	    a4 += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		fa += x[ka - empr28_1.m - 1] * 2.;
		a3 += x[ka - empr28_1.m - 1];
		a4 += x[ka - empr28_1.m - 1];
	    }
	    fa -= x[ka - empr28_1.m] * 8.;
	    a1 -= x[ka - empr28_1.m];
	    a4 -= x[ka - empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka - empr28_1.m + 1] * 2.;
		a3 -= x[ka - empr28_1.m + 1];
		a4 += x[ka - empr28_1.m + 1];
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		fa += x[ka - 2];
		a3 += x[ka - 2];
	    }
	    fa -= x[ka - 1] * 8.;
	    a2 -= x[ka - 1];
	    a3 -= x[ka - 1] * 4.;
	}
	if (i__ < empr28_1.m) {
	    fa -= x[ka + 1] * 8.;
	    a2 += x[ka + 1];
	    a3 += x[ka + 1] * 4.;
	    if (i__ < empr28_1.m - 1) {
		fa += x[ka + 2];
		a3 -= x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		fa += x[ka + empr28_1.m - 1] * 2.;
		a3 += x[ka + empr28_1.m - 1];
		a4 -= x[ka + empr28_1.m - 1];
	    }
	    fa -= x[ka + empr28_1.m] * 8.;
	    a1 += x[ka + empr28_1.m];
	    a4 += x[ka + empr28_1.m] * 4.;
	    if (i__ < empr28_1.m) {
		fa += x[ka + empr28_1.m + 1] * 2.;
		a3 -= x[ka + empr28_1.m + 1];
		a4 -= x[ka + empr28_1.m + 1];
	    }
	}
	if (j < empr28_1.m - 1) {
	    fa += x[ka + empr28_1.m + empr28_1.m];
	    a4 -= x[ka + empr28_1.m + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    if (i__ > 1) {
		fa = fa - h__ - h__;
		a3 -= h__;
		a4 += h__;
	    }
	    fa += h__ * 8.;
	    a1 -= h__;
	    a4 -= h__ * 4.;
	    if (i__ < empr28_1.m) {
		fa -= h__ * 2.;
		a3 += h__;
		a4 += h__;
	    }
	    fa += h__;
	    a4 -= h__;
	}
	if (j == empr28_1.m - 1) {
	    fa -= h__;
	    a4 += h__;
	}
	fa += empr28_1.par * .25 * (a1 * a3 - a2 * a4);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	g[ka] += fa * 20.;
	a1 = 0.;
	a2 = 0.;
	a3 = 0.;
	a4 = 0.;
	ga1[0] = 0.;
	ga1[1] = 0.;
	ga2[0] = 0.;
	ga2[1] = 0.;
	for (k = 1; k <= 6; ++k) {
	    ga3[k - 1] = 0.;
	    ga4[k - 1] = 0.;
/* L801: */
	}
	if (j > 2) {
	    g[ka - empr28_1.m - empr28_1.m] += fa;
	    ga4[0] += 1.;
	    a4 += x[ka - empr28_1.m - empr28_1.m];
	}
	if (j > 1) {
	    if (i__ > 1) {
		g[ka - empr28_1.m - 1] += fa * 2.;
		ga3[0] += 1.;
		ga4[1] += 1.;
		a3 += x[ka - empr28_1.m - 1];
		a4 += x[ka - empr28_1.m - 1];
	    }
	    g[ka - empr28_1.m] -= fa * 8.;
	    ga1[0] += -1.;
	    a1 -= x[ka - empr28_1.m];
	    if (i__ < empr28_1.m) {
		g[ka - empr28_1.m + 1] += fa * 2.;
		ga3[1] += -1.;
		ga4[2] += 1.;
		a3 -= x[ka - empr28_1.m + 1];
		a4 += x[ka - empr28_1.m + 1];
	    }
	}
	if (i__ > 1) {
	    if (i__ > 2) {
		g[ka - 2] += fa;
		ga3[2] += 1.;
		a3 += x[ka - 2];
	    }
	    g[ka - 1] -= fa * 8.;
	    ga2[0] += -1.;
	    a2 -= x[ka - 1];
	}
	if (i__ < empr28_1.m) {
	    g[ka + 1] -= fa * 8.;
	    ga2[1] += 1.;
	    a2 += x[ka + 1];
	    if (i__ < empr28_1.m - 1) {
		g[ka + 2] += fa;
		ga3[3] += -1.;
		a3 -= x[ka + 2];
	    }
	}
	if (j < empr28_1.m) {
	    if (i__ > 1) {
		g[ka + empr28_1.m - 1] += fa * 2.;
		ga3[4] += 1.;
		ga4[3] += -1.;
		a3 += x[ka + empr28_1.m - 1];
		a4 -= x[ka + empr28_1.m - 1];
	    }
	    g[ka + empr28_1.m] -= fa * 8.;
	    ga1[1] += 1.;
	    a1 += x[ka + empr28_1.m];
	    if (i__ < empr28_1.m) {
		g[ka + empr28_1.m + 1] += fa * 2.;
		ga3[5] += -1.;
		ga4[4] += -1.;
		a3 -= x[ka + empr28_1.m + 1];
		a4 -= x[ka + empr28_1.m + 1];
	    }
	}
	if (j < empr28_1.m - 1) {
	    g[ka + empr28_1.m + empr28_1.m] += fa;
	    ga4[5] += -1.;
	    a4 -= x[ka + empr28_1.m + empr28_1.m];
	}
	if (j == empr28_1.m) {
	    if (i__ > 1) {
		a3 -= h__;
		a4 += h__;
	    }
	    a1 -= h__;
	    if (i__ < empr28_1.m) {
		a3 += h__;
		a4 += h__;
	    }
	    a4 -= h__;
	}
	if (j == empr28_1.m - 1) {
	    a4 += h__;
	}
	if (ka > empr28_1.m + empr28_1.m) {
	    g[ka - empr28_1.m - empr28_1.m] += empr28_1.par * .25 * (-a2 * 
		    ga4[0]) * fa;
	}
	if (ka > empr28_1.m + 1) {
	    g[ka - empr28_1.m - 1] += empr28_1.par * .25 * (a1 * ga3[0] - a2 *
		     ga4[1]) * fa;
	}
	if (ka > empr28_1.m) {
	    g[ka - empr28_1.m] += empr28_1.par * .25 * (ga1[0] * a3) * fa;
	}
	if (ka > empr28_1.m - 1) {
	    g[ka - empr28_1.m + 1] += empr28_1.par * .25 * (a1 * ga3[1] - a2 *
		     ga4[2]) * fa;
	}
	if (ka > 2) {
	    g[ka - 2] += empr28_1.par * .25 * (a1 * ga3[2]) * fa;
	}
	if (ka > 1) {
	    g[ka - 1] += empr28_1.par * .25 * (-ga2[0] * a4) * fa;
	}
	if (ka <= *n - 1) {
	    g[ka + 1] += empr28_1.par * .25 * (-ga2[1] * a4) * fa;
	}
	if (ka <= *n - 2) {
	    g[ka + 2] += empr28_1.par * .25 * (a1 * ga3[3]) * fa;
	}
	if (ka <= *n - empr28_1.m + 1) {
	    g[ka + empr28_1.m - 1] += empr28_1.par * .25 * (a1 * ga3[4] - a2 *
		     ga4[3]) * fa;
	}
	if (ka <= *n - empr28_1.m) {
	    g[ka + empr28_1.m] += empr28_1.par * .25 * (ga1[1] * a3) * fa;
	}
	if (ka <= *n - empr28_1.m - 1) {
	    g[ka + empr28_1.m + 1] += empr28_1.par * .25 * (a1 * ga3[5] - a2 *
		     ga4[4]) * fa;
	}
	if (ka <= *n - empr28_1.m - empr28_1.m) {
	    g[ka + empr28_1.m + empr28_1.m] += empr28_1.par * .25 * (-a2 * 
		    ga4[5]) * fa;
	}
/* L802: */
    }
    *f *= .5;
    return 0;
L240:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	w = 0.;
	i__4 = *n - 1;
	for (i__ = 1; i__ <= i__4; ++i__) {
	    w += (doublereal) ka / (doublereal) (ka + i__) * x[i__];
/* L241: */
	}
	fa = x[ka] - (.4 / (doublereal) (*n) * x[ka] * (w + .5 + (doublereal) 
		ka / (doublereal) (ka + *n) * .5 * x[*n]) + 1.);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	w = w + .5 + (doublereal) ka * .5 / (doublereal) (ka + *n) * x[*n];
	i__4 = *n - 1;
	for (i__ = 1; i__ <= i__4; ++i__) {
	    g[i__] -= .4 / (doublereal) (*n) * x[ka] * (doublereal) ka / (
		    doublereal) (ka + i__) * fa;
/* L242: */
	}
	g[*n] -= .2 / (doublereal) (*n) * x[ka] * (doublereal) ka / (
		doublereal) (ka + *n) * fa;
	g[ka] += (1. - w * .4 / (doublereal) (*n)) * fa;
/* L243: */
    }
    *f *= .5;
    return 0;
L410:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = 1. - x[1];
	    g[1] -= fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka] - x[ka - 1];
	    fa = (doublereal) (ka - 1) * 10. * (d__1 * d__1);
	    g[ka] += (doublereal) (ka - 1) * 20. * (x[ka] - x[ka - 1]) * fa;
	    g[ka - 1] -= (doublereal) (ka - 1) * 20. * (x[ka] - x[ka - 1]) * 
		    fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L411: */
    }
    *f *= .5;
    return 0;
L420:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = x[ka] - d__1 * d__1 * .1;
	    g[1] -= x[1] * .2 * fa;
	    g[*n] += fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka + 1];
	    fa = x[ka] - d__1 * d__1 * .1;
	    g[ka] += fa;
	    g[ka + 1] -= x[ka + 1] * .2 * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L421: */
    }
    *f *= .5;
    return 0;
L650:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = 0.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
/* Computing 3rd power */
	    d__1 = x[j];
	    s += d__1 * (d__1 * d__1);
/* L651: */
	}
	fa = x[ka] - 1. / (doublereal) (*n << 1) * (s + (doublereal) ka);
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j == ka) {
/* Computing 2nd power */
		d__1 = x[j];
		g[j] += (1. - d__1 * d__1 * 3. / ((doublereal) (*n) * 2.)) * 
			fa;
	    } else {
/* Computing 2nd power */
		d__1 = x[j];
		g[j] -= d__1 * d__1 * 3. / ((doublereal) (*n) * 2.) * fa;
	    }
/* L652: */
	}
/* L653: */
    }
    *f *= .5;
    return 0;
L660:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1 * exp(x[ka]);
	if (*n == 1) {
	    fa = x[ka] * -2. - s;
	    g[ka] -= (s + 2.) * fa;
	} else if (ka == 1) {
	    fa = x[ka] * -2. + x[ka + 1] - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka + 1] += fa;
	} else if (ka == *n) {
	    fa = x[ka - 1] - x[ka] * 2. - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka - 1] += fa;
	} else {
	    fa = x[ka - 1] - x[ka] * 2. + x[ka + 1] - s;
	    g[ka] -= (s + 2.) * fa;
	    g[ka - 1] += fa;
	    g[ka + 1] += fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L661: */
    }
    *f *= .5;
    return 0;
L670:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .1;
	if (*n == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	} else if (ka == 1) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka + 1] * 2.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka + 1] -= fa * 2.;
	} else if (ka == *n) {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1];
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka - 1] -= fa;
	} else {
	    fa = (3. - s * x[ka]) * x[ka] + 1. - x[ka - 1] - x[ka + 1] * 2.;
	    g[ka] += (3. - s * 2. * x[ka]) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa * 2.;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L671: */
    }
    *f *= .5;
    return 0;
L680:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s1 = 1.;
	s2 = 1.;
	s3 = 1.;
	j1 = 3;
	j2 = 3;
	if (ka - j1 > 1) {
	    i1 = ka - j1;
	} else {
	    i1 = 1;
	}
	if (ka + j2 < *n) {
	    i2 = ka + j2;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		s = s + x[j] + d__1 * d__1;
	    }
/* L681: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = (s1 + s2 * (d__1 * d__1)) * x[ka] + 1. - s3 * s;
/* Computing 2nd power */
	d__1 = x[ka];
	g[ka] += (s1 + s2 * 3. * (d__1 * d__1)) * fa;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    g[j] -= s3 * (x[j] * 2. + 1.) * fa;
/* L682: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L683: */
    }
    *f *= .5;
    return 0;
L690:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
	    fa = d__1 * d__1 - 1.;
	    g[1] += x[1] * 2. * fa;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - 1];
	    fa = d__1 * d__1 + log(x[ka]) - 1.;
	    g[ka - 1] += x[ka - 1] * 2. * fa;
	    g[ka] += 1. / x[ka] * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L691: */
    }
    *f *= .5;
    return 0;
L340:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka == 1) {
	    fa = x[1];
	    g[1] += fa;
	} else {
	    fa = cos(x[ka - 1]) + x[ka] - 1.;
	    g[ka] += fa;
	    g[ka - 1] -= sin(x[ka - 1]) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L341: */
    }
    *f *= .5;
    return 0;
L360:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
/* Computing 2nd power */
	d__1 = 1. / (doublereal) (*n + 1);
	s = d__1 * d__1;
	if (*n == 1) {
	    fa = x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	} else if (ka == 1) {
	    fa = x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka == *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - 1. + s * (x[ka] + sin(x[ka]));
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka - 1] -= fa;
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + s * (x[ka] + sin(x[ka])
		    );
	    g[ka] += (s * (cos(x[ka]) + 1.) + 2.) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L361: */
    }
    *f *= .5;
    return 0;
L380:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	if (ka - 5 > 1) {
	    i1 = ka - 5;
	} else {
	    i1 = 1;
	}
	if (ka + 1 < *n) {
	    i2 = ka + 1;
	} else {
	    i2 = *n;
	}
	s = 0.;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
		s += x[j] * (x[j] + 1.);
	    }
/* L381: */
	}
/* Computing 2nd power */
	d__1 = x[ka];
	fa = x[ka] * (d__1 * d__1 * 5. + 2.) + 1. - s;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing 2nd power */
	d__1 = x[ka];
	g[ka] += (d__1 * d__1 * 15. + 2.) * fa;
	i__4 = i2;
	for (j = i1; j <= i__4; ++j) {
	    if (j != ka) {
		g[j] -= (x[j] * 2. + 1.) * fa;
	    }
/* L382: */
	}
/* L383: */
    }
    *f *= .5;
    return 0;
L430:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	alf = 5.;
	bet = 14.;
	gam = 3.;
	d__1 = (doublereal) ka - (doublereal) (*n) / 2.;
	fa = bet * *n * x[ka] + pow_dd(&d__1, &gam);
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) ka / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		fa += t * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf));
	    }
/* L431: */
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (j != ka) {
/* Computing 2nd power */
		d__1 = x[j];
		t = sqrt(d__1 * d__1 + (doublereal) ka / (doublereal) j);
		s1 = log(t);
		d__1 = sin(s1);
		d__2 = cos(s1);
		d__3 = sin(s1);
		d__4 = alf - 1;
		d__5 = cos(s1);
		d__6 = alf - 1;
		g[j] += x[j] * (pow_dd(&d__1, &alf) + pow_dd(&d__2, &alf) + 
			alf * pow_dd(&d__3, &d__4) * cos(s1) - alf * sin(s1) *
			 pow_dd(&d__5, &d__6)) / t * fa;
	    } else {
		g[j] += bet * *n * fa;
	    }
/* L432: */
	}
/* L433: */
    }
    *f *= .5;
    return 0;
L440:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	c__ = .5;
	h__ = 1. / (doublereal) (*n);
	fa = 1. - c__ * h__ / 4.;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    s = c__ * h__ * (doublereal) ka / (doublereal) (ka + j << 1);
	    if (j == *n) {
		s /= 2.;
	    }
	    fa -= s * x[j];
/* L441: */
	}
	fa = x[ka] * fa - 1.;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    sx[j - 1] = c__ * h__ * (doublereal) ka / (doublereal) (ka + j << 
		    1);
/* L442: */
	}
	sx[*n - 1] *= .5;
	i__4 = *n;
	for (j = 1; j <= i__4; ++j) {
	    if (ka != j) {
		g[j] -= sx[j - 1] * x[ka] * fa;
	    } else {
		t = 1. - c__ * h__ / 4.;
		i__2 = *n;
		for (l = 1; l <= i__2; ++l) {
		    if (l == ka) {
			t -= sx[ka - 1] * 2. * x[ka];
		    } else {
			t -= sx[l - 1] * x[l];
		    }
/* L443: */
		}
		g[j] += t * fa;
	    }
/* L444: */
	}
/* L445: */
    }
    *f *= .5;
    return 0;
L270:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 * 2.;
	if (ka == 1) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - h__ * x[ka + 1];
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka + 1] -= (h__ + 1.) * fa;
	} else if (1 < ka && ka < *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] - t * (d__1 * d__1) - 
		    h__ * (x[ka + 1] - x[ka - 1]);
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka - 1] -= (1. - h__) * fa;
	    g[ka + 1] -= (h__ + 1.) * fa;
	} else if (ka == *n) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = -x[ka - 1] + x[ka] * 2. - .5 - t * (d__1 * d__1) - h__ * (.5 
		    - x[ka - 1]);
	    g[ka] += (1. - t * x[ka]) * 2. * fa;
	    g[ka - 1] -= (1. - h__) * fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L271: */
    }
    *f *= .5;
    return 0;
L280:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	s = .5;
	h__ = 1. / (doublereal) (*n + 1);
/* Computing 2nd power */
	d__1 = h__;
	t = d__1 * d__1 / s;
	t1 = h__ * 2.;
	al = 0.;
	be = .5;
	s1 = 0.;
	i__4 = ka;
	for (j = 1; j <= i__4; ++j) {
	    if (j == 1) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - al) / t1);
	    }
	    if (1 < j && j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (x[j + 1] - x[j - 1]) / 
			t1);
	    }
	    if (j == *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s1 += (doublereal) j * (d__1 * d__1 + (be - x[j - 1]) / t1);
	    }
/* L281: */
	}
	s1 = (1. - (doublereal) ka * h__) * s1;
	if (ka == *n) {
	    goto L283;
	}
	s2 = 0.;
	i__4 = *n;
	for (j = ka + 1; j <= i__4; ++j) {
	    if (j < *n) {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (x[j + 1] 
			- x[j - 1]) / t1);
	    } else {
/* Computing 2nd power */
		d__1 = x[j];
		s2 += (1. - (doublereal) j * h__) * (d__1 * d__1 + (be - x[j 
			- 1]) / t1);
	    }
/* L282: */
	}
	s1 += (doublereal) ka * s2;
L283:
	fa = x[ka] - (doublereal) ka * .5 * h__ - t * s1;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing 2nd power */
	d__1 = h__;
	s1 = d__1 * d__1 / s;
	s2 = 1. - (doublereal) ka * h__;
	i__4 = ka;
	for (j = 1; j <= i__4; ++j) {
	    sx[j - 1] = (doublereal) j * s2;
/* L284: */
	}
	if (ka == *n) {
	    goto L286;
	}
	i__4 = *n;
	for (j = ka + 1; j <= i__4; ++j) {
	    sx[j - 1] = (doublereal) ka * (1. - (doublereal) j * h__);
/* L285: */
	}
L286:
	g[1] -= s1 * (sx[0] * 2. * x[1] - sx[1] / t1) * fa;
	g[*n] -= s1 * (sx[*n - 2] / t1 + sx[*n - 1] * 2. * x[*n]) * fa;
	i__4 = *n - 1;
	for (j = 2; j <= i__4; ++j) {
	    g[j] -= s1 * ((sx[j - 2] - sx[j]) / t1 + sx[j - 1] * 2. * x[j]) * 
		    fa;
/* L287: */
	}
	g[ka] += fa;
	return 0;
/* L288: */
    }
    *f *= .5;
    return 0;
L290:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	a = -.009;
	b = .001;
	al = 0.;
	be = 25.;
	ga = 20.;
	ca = .3;
	cb = .3;
	h__ = (b - a) / (doublereal) (*n + 1);
	t = a + (doublereal) ka * h__;
/* Computing 2nd power */
	d__1 = h__;
	h__ = d__1 * d__1;
	s = (doublereal) ka / (doublereal) (*n + 1);
	u = al * (1. - s) + be * s + x[ka];
	ff = cb * exp(ga * (u - be)) - ca * exp(ga * (al - u));
	fg = ga * (cb * exp(ga * (u - be)) + ca * exp(ga * (al - u)));
	if (t <= 0.) {
	    ff += ca;
	} else {
	    ff -= cb;
	}
	if (*n == 1) {
	    fa = -al + x[ka] * 2. - be + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	} else if (ka == 1) {
	    fa = -al + x[ka] * 2. - x[ka + 1] + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka < *n) {
	    fa = -x[ka - 1] + x[ka] * 2. - x[ka + 1] + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	} else {
	    fa = -x[ka - 1] + x[ka] * 2. - be + h__ * ff;
	    g[ka] += (h__ * fg + 2.) * fa;
	    g[ka - 1] -= fa;
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* L291: */
    }
    *f *= .5;
    return 0;
L300:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	al1 = 0.;
	al2 = 0.;
	be1 = 0.;
	be2 = 0.;
	n1 = *n / 2;
	h__ = 1. / (doublereal) (n1 + 1);
	t = (doublereal) ka * h__;
	if (ka == 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al1;
	} else if (ka == n1 + 1) {
	    s1 = x[ka] * 2. - x[ka + 1];
	    b = al2;
	} else if (ka == n1) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be1;
	} else if (ka == *n) {
	    s1 = -x[ka - 1] + x[ka] * 2.;
	    b = be2;
	} else {
	    s1 = -x[ka - 1] + x[ka] * 2. - x[ka + 1];
	    b = 0.;
	}
	if (ka <= n1) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = x[n1 + ka];
	    s2 = d__1 * d__1 + x[ka] + d__2 * d__2 * .1 - 1.2;
	} else {
/* Computing 2nd power */
	    d__1 = x[ka - n1];
/* Computing 2nd power */
	    d__2 = x[ka];
	    s2 = d__1 * d__1 * .2 + d__2 * d__2 + x[ka] * 2. - .6;
	}
/* Computing 2nd power */
	d__1 = h__;
	fa = s1 + d__1 * d__1 * s2 - b;
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
/* Computing 2nd power */
	d__1 = (doublereal) (n1 + 1);
	h__ = 1. / (d__1 * d__1);
	if (ka == 1) {
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else if (ka == n1 + 1) {
	    g[1] += h__ * .4 * x[1] * fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	    g[ka + 1] -= fa;
	} else if (ka == n1) {
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else if (ka == *n) {
	    g[n1] += h__ * .4 * x[n1] * fa;
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	} else if (ka < n1) {
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 1.) + 2.) * fa;
	    g[ka + 1] -= fa;
	    g[n1 + ka] += h__ * .2 * x[n1 + ka] * fa;
	} else {
	    g[ka - n1] += h__ * .4 * x[ka - n1] * fa;
	    g[ka - 1] -= fa;
	    g[ka] += (h__ * (x[ka] * 2. + 2.) + 2.) * fa;
	    g[ka + 1] -= fa;
	}
/* L301: */
    }
    *f *= .5;
    return 0;
L710:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	la = 1;
	h__ = 1. / (doublereal) (nd + 1);
	h2 = la * h__ * h__;
	if (l == 1 && k == 1) {
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * exp(x[1]);
	}
	if (1 < l && l < nd && k == 1) {
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * exp(x[l]);
	}
	if (l == nd && k == 1) {
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * exp(x[nd]);
	}
	if (l == 1 && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * exp(
		    x[ka]);
	}
	if (l == 1 && k == nd) {
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * exp(x[
		    ka]);
	}
	if (l == nd && k == nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * exp(x[ka]);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * exp(x[ka]);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	if (l == 1 && k == 1) {
	    g[1] += (h2 * exp(x[1]) + 4.) * fa;
	    g[2] -= fa;
	    g[nd + 1] -= fa;
	}
	if (1 < l && l < nd && k == 1) {
	    g[l] += (h2 * exp(x[l]) + 4.) * fa;
	    g[l - 1] -= fa;
	    g[l + 1] -= fa;
	    g[l + nd] -= fa;
	}
	if (l == nd && k == 1) {
	    g[nd] += (h2 * exp(x[nd]) + 4.) * fa;
	    g[nd - 1] -= fa;
	    g[nd + nd] -= fa;
	}
	if (l == 1 && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == nd && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == 1 && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	}
	if (1 < l && l < nd && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
	if (l == nd && k == nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    g[ka] += (h2 * exp(x[ka]) + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
/* L711: */
    }
    *f *= .5;
    return 0;
L820:
    i__1 = empr28_1.na;
    for (ka = 1; ka <= i__1; ++ka) {
	nd = (integer) sqrt((doublereal) (*n));
	l = ka % nd;
	if (l == 0) {
	    k = ka / nd;
	    l = nd;
	} else {
	    k = ka / nd + 1;
	}
	h__ = 1. / (doublereal) (nd + 1);
	h2 = h__ * h__;
	if (l == 1 && k == 1) {
/* Computing 2nd power */
	    d__1 = x[1];
/* Computing 2nd power */
	    d__2 = h__ + 1.;
	    fa = x[1] * 4. - x[2] - x[nd + 1] + h2 * (d__1 * d__1) - 24. / (
		    d__2 * d__2);
	}
	if (1 < l && l < nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[l];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 1.;
	    fa = x[l] * 4. - x[l - 1] - x[l + 1] - x[l + nd] + h2 * (d__1 * 
		    d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == 1) {
/* Computing 2nd power */
	    d__1 = x[nd];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[nd] * 4. - x[nd - 1] - x[nd + nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (l == 1 && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 1.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) k * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - nd] - x[ka - 1] - x[ka + nd] + h2 * (
		    d__1 * d__1) - 12. / (d__2 * d__2);
	}
	if (l == 1 && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 1.;
/* Computing 2nd power */
	    d__3 = h__ + 2.;
	    fa = x[ka] * 4. - x[ka + 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    12. / (d__2 * d__2) - 12. / (d__3 * d__3);
	}
	if (1 < l && l < nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) l * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] + h2 * (d__1 
		    * d__1) - 12. / (d__2 * d__2);
	}
	if (l == nd && k == nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
/* Computing 2nd power */
	    d__2 = (doublereal) nd * h__ + 2.;
	    fa = x[ka] * 4. - x[ka - 1] - x[ka - nd] + h2 * (d__1 * d__1) - 
		    24. / (d__2 * d__2);
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
/* Computing 2nd power */
	    d__1 = x[ka];
	    fa = x[ka] * 4. - x[ka - 1] - x[ka + 1] - x[ka - nd] - x[ka + nd] 
		    + h2 * (d__1 * d__1);
	}
/* Computing 2nd power */
	d__1 = fa;
	*f += d__1 * d__1;
	if (l == 1 && k == 1) {
	    g[1] += (h2 * x[1] * 2. + 4.) * fa;
	    g[2] -= fa;
	    g[nd + 1] -= fa;
	}
	if (1 < l && l < nd && k == 1) {
	    g[l] += (h2 * x[l] * 2. + 4.) * fa;
	    g[l - 1] -= fa;
	    g[l + 1] -= fa;
	    g[l + nd] -= fa;
	}
	if (l == nd && k == 1) {
	    g[nd] += (h2 * x[nd] * 2. + 4.) * fa;
	    g[nd - 1] -= fa;
	    g[nd + nd] -= fa;
	}
	if (l == 1 && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == nd && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + nd] -= fa;
	}
	if (l == 1 && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka + 1] -= fa;
	}
	if (1 < l && l < nd && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	}
	if (l == nd && k == nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	}
	if (1 < l && l < nd && 1 < k && k < nd) {
	    g[ka] += (h2 * x[ka] * 2. + 4.) * fa;
	    g[ka - nd] -= fa;
	    g[ka - 1] -= fa;
	    g[ka + 1] -= fa;
	    g[ka + nd] -= fa;
	}
/* L821: */
    }
    *f *= .5;
    return 0;
} /* tfbu28_ */

