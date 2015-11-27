/* lvumm.f -- translated by f2c (version 20100827).
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
    doublereal y[123];
} empr06_;

#define empr06_1 empr06_

/* Table of constant values */

static integer c__2 = 2;
static doublecomplex c_b168 = {2.,0.};
static integer c__3 = 3;

/* SUBROUTINE TIUD06             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 90/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  INITIATION OF VARIABLES FOR NONLINEAR MINIMAX APPROXIMATION. */
/*  UNCONSTRAINED DENSE VERSION. */

/* PARAMETERS : */
/*  IO  N  NUMBER OF VARIABLES. */
/*  IO  NA  NUMBER OF PARTIAL FUNCTIONS. */
/*  RO  X(N)  VECTOR OF VARIABLES. */
/*  RO  FMIN  LOWER BOUND FOR VALUE OF THE OBJECTIVE FUNCTION. */
/*  RO  XMAX  MAXIMUM STEPSIZE. */
/*  IO  NEXT  NUMBER OF THE TEST PROBLEM. */
/*  IO  IEXT  TYPE OF OBJECTIVE FUNCTION. IEXT<0-MAXIMUM OF VALUES. */
/*         IEXT=0-MAXIMUM OF ABSOLUTE VALUES. */
/*  IO  IERR  ERROR INDICATOR. */

/* Subroutine */ int tiud06_(integer *n, integer *na, doublereal *x, 
	doublereal *fmin, doublereal *xmax, integer *next, integer *iext, 
	integer *ierr)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    double exp(doublereal), sin(doublereal), cos(doublereal), sqrt(doublereal)
	    , atan(doublereal);

    /* Local variables */
    static integer i__;
    static doublereal t;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    *fmin = -1e60;
    *xmax = 1e3;
    *iext = -1;
    *ierr = 0;
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L190;
	case 3:  goto L180;
	case 4:  goto L80;
	case 5:  goto L20;
	case 6:  goto L170;
	case 7:  goto L120;
	case 8:  goto L90;
	case 9:  goto L230;
	case 10:  goto L210;
	case 11:  goto L140;
	case 12:  goto L150;
	case 13:  goto L200;
	case 14:  goto L30;
	case 15:  goto L130;
	case 16:  goto L100;
	case 17:  goto L40;
	case 18:  goto L110;
	case 19:  goto L50;
	case 20:  goto L60;
	case 21:  goto L70;
	case 22:  goto L220;
	case 23:  goto L160;
	case 24:  goto L240;
	case 25:  goto L250;
    }
L10:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	x[1] = 2.;
	x[2] = 2.;
    } else {
	*ierr = 1;
    }
    return 0;
L190:
    if (*n >= 2 && *na >= 3) {
	*n = 2;
	*na = 3;
	x[1] = 3.;
	x[2] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L180:
    if (*n >= 2 && *na >= 2) {
	*n = 2;
	*na = 2;
	x[1] = 1.41831;
	x[2] = -4.79462;
    } else {
	*ierr = 1;
    }
    return 0;
L80:
    if (*n >= 3 && *na >= 6) {
	*n = 3;
	*na = 6;
	x[1] = 1.;
	x[2] = 1.;
	x[3] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L20:
    if (*n >= 4 && *na >= 4) {
	*n = 4;
	*na = 4;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 0.;
/* L21: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L170:
    if (*n >= 4 && *na >= 4) {
	*n = 4;
	*na = 4;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 0.;
/* L171: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L120:
    if (*n >= 3 && *na >= 21) {
	*n = 3;
	*na = 21;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L121: */
	}
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    t = (doublereal) (i__ - 1) * 10. / (doublereal) (*na - 1);
	    empr06_1.y[i__ - 1] = t;
	    empr06_1.y[*na + i__ - 1] = exp(-t) * .14999999999999999 + exp(t *
		     -5.) * .019230769230769232 - exp(t * -2.) * 
		    .015384615384615385 * (sin(t * 2.) * 3. + cos(t * 2.) * 
		    11.);
/* L122: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L90:
    if (*n >= 3 && *na >= 15) {
	*n = 3;
	*na = 15;
	x[1] = 1.;
	x[2] = 1.;
	x[3] = 1.;
	empr06_1.y[0] = .14;
	empr06_1.y[1] = .18;
	empr06_1.y[2] = .22;
	empr06_1.y[3] = .25;
	empr06_1.y[4] = .29;
	empr06_1.y[5] = .32;
	empr06_1.y[6] = .35;
	empr06_1.y[7] = .39;
	empr06_1.y[8] = .37;
	empr06_1.y[9] = .58;
	empr06_1.y[10] = .73;
	empr06_1.y[11] = .96;
	empr06_1.y[12] = 1.34;
	empr06_1.y[13] = 2.1;
	empr06_1.y[14] = 4.39;
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L230:
    if (*n >= 4 && *na >= 11) {
	*n = 4;
	*na = 11;
	x[1] = .25;
	x[2] = .39;
	x[3] = .415;
	x[4] = .39;
	empr06_1.y[0] = .1957;
	empr06_1.y[1] = .1947;
	empr06_1.y[2] = .1735;
	empr06_1.y[3] = .16;
	empr06_1.y[4] = .0844;
	empr06_1.y[5] = .0627;
	empr06_1.y[6] = .0456;
	empr06_1.y[7] = .0342;
	empr06_1.y[8] = .0323;
	empr06_1.y[9] = .0235;
	empr06_1.y[10] = .0246;
	empr06_1.y[11] = 4.;
	empr06_1.y[12] = 2.;
	empr06_1.y[13] = 1.;
	empr06_1.y[14] = .5;
	empr06_1.y[15] = .25;
	empr06_1.y[16] = .167;
	empr06_1.y[17] = .125;
	empr06_1.y[18] = .1;
	empr06_1.y[19] = .0833;
	empr06_1.y[20] = .0714;
	empr06_1.y[21] = .0625;
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L210:
    if (*n >= 4 && *na >= 20) {
	*n = 4;
	*na = 20;
	x[1] = 25.;
	x[2] = 5.;
	x[3] = -5.;
	x[4] = -1.;
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L140:
    if (*n >= 4 && *na >= 21) {
	*n = 4;
	*na = 21;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L141: */
	}
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    empr06_1.y[i__ - 1] = (doublereal) (i__ - 1) * .75 / (doublereal) 
		    (*na - 1) + .25;
	    empr06_1.y[*na + i__ - 1] = sqrt(empr06_1.y[i__ - 1]);
/* L142: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L150:
    if (*n >= 4 && *na >= 21) {
	*n = 4;
	*na = 21;
	x[1] = 1.;
	x[2] = 1.;
	x[3] = -3.;
	x[4] = -1.;
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    empr06_1.y[i__ - 1] = (doublereal) (i__ - 1) / (doublereal) (*na 
		    - 1) - .5;
	    empr06_1.y[*na + i__ - 1] = 1. / (empr06_1.y[i__ - 1] + 1.);
/* L151: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L200:
    if (*n >= 4 && *na >= 61) {
	*n = 4;
	*na = 61;
	*iext = 0;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L201: */
	}
	x[3] = 10.;
	empr06_1.y[0] = 1.;
	empr06_1.y[1] = 1.01;
	empr06_1.y[2] = 1.02;
	empr06_1.y[3] = 1.03;
	empr06_1.y[4] = 1.05;
	empr06_1.y[5] = 1.075;
	empr06_1.y[6] = 1.1;
	empr06_1.y[7] = 1.125;
	empr06_1.y[8] = 1.15;
	empr06_1.y[9] = 1.2;
	empr06_1.y[10] = 1.25;
	empr06_1.y[11] = 1.3;
	empr06_1.y[12] = 1.35;
	empr06_1.y[13] = 1.4;
	empr06_1.y[14] = 1.5;
	empr06_1.y[15] = 1.6;
	empr06_1.y[16] = 1.7;
	empr06_1.y[17] = 1.8;
	empr06_1.y[18] = 1.9;
	empr06_1.y[19] = 2.;
	empr06_1.y[20] = 2.1;
	empr06_1.y[21] = 2.2;
	empr06_1.y[22] = 2.3;
	empr06_1.y[23] = 2.5;
	empr06_1.y[24] = 2.75;
	empr06_1.y[25] = 3.;
	empr06_1.y[26] = 3.25;
	empr06_1.y[27] = 3.5;
	empr06_1.y[28] = 4.;
	empr06_1.y[29] = 4.5;
	empr06_1.y[30] = 5.;
	empr06_1.y[31] = 5.5;
	empr06_1.y[32] = 6.;
	empr06_1.y[33] = 6.5;
	empr06_1.y[34] = 7.;
	empr06_1.y[35] = 7.5;
	empr06_1.y[36] = 8.;
	empr06_1.y[37] = 8.5;
	empr06_1.y[38] = 9.;
	empr06_1.y[39] = 10.;
	empr06_1.y[40] = 11.;
	empr06_1.y[41] = 12.;
	empr06_1.y[42] = 13.;
	empr06_1.y[43] = 15.;
	empr06_1.y[44] = 17.5;
	empr06_1.y[45] = 20.;
	empr06_1.y[46] = 22.5;
	empr06_1.y[47] = 25.;
	empr06_1.y[48] = 30.;
	empr06_1.y[49] = 35.;
	empr06_1.y[50] = 40.;
	empr06_1.y[51] = 50.;
	empr06_1.y[52] = 60.;
	empr06_1.y[53] = 70.;
	empr06_1.y[54] = 80.;
	empr06_1.y[55] = 100.;
	empr06_1.y[56] = 150.;
	empr06_1.y[57] = 200.;
	empr06_1.y[58] = 300.;
	empr06_1.y[59] = 500.;
	empr06_1.y[60] = 1e5;
	empr06_1.y[61] = .97386702052733792831;
	empr06_1.y[62] = .97390711665677071911;
	empr06_1.y[63] = .97394794566286525039;
	empr06_1.y[64] = .97398947529386626621;
	empr06_1.y[65] = .97407451325974368215;
	empr06_1.y[66] = .97418422166965892644;
	empr06_1.y[67] = .97429732692565188272;
	empr06_1.y[68] = .97441344289222034304;
	empr06_1.y[69] = .97453221704823108216;
	empr06_1.y[70] = .97477647977277153145;
	empr06_1.y[71] = .97502785781178233026;
	empr06_1.y[72] = .97528446418205610067;
	empr06_1.y[73] = .97554472005909873148;
	empr06_1.y[74] = .97580730389916439626;
	empr06_1.y[75] = .97633521198091785788;
	empr06_1.y[76] = .97686134356195586299;
	empr06_1.y[77] = .97738094095418268249;
	empr06_1.y[78] = .97789073928751194169;
	empr06_1.y[79] = .97838854811088140808;
	empr06_1.y[80] = .97887295363155439576;
	empr06_1.y[81] = .97934310478576951385;
	empr06_1.y[82] = .97979855827226762515;
	empr06_1.y[83] = .98023916551033862691;
	empr06_1.y[84] = .98107624468416045728;
	empr06_1.y[85] = .98204290774765289406;
	empr06_1.y[86] = .98292719363632655668;
	empr06_1.y[87] = .98373656564197279264;
	empr06_1.y[88] = .98447846610682328991;
	empr06_1.y[89] = .98578713114264981186;
	empr06_1.y[90] = .98690124654380846379;
	empr06_1.y[91] = .9878587905485517338;
	empr06_1.y[92] = .98868928566806726978;
	empr06_1.y[93] = .98941568049711884384;
	empr06_1.y[94] = .99005592865089067038;
	empr06_1.y[95] = .99062420259214811899;
	empr06_1.y[96] = .9911318001873848773;
	empr06_1.y[97] = .99158781685339306121;
	empr06_1.y[98] = .99199964493176098231;
	empr06_1.y[99] = .99237334707422899195;
	empr06_1.y[100] = .99302559755582945576;
	empr06_1.y[101] = .99357562712206729735;
	empr06_1.y[102] = .994045600315813543;
	empr06_1.y[103] = .99445173790980305195;
	empr06_1.y[104] = .99511816085114882367;
	empr06_1.y[105] = .99575584307408838284;
	empr06_1.y[106] = .99624640327264396775;
	empr06_1.y[107] = .99663543022201287399;
	empr06_1.y[108] = .99695146031888813172;
	empr06_1.y[109] = .99743367936799001685;
	empr06_1.y[110] = .99778424120023198554;
	empr06_1.y[111] = .99805056960591223604;
	empr06_1.y[112] = .99842841443786596919;
	empr06_1.y[113] = .99868358857261655169;
	empr06_1.y[114] = .99886748198687248566;
	empr06_1.y[115] = .99900629944600342584;
	empr06_1.y[116] = .99920194660435455419;
	empr06_1.y[117] = .99946519560889341627;
	empr06_1.y[118] = .99959785208794891934;
	empr06_1.y[119] = .99973120214935885075;
	empr06_1.y[120] = .99983838442420395745;
	empr06_1.y[121] = .999999189398046846077;
    } else {
	*ierr = 1;
    }
    return 0;
L30:
    if (*n >= 5 && *na >= 21) {
	*n = 5;
	*na = 21;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 0.;
/* L31: */
	}
	x[1] = .5;
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L130:
    if (*n >= 5 && *na >= 30) {
	*n = 5;
	*na = 30;
	x[1] = 0.;
	x[2] = -1.;
	x[3] = 10.;
	x[4] = 1.;
	x[5] = 10.;
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    empr06_1.y[i__ - 1] = (doublereal) (i__ - 1) * 2. / (doublereal) (
		    *na - 1) - 1.;
	    t = empr06_1.y[i__ - 1] * 8.;
/* Computing 2nd power */
	    d__1 = t - 1.;
	    empr06_1.y[*na + i__ - 1] = sqrt(d__1 * d__1 + 1.) * atan(t) / t;
/* L132: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L100:
    if (*n >= 6 && *na >= 51) {
	*n = 6;
	*na = 51;
	x[1] = 2.;
	x[2] = 2.;
	x[3] = 7.;
	x[4] = 0.;
	x[5] = -2.;
	x[6] = 1.;
	i__1 = *na;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    t = (doublereal) (i__ - 1) * .1;
	    empr06_1.y[i__ - 1] = exp(-t) * .5 - exp(t * -2.) + exp(t * -3.) *
		     .5 + exp(t * -1.5) * 1.5 * sin(t * 7.) + exp(t * -2.5) * 
		    sin(t * 5.);
/* L101: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L40:
    if (*n >= 6 && *na >= 11) {
	*n = 6;
	*na = 11;
	x[1] = .8;
	x[2] = 1.5;
	x[3] = 1.2;
	x[4] = 3.;
	x[5] = .8;
	x[6] = 6.;
	empr06_1.y[0] = .5;
	empr06_1.y[1] = .6;
	empr06_1.y[2] = .7;
	empr06_1.y[3] = .77;
	empr06_1.y[4] = .9;
	empr06_1.y[5] = 1.;
	empr06_1.y[6] = 1.1;
	empr06_1.y[7] = 1.23;
	empr06_1.y[8] = 1.3;
	empr06_1.y[9] = 1.4;
	empr06_1.y[10] = 1.5;
    } else {
	*ierr = 1;
    }
    return 0;
L110:
    if (*n >= 9 && *na >= 41) {
	*n = 9;
	*na = 41;
	x[1] = 0.;
	x[2] = 1.;
	x[3] = 0.;
	x[4] = -.15;
	x[5] = 0.;
	x[6] = -.68;
	x[7] = 0.;
	x[8] = -.72;
	x[9] = .37;
	for (i__ = 1; i__ <= 6; ++i__) {
/* L111: */
	    empr06_1.y[i__ - 1] = (i__ - 1) * .01;
	}
	for (i__ = 7; i__ <= 20; ++i__) {
/* L112: */
	    empr06_1.y[i__ - 1] = (i__ - 7) * .03 + .07;
	}
	empr06_1.y[20] = .5;
	for (i__ = 22; i__ <= 35; ++i__) {
/* L113: */
	    empr06_1.y[i__ - 1] = (i__ - 22) * .03 + .54;
	}
	for (i__ = 36; i__ <= 41; ++i__) {
/* L114: */
	    empr06_1.y[i__ - 1] = (i__ - 36) * .01 + .95;
	}
	for (i__ = 1; i__ <= 41; ++i__) {
	    empr06_1.y[i__ + 40] = cos(empr06_1.y[i__ - 1] * 
		    3.14159265358979324);
/* L115: */
	    empr06_1.y[i__ + 81] = sin(empr06_1.y[i__ - 1] * 
		    3.14159265358979324);
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L50:
    if (*n >= 7 && *na >= 5) {
	*n = 7;
	*na = 5;
	x[1] = 1.;
	x[2] = 2.;
	x[3] = 0.;
	x[4] = 4.;
	x[5] = 0.;
	x[6] = 1.;
	x[7] = 1.;
    } else {
	*ierr = 1;
    }
    return 0;
L60:
    if (*n >= 10 && *na >= 9) {
	*n = 10;
	*na = 9;
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
    } else {
	*ierr = 1;
    }
    return 0;
L70:
    if (*n >= 20 && *na >= 18) {
	*n = 20;
	*na = 18;
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
    } else {
	*ierr = 1;
    }
    return 0;
L220:
    if (*n >= 10 && *na >= 2) {
	*n = 10;
	*na = 2;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = .1;
/* L221: */
	}
	x[1] = 100.;
    } else {
	*ierr = 1;
    }
    return 0;
L160:
    if (*n >= 11 && *na >= 10) {
	*n = 11;
	*na = 10;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 1.;
/* L161: */
	}
    } else {
	*ierr = 1;
    }
    return 0;
L240:
    if (*n >= 20 && *na >= 31) {
	*n = 20;
	*na = 31;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    x[i__] = 0.;
/* L241: */
	}
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
L250:
    if (*n >= 11 && *na >= 65) {
	*n = 11;
	*na = 65;
	x[1] = 1.3;
	x[2] = .65;
	x[3] = .65;
	x[4] = .7;
	x[5] = .6;
	x[6] = 3.;
	x[7] = 5.;
	x[8] = 7.;
	x[9] = 2.;
	x[10] = 4.5;
	x[11] = 5.5;
	empr06_1.y[0] = 1.366;
	empr06_1.y[1] = 1.191;
	empr06_1.y[2] = 1.112;
	empr06_1.y[3] = 1.013;
	empr06_1.y[4] = .991;
	empr06_1.y[5] = .885;
	empr06_1.y[6] = .831;
	empr06_1.y[7] = .847;
	empr06_1.y[8] = .786;
	empr06_1.y[9] = .725;
	empr06_1.y[10] = .746;
	empr06_1.y[11] = .679;
	empr06_1.y[12] = .608;
	empr06_1.y[13] = .655;
	empr06_1.y[14] = .616;
	empr06_1.y[15] = .606;
	empr06_1.y[16] = .602;
	empr06_1.y[17] = .626;
	empr06_1.y[18] = .651;
	empr06_1.y[19] = .724;
	empr06_1.y[20] = .649;
	empr06_1.y[21] = .649;
	empr06_1.y[22] = .694;
	empr06_1.y[23] = .644;
	empr06_1.y[24] = .624;
	empr06_1.y[25] = .661;
	empr06_1.y[26] = .612;
	empr06_1.y[27] = .558;
	empr06_1.y[28] = .553;
	empr06_1.y[29] = .495;
	empr06_1.y[30] = .5;
	empr06_1.y[31] = .423;
	empr06_1.y[32] = .395;
	empr06_1.y[33] = .375;
	empr06_1.y[34] = .372;
	empr06_1.y[35] = .391;
	empr06_1.y[36] = .396;
	empr06_1.y[37] = .405;
	empr06_1.y[38] = .428;
	empr06_1.y[39] = .429;
	empr06_1.y[40] = .523;
	empr06_1.y[41] = .562;
	empr06_1.y[42] = .607;
	empr06_1.y[43] = .653;
	empr06_1.y[44] = .672;
	empr06_1.y[45] = .708;
	empr06_1.y[46] = .633;
	empr06_1.y[47] = .668;
	empr06_1.y[48] = .645;
	empr06_1.y[49] = .632;
	empr06_1.y[50] = .591;
	empr06_1.y[51] = .559;
	empr06_1.y[52] = .597;
	empr06_1.y[53] = .625;
	empr06_1.y[54] = .739;
	empr06_1.y[55] = .71;
	empr06_1.y[56] = .729;
	empr06_1.y[57] = .72;
	empr06_1.y[58] = .636;
	empr06_1.y[59] = .581;
	empr06_1.y[60] = .428;
	empr06_1.y[61] = .292;
	empr06_1.y[62] = .162;
	empr06_1.y[63] = .098;
	empr06_1.y[64] = .054;
	*xmax = 10.;
	*iext = 0;
    } else {
	*ierr = 1;
    }
    return 0;
} /* tiud06_ */

/* SUBROUTINE TAFU06             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 91/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  VALUES OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  FA  VALUE OF THE PARTIAL FUNCTION AT THE */
/*          SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tafu06_(integer *n, integer *ka, doublereal *x, 
	doublereal *fa, integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, d__10, 
	    d__11, d__12, d__13, d__14, d__15, d__16, d__17, d__18, d__19, 
	    d__20;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    double exp(doublereal), sqrt(doublereal), cos(doublereal), sin(doublereal)
	    , pow_dd(doublereal *, doublereal *);
    void z_div(doublecomplex *, doublecomplex *, doublecomplex *);
    double z_abs(doublecomplex *), pow_di(doublereal *, integer *);

    /* Local variables */
    static integer i__, j;
    static doublereal t;
    static doublecomplex c1, c2, c3;
    static doublereal x1, x2, x3, x4, x5, x6, x7, x8;
    static doublecomplex ca[4], cb[4];
    static doublereal xa[3], xb[3], xc[3], beta;

    /* Parameter adjustments */
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L190;
	case 3:  goto L180;
	case 4:  goto L80;
	case 5:  goto L20;
	case 6:  goto L170;
	case 7:  goto L120;
	case 8:  goto L90;
	case 9:  goto L230;
	case 10:  goto L210;
	case 11:  goto L140;
	case 12:  goto L150;
	case 13:  goto L200;
	case 14:  goto L30;
	case 15:  goto L130;
	case 16:  goto L100;
	case 17:  goto L40;
	case 18:  goto L110;
	case 19:  goto L50;
	case 20:  goto L60;
	case 21:  goto L70;
	case 22:  goto L220;
	case 23:  goto L160;
	case 24:  goto L240;
	case 25:  goto L250;
    }
L10:
    x1 = x[1] * x[1];
    x2 = x[2] * x[2];
    x3 = x[1] + x[1];
    x4 = x[2] + x[2];
	
    switch (*ka) {
	case 1:  goto L11;
	case 2:  goto L12;
	case 3:  goto L13;
    }
L11:
	*fa = x1 + x2 * x2;
    return 0;
L12:
    *fa = 8. - (x[1] + x[2]) * 4. + x1 + x2;
    return 0;
L13:
    *fa = exp(x[2] - x[1]) * 2.;
    return 0;
L190:
    x1 = x[1] * 10. / (x[1] + .1);
/* Computing 2nd power */
    d__1 = x[2];
    x2 = d__1 * d__1 * 2.;
    if (*ka == 1) {
	*fa = (x[1] + x1 + x2) * .5;
    } else if (*ka == 2) {
	*fa = (-x[1] + x1 + x2) * .5;
    } else if (*ka == 3) {
	*fa = (x[1] - x1 + x2) * .5;
    }
    return 0;
L180:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    x1 = d__1 * d__1 + d__2 * d__2;
    x2 = sqrt(x1);
    if (*ka == 1) {
/* Computing 2nd power */
	d__1 = x[1] - x2 * cos(x2);
	*fa = d__1 * d__1 + x1 * .005;
    } else if (*ka == 2) {
/* Computing 2nd power */
	d__1 = x[2] - x2 * sin(x2);
	*fa = d__1 * d__1 + x1 * .005;
    }
    return 0;
L80:
    switch (*ka) {
	case 1:  goto L81;
	case 2:  goto L82;
	case 3:  goto L83;
	case 4:  goto L84;
	case 5:  goto L85;
	case 6:  goto L86;
    }
L81:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
    *fa = d__1 * d__1 + d__2 * d__2 + d__3 * d__3 - 1.;
    return 0;
L82:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3] - 2.;
    *fa = d__1 * d__1 + d__2 * d__2 + d__3 * d__3;
    return 0;
L83:
    *fa = x[1] + x[2] + x[3] - 1.;
    return 0;
L84:
    *fa = x[1] + x[2] - x[3] + 1.;
    return 0;
L85:
/* Computing 3rd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3] * 5. - x[1] + 1.;
    *fa = (d__1 * (d__1 * d__1) + d__2 * d__2 * 3. + d__3 * d__3) * 2.;
    return 0;
L86:
/* Computing 2nd power */
    d__1 = x[1];
    *fa = d__1 * d__1 - x[3] * 9.;
    return 0;
L20:
    x1 = x[1] * x[1];
    x2 = x[2] * x[2];
    x3 = x[3] * x[3];
    x4 = x[4] * x[4];
    x5 = x[1] + x[1];
    x6 = x[2] + x[2];
    x7 = x[3] + x[3];
    x8 = x[4] + x[4];
    *fa = x1 + x2 + x3 + x3 + x4 - (x[1] + x[2]) * 5. - x[3] * 21. + x[4] * 
	    7.;
/* L21: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L22;
	case 3:  goto L23;
	case 4:  goto L24;
    }
L22:
    *fa += (x1 + x2 + x3 + x4 + x[1] - x[2] + x[3] - x[4] - 8.) * 10.;
    return 0;
L23:
    *fa += (x1 + x2 + x2 + x3 + x4 + x4 - x[1] - x[4] - 10.) * 10.;
    return 0;
L24:
    *fa += (x1 + x2 + x3 + x5 - x[2] - x[4] - 5.) * 10.;
    return 0;
L170:
/* Computing 4th power */
    d__1 = x[4] + 1., d__1 *= d__1;
    x1 = x[1] - d__1 * d__1;
    x2 = x1 * x1;
    x3 = x[2] - x2 * x2;
    x4 = x3 * x3;
/* Computing 2nd power */
    d__1 = x[3];
/* Computing 2nd power */
    d__2 = x[4];
    *fa = x2 + x4 + d__1 * d__1 * 2. + d__2 * d__2 - (x1 + x3) * 5. - x[3] * 
	    21. + x[4] * 7.;
    switch (*ka) {
	case 1:  goto L171;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
    }
L171:
    return 0;
L172:
/* Computing 2nd power */
    d__1 = x[3];
/* Computing 2nd power */
    d__2 = x[4];
    *fa += (x2 + x4 + d__1 * d__1 + d__2 * d__2 + x1 - x3 + x[3] - x[4] - 8.) 
	    * 10.;
    return 0;
L173:
/* Computing 2nd power */
    d__1 = x[3];
/* Computing 2nd power */
    d__2 = x[4];
    *fa += (x2 + x4 * 2. + d__1 * d__1 + d__2 * d__2 * 2. - x1 - x[4] - 10.) *
	     10.;
    return 0;
L174:
/* Computing 2nd power */
    d__1 = x[3];
    *fa += (x2 + x4 + d__1 * d__1 + x1 * 2. - x3 - x[4] - 5.) * 10.;
    return 0;
L120:
    t = empr06_1.y[*ka - 1];
    *fa = x[3] / x[2] * exp(-x[1] * t) * sin(x[2] * t) - empr06_1.y[*ka + 20];
    return 0;
L90:
/* Computing MIN */
    i__1 = *ka, i__2 = 16 - *ka;
    *fa = empr06_1.y[*ka - 1] - x[1] - (doublereal) (*ka) / ((doublereal) (16 
	    - *ka) * x[2] + (doublereal) min(i__1,i__2) * x[3]);
    return 0;
L230:
    t = empr06_1.y[*ka + 10];
    *fa = empr06_1.y[*ka - 1] - x[1] * t * (t + x[2]) / ((t + x[3]) * t + x[4]
	    );
    return 0;
L210:
    t = (doublereal) (*ka) * .2;
/* Computing 2nd power */
    d__1 = x[1] + x[2] * t - exp(t);
/* Computing 2nd power */
    d__2 = x[3] + x[4] * sin(t) - cos(t);
    *fa = d__1 * d__1 + d__2 * d__2;
    return 0;
L140:
    t = empr06_1.y[*ka - 1];
/* Computing 2nd power */
    d__1 = (x[1] * t + x[2]) * t + x[3];
    *fa = x[4] - d__1 * d__1 - empr06_1.y[*ka + 20];
    return 0;
L150:
    t = empr06_1.y[*ka - 1];
    *fa = x[1] * exp(x[3] * t) + x[2] * exp(x[4] * t) - empr06_1.y[*ka + 20];
    return 0;
L200:
    t = empr06_1.y[*ka - 1];
    d__2 = (d__1 = (t + x[2] + 1. / (x[3] * t + x[4])) / ((t + 1.) * 
	    empr06_1.y[*ka + 60]), abs(d__1));
    d__3 = t + .5;
    *fa = x[1] * pow_dd(&d__2, &d__3) - 1.;
    return 0;
L30:
    t = (doublereal) (*ka - 1) * .1 - 1.;
    x1 = x[1] + t * x[2];
    x2 = 1. / (t * (x[3] + t * (x[4] + t * x[5])) + 1.);
    x3 = x1 * x2 - exp(t);
    *fa = x3;
L31:
    return 0;
L130:
    t = empr06_1.y[*ka - 1];
    *fa = (x[1] + t * (x[2] + t * x[3])) / (t * (x[4] + t * x[5]) + 1.) - 
	    empr06_1.y[*ka + 29];
    return 0;
L100:
    t = (doublereal) (*ka - 1) * .1;
    *fa = x[1] * exp(-x[2] * t) * cos(x[3] * t + x[4]) + x[5] * exp(-x[6] * t)
	     - empr06_1.y[*ka - 1];
    return 0;
L40:
    beta = empr06_1.y[*ka - 1] * 1.5707963267948966;
    for (i__ = 1; i__ <= 3; ++i__) {
	j = i__ + i__;
	xa[i__ - 1] = x[j - 1];
	xb[i__ - 1] = x[j];
/* L41: */
    }
    ca[3].r = 1., ca[3].i = 0.;
    z__1.r = ca[3].r * 10., z__1.i = ca[3].i * 10.;
    cb[3].r = z__1.r, cb[3].i = z__1.i;
    for (j = 1; j <= 3; ++j) {
	i__ = 4 - j;
	xc[i__ - 1] = beta * xa[i__ - 1];
	t = xc[i__ - 1];
	x1 = cos(t);
	x2 = sin(t);
	z__1.r = x1, z__1.i = 0.;
	c1.r = z__1.r, c1.i = z__1.i;
	d__1 = x2 * xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c2.r = z__1.r, c2.i = z__1.i;
	d__1 = x2 / xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c3.r = z__1.r, c3.i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c1.r * cb[i__2].r - c1.i * cb[i__2].i, z__2.i = c1.r * cb[
		i__2].i + c1.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c2.r * ca[i__3].r - c2.i * ca[i__3].i, z__3.i = c2.r * ca[
		i__3].i + c2.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	cb[i__1].r = z__1.r, cb[i__1].i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c3.r * cb[i__2].r - c3.i * cb[i__2].i, z__2.i = c3.r * cb[
		i__2].i + c3.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c1.r * ca[i__3].r - c1.i * ca[i__3].i, z__3.i = c1.r * ca[
		i__3].i + c1.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	ca[i__1].r = z__1.r, ca[i__1].i = z__1.i;
/* L42: */
    }
    z__1.r = -ca[0].r, z__1.i = -ca[0].i;
    c1.r = z__1.r, c1.i = z__1.i;
    z__1.r = cb[0].r - c1.r, z__1.i = cb[0].i - c1.i;
    c2.r = z__1.r, c2.i = z__1.i;
    z__3.r = c1.r * 2., z__3.i = c1.i * 2.;
    z_div(&z__2, &z__3, &c2);
    z__1.r = z__2.r + 1., z__1.i = z__2.i;
    c3.r = z__1.r, c3.i = z__1.i;
    *fa = z_abs(&c3);
    return 0;
L110:
    t = empr06_1.y[*ka + 40];
    beta = empr06_1.y[*ka + 81];
/* Computing 2nd power */
    d__1 = x[1] + (x[2] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[2]) * beta;
    x1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[3] + (x[4] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[4]) * beta;
    x2 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[5] + (x[6] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[6]) * beta;
    x3 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[7] + (x[8] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[8]) * beta;
    x4 = d__1 * d__1 + d__2 * d__2;
    if (x2 == 0.) {
	x2 = 1e-30;
    }
    if (x4 == 0.) {
	x4 = 1e-30;
    }
    *fa = x[9] * sqrt(x1 / x2) * sqrt(x3 / x4) - (d__1 = 1. - empr06_1.y[*ka 
	    - 1] * 2., abs(d__1));
    return 0;
L50:
/* Computing 2nd power */
    d__1 = x[1] - 10.;
/* Computing 2nd power */
    d__2 = x[2] - 12.;
/* Computing 4th power */
    d__3 = x[3], d__3 *= d__3;
/* Computing 2nd power */
    d__4 = x[4] - 11.;
/* Computing 6th power */
    d__5 = x[5], d__5 *= d__5;
/* Computing 2nd power */
    d__6 = x[6];
/* Computing 4th power */
    d__7 = x[7], d__7 *= d__7;
    *fa = d__1 * d__1 + d__2 * d__2 * 5. + d__3 * d__3 + d__4 * d__4 * 3. + 
	    d__5 * (d__5 * d__5) * 10. + d__6 * d__6 * 7. + d__7 * d__7 - x[6]
	     * 4. * x[7] - x[6] * 10. - x[7] * 8.;
/* L51: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L52;
	case 3:  goto L53;
	case 4:  goto L54;
	case 5:  goto L55;
    }
L52:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 4th power */
    d__2 = x[2], d__2 *= d__2;
/* Computing 2nd power */
    d__3 = x[4];
    *fa += (d__1 * d__1 * 2. + d__2 * d__2 * 3. + x[3] + d__3 * d__3 * 4. + x[
	    5] * 5. - 127.) * 10.;
    return 0;
L53:
/* Computing 2nd power */
    d__1 = x[3];
    *fa += (x[1] * 7. + x[2] * 3. + d__1 * d__1 * 10. + x[4] - x[5] - 282.) * 
	    10.;
    return 0;
L54:
/* Computing 2nd power */
    d__1 = x[2];
/* Computing 2nd power */
    d__2 = x[6];
    *fa += (x[1] * 23. + d__1 * d__1 + d__2 * d__2 * 6. - x[7] * 8. - 196.) * 
	    10.;
    return 0;
L55:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[3];
    *fa += (d__1 * d__1 * 4. + d__2 * d__2 - x[1] * 3. * x[2] + d__3 * d__3 * 
	    2. + x[6] * 5. - x[7] * 11.) * 10.;
    return 0;
L60:
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
/* L61: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L66;
	case 7:  goto L67;
	case 8:  goto L68;
	case 9:  goto L69;
    }
L62:
/* Computing 2nd power */
    d__1 = x[1] - 2.;
/* Computing 2nd power */
    d__2 = x[2] - 3.;
/* Computing 2nd power */
    d__3 = x[3];
    *fa += (d__1 * d__1 * 3. + d__2 * d__2 * 4. + d__3 * d__3 * 2. - x[4] * 
	    7. - 120.) * 10.;
    return 0;
L63:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[3] - 6.;
    *fa += (d__1 * d__1 * 5. + x[2] * 8. + d__2 * d__2 - x[4] * 2. - 40.) * 
	    10.;
    return 0;
L64:
/* Computing 2nd power */
    d__1 = x[1] - 8.;
/* Computing 2nd power */
    d__2 = x[2] - 4.;
/* Computing 2nd power */
    d__3 = x[5];
    *fa += (d__1 * d__1 * .5 + d__2 * d__2 * 2. + d__3 * d__3 * 3. - x[6] - 
	    30.) * 10.;
    return 0;
L65:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2] - 2.;
    *fa += (d__1 * d__1 + d__2 * d__2 * 2. - x[1] * 2. * x[2] + x[5] * 14. - 
	    x[6] * 6.) * 10.;
    return 0;
L66:
    *fa += (x[1] * 4. + x[2] * 5. - x[7] * 3. + x[8] * 9. - 105.) * 10.;
    return 0;
L67:
    *fa += (x[1] * 10. - x[2] * 8. - x[7] * 17. + x[8] * 2.) * 10.;
    return 0;
L68:
/* Computing 2nd power */
    d__1 = x[9] - 8.;
    *fa += (x[2] * 6. - x[1] * 3. + d__1 * d__1 * 12. - x[10] * 7.) * 10.;
    return 0;
L69:
    *fa += (x[2] * 2. - x[1] * 8. + x[9] * 5. - x[10] * 2. - 12.) * 10.;
    return 0;
L70:
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
/* L71: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L66;
	case 7:  goto L67;
	case 8:  goto L68;
	case 9:  goto L69;
	case 10:  goto L72;
	case 11:  goto L73;
	case 12:  goto L74;
	case 13:  goto L75;
	case 14:  goto L76;
	case 15:  goto L77;
	case 16:  goto L78;
	case 17:  goto L79;
	case 18:  goto L89;
    }
L72:
    *fa += (x[1] + x[2] + x[11] * 4. - x[12] * 21.) * 10.;
    return 0;
L73:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 + x[11] * 15. - x[12] * 8. - 28.) * 10.;
    return 0;
L74:
/* Computing 2nd power */
    d__1 = x[13];
    *fa += (x[1] * 4. + x[2] * 9. + d__1 * d__1 * 5. - x[14] * 9. - 87.) * 
	    10.;
    return 0;
L75:
/* Computing 2nd power */
    d__1 = x[13] - 6.;
    *fa += (x[1] * 3. + x[2] * 4. + d__1 * d__1 * 3. - x[14] * 14. - 10.) * 
	    10.;
    return 0;
L76:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 * 14. + x[15] * 35. - x[16] * 79. - 92.) * 10.;
    return 0;
L77:
/* Computing 2nd power */
    d__1 = x[2];
    *fa += (d__1 * d__1 * 15. + x[15] * 11. - x[16] * 61. - 54.) * 10.;
    return 0;
L78:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 4th power */
    d__2 = x[17], d__2 *= d__2;
    *fa += (d__1 * d__1 * 5. + x[2] * 2. + d__2 * d__2 * 9. - x[18] - 68.) * 
	    10.;
    return 0;
L79:
/* Computing 2nd power */
    d__1 = x[1];
    *fa += (d__1 * d__1 - x[2] + x[19] * 19. - x[20] * 20. + 19.) * 10.;
    return 0;
L89:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
/* Computing 2nd power */
    d__3 = x[19];
    *fa += (d__1 * d__1 * 7. + d__2 * d__2 * 5. + d__3 * d__3 - x[20] * 30.) *
	     10.;
    return 0;
L220:
    x1 = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x3 = 1.;
	x4 = x[i__];
	if (i__ == 1) {
	    x3 = 1e-8;
	}
	if (i__ == 4) {
	    x3 = 4.;
	}
	if (i__ == 2 && *ka == 1) {
	    x4 = x[i__] + 2.;
	}
	if (i__ == 2 && *ka == 2) {
	    x4 = x[i__] - 2.;
	}
/* Computing 2nd power */
	d__1 = x4;
	x1 += x3 * (d__1 * d__1);
/* L221: */
    }
    *fa = exp(x1);
    return 0;
L160:
    *fa = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x1 = (doublereal) (i__ + *ka - 1) * 1.;
	x2 = x[i__] - sin((doublereal) ((i__ << 1) + *ka - 3));
/* Computing 2nd power */
	d__1 = x2;
	*fa += x1 * exp(d__1 * d__1);
/* L161: */
    }
    return 0;
L240:
    if (*ka == 1) {
	*fa = x[1];
    } else if (*ka == 2) {
/* Computing 2nd power */
	d__1 = x[1];
	*fa = x[2] - d__1 * d__1 - 1.;
    } else {
	t = (doublereal) (*ka - 2) / 29.;
	x1 = 0.;
	x2 = x[1];
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    i__2 = i__ - 2;
	    x1 += (doublereal) (i__ - 1) * x[i__] * pow_di(&t, &i__2);
	    i__2 = i__ - 1;
	    x2 += x[i__] * pow_di(&t, &i__2);
/* L241: */
	}
/* Computing 2nd power */
	d__1 = x2;
	*fa = x1 - d__1 * d__1 - 1.;
    }
    return 0;
L250:
    t = (doublereal) (*ka - 1) * .1;
/* Computing 2nd power */
    d__1 = t - x[9];
/* Computing 2nd power */
    d__2 = t - x[10];
/* Computing 2nd power */
    d__3 = t - x[11];
    *fa = empr06_1.y[*ka - 1] - x[1] * exp(-x[5] * t) - x[2] * exp(-x[6] * (
	    d__1 * d__1)) - x[3] * exp(-x[7] * (d__2 * d__2)) - x[4] * exp(-x[
	    8] * (d__3 * d__3));
    return 0;
} /* tafu06_ */

/* SUBROUTINE TAGU06             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 90/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  GRADIENTS OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  GA(N)  GRADIENT OF THE PARTIAL FUNCTION AT THE */
/*          SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tagu06_(integer *n, integer *ka, doublereal *x, 
	doublereal *ga, integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6;

    /* Builtin functions */
    double exp(doublereal), sqrt(doublereal), cos(doublereal), sin(doublereal)
	    ;
    void pow_zi(doublecomplex *, doublecomplex *, integer *), z_div(
	    doublecomplex *, doublecomplex *, doublecomplex *);
    double pow_dd(doublereal *, doublereal *), z_abs(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);
    double pow_di(doublereal *, integer *);

    /* Local variables */
    static integer i__, j;
    static doublereal t;
    static doublecomplex c1, c2, c3;
    static doublereal x1, x2, x3, x4, x5, x6, x7, x8;
    static doublecomplex ca[4], cb[4], cc[6];
    static doublereal fa, xa[3], xb[3], xc[3], beta;

    /* Parameter adjustments */
    --ga;
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L190;
	case 3:  goto L180;
	case 4:  goto L80;
	case 5:  goto L20;
	case 6:  goto L170;
	case 7:  goto L120;
	case 8:  goto L90;
	case 9:  goto L230;
	case 10:  goto L210;
	case 11:  goto L140;
	case 12:  goto L150;
	case 13:  goto L200;
	case 14:  goto L30;
	case 15:  goto L130;
	case 16:  goto L100;
	case 17:  goto L40;
	case 18:  goto L110;
	case 19:  goto L50;
	case 20:  goto L60;
	case 21:  goto L70;
	case 22:  goto L220;
	case 23:  goto L160;
	case 24:  goto L240;
	case 25:  goto L250;
    }
L10:
    x1 = x[1] * x[1];
    x2 = x[2] * x[2];
    x3 = x[1] + x[1];
    x4 = x[2] + x[2];
    switch (*ka) {
	case 1:  goto L11;
	case 2:  goto L12;
	case 3:  goto L13;
    }
L11:
    ga[1] = x3;
    ga[2] = (x2 + x2) * x4;
    return 0;
L12:
    ga[1] = x3 - 4.;
    ga[2] = x4 - 4.;
    return 0;
L13:
    fa = exp(x[2] - x[1]) * 2.;
    ga[1] = -fa;
    ga[2] = fa;
    return 0;
L190:
/* Computing 2nd power */
    d__1 = x[1] + .1;
    x1 = 1. / (d__1 * d__1);
    ga[2] = x[2] * 2.;
    if (*ka == 1) {
	ga[1] = (x1 + 1.) * .5;
    } else if (*ka == 2) {
	ga[1] = (x1 - 1.) * .5;
    } else if (*ka == 3) {
	ga[1] = (1. - x1) * .5;
    }
    return 0;
L180:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    x1 = d__1 * d__1 + d__2 * d__2;
    x2 = sqrt(x1);
    x3 = cos(x2);
    x4 = sin(x2);
    if (*ka == 1) {
	x5 = (x[1] - x2 * x3) * 2.;
	x6 = -(x3 / x2 - x4);
	ga[1] = x5 * (x[1] * x6 + 1.) + x[1] * .01;
	ga[2] = x5 * x[2] * x6 + x[2] * .01;
    } else if (*ka == 2) {
	x5 = (x[2] - x2 * x4) * 2.;
	x6 = -(x4 / x2 + x3);
	ga[1] = x5 * x[1] * x6 + x[1] * .01;
	ga[2] = x5 * (x[2] * x6 + 1.) + x[2] * .01;
    }
    return 0;
L80:
    switch (*ka) {
	case 1:  goto L81;
	case 2:  goto L82;
	case 3:  goto L83;
	case 4:  goto L84;
	case 5:  goto L85;
	case 6:  goto L86;
    }
L81:
    ga[1] = x[1] * 2.;
    ga[2] = x[2] * 2.;
    ga[3] = x[3] * 2.;
    return 0;
L82:
    ga[1] = x[1] * 2.;
    ga[2] = x[2] * 2.;
    ga[3] = (x[3] - 2.) * 2.;
    return 0;
L83:
    ga[1] = 1.;
    ga[2] = 1.;
    ga[3] = 1.;
    return 0;
L84:
    ga[1] = 1.;
    ga[2] = 1.;
    ga[3] = -1.;
    return 0;
L85:
/* Computing 2nd power */
    d__1 = x[1];
    ga[1] = d__1 * d__1 * 6. - (x[3] * 5. - x[1] + 1.) * 4.;
    ga[2] = x[2] * 12.;
    ga[3] = (x[3] * 5. - x[1] + 1.) * 20.;
    return 0;
L86:
    ga[1] = x[1] * 2.;
    ga[2] = 0.;
    ga[3] = -9.;
    return 0;
L20:
    x1 = x[1] * x[1];
    x2 = x[2] * x[2];
    x3 = x[3] * x[3];
    x4 = x[4] * x[4];
    x5 = x[1] + x[1];
    x6 = x[2] + x[2];
    x7 = x[3] + x[3];
    x8 = x[4] + x[4];
    ga[1] = x5 - 5.;
    ga[2] = x6 - 5.;
    ga[3] = x7 + x7 - 21.;
    ga[4] = x8 + 7.;
/* L21: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L22;
	case 3:  goto L23;
	case 4:  goto L24;
    }
L22:
    ga[1] += (x5 + 1.) * 10.;
    ga[2] += (x6 - 1.) * 10.;
    ga[3] += (x7 + 1.) * 10.;
    ga[4] += (x8 - 1.) * 10.;
    return 0;
L23:
    ga[1] += (x5 - 1.) * 10.;
    ga[2] += (x6 + x6) * 10.;
    ga[3] += x7 * 10.;
    ga[4] += (x8 + x8 - 1.) * 10.;
    return 0;
L24:
    ga[1] += (x5 + 2.) * 10.;
    ga[2] += (x6 - 1.) * 10.;
    ga[3] += x7 * 10.;
    ga[4] += -10.;
    return 0;
L170:
/* Computing 4th power */
    d__1 = x[4] + 1., d__1 *= d__1;
    x1 = x[1] - d__1 * d__1;
    x2 = x1 * x1;
    x3 = x[2] - x2 * x2;
    x4 = x1 * x3;
/* Computing 3rd power */
    d__1 = x[4] + 1.;
    x5 = d__1 * (d__1 * d__1) * -4.;
    ga[1] = x1 * 2. - x4 * 8. - (1. - x1 * 4.) * 5.;
    ga[2] = x3 * 2. - 5.;
    ga[3] = x[3] * 4. - 21.;
    ga[4] = x1 * 2. * x5 - x4 * 8. * x5 + x[4] * 2. - (x5 - x1 * 4. * x5) * 
	    5. + 7.;
    switch (*ka) {
	case 1:  goto L171;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
    }
L171:
    return 0;
L172:
    ga[1] += (x1 * 2. - x4 * 8. + 1. + x1 * 4.) * 10.;
    ga[2] += (x3 * 2. - 1.) * 10.;
    ga[3] += (x[3] * 2. + 1.) * 10.;
    ga[4] += (x1 * 2. * x5 - x4 * 8. * x5 + x[4] * 2. + x5 + x1 * 4. * x5 - 
	    1.) * 10.;
    return 0;
L173:
    ga[1] += (x1 * 2. - x4 * 16. - 1.) * 10.;
    ga[2] += x3 * 4. * 10.;
    ga[3] += x[3] * 2. * 10.;
    ga[4] += (x1 * 2. * x5 - x4 * 16. * x5 + x[4] * 4. - x5 - 1.) * 10.;
    return 0;
L174:
    ga[1] += (x1 * 2. - x4 * 8. + 2. + x1 * 4.) * 10.;
    ga[2] += (x3 * 2. - 1.) * 10.;
    ga[3] += x[3] * 2. * 10.;
    ga[4] += (x1 * 2. * x5 - x4 * 8. * x5 + x5 * 2. + x1 * 4. * x5 - 1.) * 
	    10.;
    return 0;
L120:
    t = empr06_1.y[*ka - 1];
    x1 = exp(-x[1] * t) / x[2];
    x2 = sin(x[2] * t);
    x3 = cos(x[2] * t);
    ga[1] = -t * x[3] * x1 * x2;
    ga[2] = x[3] * x1 * (t * x3 - x2 / x[2]);
    ga[3] = x1 * x2;
    return 0;
L90:
    d__1 = (doublereal) (16 - *ka);
    c1.r = d__1, c1.i = 0.;
/* Computing MIN */
    i__1 = *ka, i__2 = 16 - *ka;
    d__1 = (doublereal) min(i__1,i__2);
    c2.r = d__1, c2.i = 0.;
    d__1 = (doublereal) (*ka);
    z__2.r = d__1, z__2.i = 0.;
    z__5.r = x[2] * c1.r, z__5.i = x[2] * c1.i;
    z__6.r = x[3] * c2.r, z__6.i = x[3] * c2.i;
    z__4.r = z__5.r + z__6.r, z__4.i = z__5.i + z__6.i;
    pow_zi(&z__3, &z__4, &c__2);
    z_div(&z__1, &z__2, &z__3);
    c3.r = z__1.r, c3.i = z__1.i;
    ga[1] = -1.;
    z__1.r = c1.r * c3.r - c1.i * c3.i, z__1.i = c1.r * c3.i + c1.i * c3.r;
    ga[2] = z__1.r;
    z__1.r = c2.r * c3.r - c2.i * c3.i, z__1.i = c2.r * c3.i + c2.i * c3.r;
    ga[3] = z__1.r;
    return 0;
L230:
    t = empr06_1.y[*ka + 10];
    x1 = x[1] * t * (t + x[2]);
    x2 = (t + x[3]) * t + x[4];
/* Computing 2nd power */
    d__1 = x2;
    x3 = x1 / (d__1 * d__1);
    ga[1] = -t * (t + x[2]) / x2;
    ga[2] = -t * x[1] / x2;
    ga[3] = t * x3;
    ga[4] = x3;
    return 0;
L210:
    t = (doublereal) (*ka) * .2;
    ga[1] = (x[1] + x[2] * t - exp(t)) * 2.;
    ga[2] = (x[1] + x[2] * t - exp(t)) * 2. * t;
    ga[3] = (x[3] + x[4] * sin(t) - cos(t)) * 2.;
    ga[4] = (x[3] + x[4] * sin(t) - cos(t)) * 2. * sin(t);
    return 0;
L140:
    t = empr06_1.y[*ka - 1];
    x1 = ((x[1] * t + x[2]) * t + x[3]) * -2.;
/* Computing 2nd power */
    d__1 = t;
    ga[1] = x1 * (d__1 * d__1);
    ga[2] = x1 * t;
    ga[3] = x1;
    ga[4] = 1.;
    return 0;
L150:
    t = empr06_1.y[*ka - 1];
    x1 = exp(x[3] * t);
    x2 = exp(x[4] * t);
    ga[1] = x1;
    ga[2] = x2;
    ga[3] = x[1] * t * x1;
    ga[4] = x[2] * t * x2;
    return 0;
L200:
    t = empr06_1.y[*ka - 1];
    x1 = t + x[2] + 1. / (x[3] * t + x[4]);
    if (x1 == 0.) {
	x1 = 1e-30;
    }
    x2 = x1 / ((t + 1.) * empr06_1.y[*ka + 60]);
    d__1 = abs(x2);
    d__2 = t + .5;
    ga[1] = pow_dd(&d__1, &d__2);
    ga[2] = x[1] * ga[1] * (t + .5) / x1;
/* Computing 2nd power */
    d__1 = x[3] * t + x[4];
    ga[4] = -ga[2] / (d__1 * d__1);
    ga[3] = ga[4] * t;
    return 0;
L30:
    t = (doublereal) (*ka - 1) * .1 - 1.;
    x1 = x[1] + t * x[2];
    x2 = 1. / (t * (x[3] + t * (x[4] + t * x[5])) + 1.);
    x3 = x1 * x2 - exp(t);
    ga[1] = x2;
    ga[2] = x2 * t;
    ga[3] = -x1 * x2 * x2 * t;
    ga[4] = ga[3] * t;
    ga[5] = ga[4] * t;
L31:
    return 0;
L130:
    t = empr06_1.y[*ka - 1];
    x1 = 1. / (t * (x[4] + t * x[5]) + 1.);
    x2 = x[1] + t * (x[2] + t * x[3]);
    ga[1] = x1;
    ga[2] = x1 * t;
    ga[3] = x1 * t * t;
/* Computing 2nd power */
    d__1 = x1;
    ga[4] = -x2 * (d__1 * d__1) * t;
/* Computing 2nd power */
    d__1 = x1;
    ga[5] = -x2 * (d__1 * d__1) * t * t;
    return 0;
L100:
    t = (doublereal) (*ka - 1) * .1;
    x1 = exp(-x[2] * t);
    x2 = cos(x[3] * t + x[4]);
    x3 = sin(x[3] * t + x[4]);
    x4 = exp(-x[6] * t);
    ga[1] = x1 * x2;
    ga[2] = -x1 * x2 * x[1] * t;
    ga[3] = -x1 * x3 * x[1] * t;
    ga[4] = -x1 * x3 * x[1];
    ga[5] = x4;
    ga[6] = -x4 * x[5] * t;
    return 0;
L40:
    beta = empr06_1.y[*ka - 1] * 1.5707963267948966;
    for (i__ = 1; i__ <= 3; ++i__) {
	j = i__ + i__;
	xa[i__ - 1] = x[j - 1];
	xb[i__ - 1] = x[j];
/* L41: */
    }
    ca[3].r = 1., ca[3].i = 0.;
    z__1.r = ca[3].r * 10., z__1.i = ca[3].i * 10.;
    cb[3].r = z__1.r, cb[3].i = z__1.i;
    for (j = 1; j <= 3; ++j) {
	i__ = 4 - j;
	xc[i__ - 1] = beta * xa[i__ - 1];
	t = xc[i__ - 1];
	x1 = cos(t);
	x2 = sin(t);
	z__1.r = x1, z__1.i = 0.;
	c1.r = z__1.r, c1.i = z__1.i;
	d__1 = x2 * xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c2.r = z__1.r, c2.i = z__1.i;
	d__1 = x2 / xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c3.r = z__1.r, c3.i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c1.r * cb[i__2].r - c1.i * cb[i__2].i, z__2.i = c1.r * cb[
		i__2].i + c1.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c2.r * ca[i__3].r - c2.i * ca[i__3].i, z__3.i = c2.r * ca[
		i__3].i + c2.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	cb[i__1].r = z__1.r, cb[i__1].i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c3.r * cb[i__2].r - c3.i * cb[i__2].i, z__2.i = c3.r * cb[
		i__2].i + c3.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c1.r * ca[i__3].r - c1.i * ca[i__3].i, z__3.i = c1.r * ca[
		i__3].i + c1.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	ca[i__1].r = z__1.r, ca[i__1].i = z__1.i;
/* L42: */
    }
    z__1.r = -ca[0].r, z__1.i = -ca[0].i;
    c1.r = z__1.r, c1.i = z__1.i;
    z__1.r = cb[0].r - c1.r, z__1.i = cb[0].i - c1.i;
    c2.r = z__1.r, c2.i = z__1.i;
    z__3.r = c1.r * 2., z__3.i = c1.i * 2.;
    z_div(&z__2, &z__3, &c2);
    z__1.r = z__2.r + 1., z__1.i = z__2.i;
    c3.r = z__1.r, c3.i = z__1.i;
    fa = z_abs(&c3);
    d_cnjg(&z__1, &c3);
    c3.r = z__1.r, c3.i = z__1.i;
    z_div(&z__1, &c_b168, &c2);
    c1.r = z__1.r, c1.i = z__1.i;
    for (i__ = 1; i__ <= 3; ++i__) {
	t = xc[i__ - 1];
	j = i__ + i__;
	i__1 = j - 1;
	i__2 = i__ - 1;
	i__3 = i__ - 1;
	z__3.r = cb[i__2].r * ca[i__3].r - cb[i__2].i * ca[i__3].i, z__3.i = 
		cb[i__2].r * ca[i__3].i + cb[i__2].i * ca[i__3].r;
	i__4 = i__;
	i__5 = i__;
	z__4.r = cb[i__4].r * ca[i__5].r - cb[i__4].i * ca[i__5].i, z__4.i = 
		cb[i__4].r * ca[i__5].i + cb[i__4].i * ca[i__5].r;
	z__2.r = z__3.r - z__4.r, z__2.i = z__3.i - z__4.i;
	i__6 = i__ - 1;
	z__5.r = xb[i__6] * c2.r, z__5.i = xb[i__6] * c2.i;
	z_div(&z__1, &z__2, &z__5);
	cc[i__1].r = z__1.r, cc[i__1].i = z__1.i;
	i__1 = j - 2;
	i__2 = i__ - 1;
	i__3 = i__;
	z__4.r = cb[i__2].r * ca[i__3].r - cb[i__2].i * ca[i__3].i, z__4.i = 
		cb[i__2].r * ca[i__3].i + cb[i__2].i * ca[i__3].r;
	i__4 = i__;
	i__5 = i__ - 1;
	z__5.r = cb[i__4].r * ca[i__5].r - cb[i__4].i * ca[i__5].i, z__5.i = 
		cb[i__4].r * ca[i__5].i + cb[i__4].i * ca[i__5].r;
	z__3.r = z__4.r - z__5.r, z__3.i = z__4.i - z__5.i;
	z__2.r = beta * z__3.r, z__2.i = beta * z__3.i;
	d__1 = sin(t);
	z__6.r = d__1 * c2.r, z__6.i = d__1 * c2.i;
	z_div(&z__1, &z__2, &z__6);
	cc[i__1].r = z__1.r, cc[i__1].i = z__1.i;
/* L43: */
    }
    for (i__ = 1; i__ <= 6; ++i__) {
	z__2.r = c1.r * c3.r - c1.i * c3.i, z__2.i = c1.r * c3.i + c1.i * 
		c3.r;
	i__1 = i__ - 1;
	z__1.r = z__2.r * cc[i__1].r - z__2.i * cc[i__1].i, z__1.i = z__2.r * 
		cc[i__1].i + z__2.i * cc[i__1].r;
	ga[i__] = z__1.r / fa;
/* L44: */
    }
    return 0;
L110:
    t = empr06_1.y[*ka + 40];
    beta = empr06_1.y[*ka + 81];
/* Computing 2nd power */
    d__1 = x[1] + (x[2] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[2]) * beta;
    x1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[3] + (x[4] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[4]) * beta;
    x2 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[5] + (x[6] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[6]) * beta;
    x3 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[7] + (x[8] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[8]) * beta;
    x4 = d__1 * d__1 + d__2 * d__2;
    if (x1 == 0.) {
	x1 = 1e-30;
    }
    if (x2 == 0.) {
	x2 = 1e-30;
    }
    if (x3 == 0.) {
	x3 = 1e-30;
    }
    if (x4 == 0.) {
	x4 = 1e-30;
    }
    fa = sqrt(x1 / x2) * sqrt(x3 / x4);
    ga[9] = fa;
    fa = x[9] * fa;
    ga[1] = fa / x1 * (x[1] + t * (x[2] + 1.));
    ga[2] = fa / x1 * (x[2] + t * 2. * t - 1. + x[1] * t);
    ga[3] = -fa / x2 * (x[3] + t * (x[4] + 1.));
    ga[4] = -fa / x2 * (x[4] + t * 2. * t - 1. + x[3] * t);
    ga[5] = fa / x3 * (x[5] + t * (x[6] + 1.));
    ga[6] = fa / x3 * (x[6] + t * 2. * t - 1. + x[5] * t);
    ga[7] = -fa / x4 * (x[7] + t * (x[8] + 1.));
    ga[8] = -fa / x4 * (x[8] + t * 2. * t - 1. + x[7] * t);
    return 0;
L50:
    ga[1] = (x[1] - 10.) * 2.;
    ga[2] = (x[2] - 12.) * 10.;
/* Computing 3rd power */
    d__1 = x[3];
    ga[3] = d__1 * (d__1 * d__1) * 4.;
    ga[4] = (x[4] - 11.) * 6.;
/* Computing 5th power */
    d__1 = x[5], d__2 = d__1, d__1 *= d__1;
    ga[5] = d__2 * (d__1 * d__1) * 60.;
    ga[6] = x[6] * 14. - x[7] * 4. - 10.;
/* Computing 3rd power */
    d__1 = x[7];
    ga[7] = d__1 * (d__1 * d__1) * 4. - x[6] * 4. - 8.;
/* L51: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L52;
	case 3:  goto L53;
	case 4:  goto L54;
	case 5:  goto L55;
    }
L52:
    ga[1] += x[1] * 40.;
/* Computing 3rd power */
    d__1 = x[2];
    ga[2] += d__1 * (d__1 * d__1) * 120.;
    ga[3] += 10.;
    ga[4] += x[4] * 80.;
    ga[5] += 50.;
    return 0;
L53:
    ga[1] += 70.;
    ga[2] += 30.;
    ga[3] += x[3] * 200.;
    ga[4] += 10.;
    ga[5] += -10.;
    return 0;
L54:
    ga[1] += 230.;
    ga[2] += x[2] * 20.;
    ga[6] += x[6] * 120.;
    ga[7] += -80.;
    return 0;
L55:
    ga[1] = ga[1] + x[1] * 80. - x[2] * 30.;
    ga[2] = ga[2] + x[2] * 20. - x[1] * 30.;
    ga[3] += x[3] * 40.;
    ga[6] += 50.;
    ga[7] += -110.;
    return 0;
L60:
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
/* L61: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L66;
	case 7:  goto L67;
	case 8:  goto L68;
	case 9:  goto L69;
    }
L62:
    ga[1] += (x[1] - 2.) * 60.;
    ga[2] += (x[2] - 3.) * 80.;
    ga[3] += x[3] * 40.;
    ga[4] += -70.;
    return 0;
L63:
    ga[1] += x[1] * 100.;
    ga[2] += 80.;
    ga[3] += (x[3] - 6.) * 20.;
    ga[4] += -20.;
    return 0;
L64:
    ga[1] += (x[1] - 8.) * 10.;
    ga[2] += (x[2] - 4.) * 40.;
    ga[5] += x[5] * 60.;
    ga[6] += -10.;
    return 0;
L65:
    ga[1] = ga[1] + x[1] * 20. - x[2] * 20.;
    ga[2] = ga[2] + (x[2] - 2.) * 40. - x[1] * 20.;
    ga[5] += 140.;
    ga[6] += -60.;
    return 0;
L66:
    ga[1] += 40.;
    ga[2] += 50.;
    ga[7] += -30.;
    ga[8] += 90.;
    return 0;
L67:
    ga[1] += 100.;
    ga[2] += -80.;
    ga[7] += -170.;
    ga[8] += 20.;
    return 0;
L68:
    ga[1] += -30.;
    ga[2] += 60.;
    ga[9] += (x[9] - 8.) * 240.;
    ga[10] += -70.;
    return 0;
L69:
    ga[1] += -80.;
    ga[2] += 20.;
    ga[9] += 50.;
    ga[10] += -20.;
    return 0;
L70:
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
/* L71: */
    switch (*ka) {
	case 1:  goto L31;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L66;
	case 7:  goto L67;
	case 8:  goto L68;
	case 9:  goto L69;
	case 10:  goto L72;
	case 11:  goto L73;
	case 12:  goto L74;
	case 13:  goto L75;
	case 14:  goto L76;
	case 15:  goto L77;
	case 16:  goto L78;
	case 17:  goto L79;
	case 18:  goto L89;
    }
L72:
    ga[1] += 10.;
    ga[2] += 10.;
    ga[11] += 40.;
    ga[12] += -210.;
    return 0;
L73:
    ga[1] += x[1] * 20.;
    ga[11] += 150.;
    ga[12] += -80.;
    return 0;
L74:
    ga[1] += 40.;
    ga[2] += 90.;
    ga[13] += x[13] * 100.;
    ga[14] += -90.;
    return 0;
L75:
    ga[1] += 30.;
    ga[2] += 40.;
    ga[13] += (x[13] - 6.) * 60.;
    ga[14] += -140.;
    return 0;
L76:
    ga[1] += x[1] * 280.;
    ga[15] += 350.;
    ga[16] += -790.;
    return 0;
L77:
    ga[2] += x[2] * 300.;
    ga[15] += 110.;
    ga[16] += -610.;
    return 0;
L78:
    ga[1] += x[1] * 100.;
    ga[2] += 20.;
/* Computing 3rd power */
    d__1 = x[17];
    ga[17] += d__1 * (d__1 * d__1) * 360.;
    ga[18] += -10.;
    return 0;
L79:
    ga[1] += x[1] * 20.;
    ga[2] += -10.;
    ga[19] += 190.;
    ga[20] += -200.;
    return 0;
L89:
    ga[1] += x[1] * 140.;
    ga[2] += x[2] * 100.;
    ga[19] += x[19] * 20.;
    ga[20] += -300.;
    return 0;
L220:
    x1 = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x3 = 1.;
	x4 = x[i__];
	if (i__ == 1) {
	    x3 = 1e-8;
	}
	if (i__ == 4) {
	    x3 = 4.;
	}
	if (i__ == 2 && *ka == 1) {
	    x4 = x[i__] + 2.;
	}
	if (i__ == 2 && *ka == 2) {
	    x4 = x[i__] - 2.;
	}
/* Computing 2nd power */
	d__1 = x4;
	x1 += x3 * (d__1 * d__1);
/* L221: */
    }
    x2 = exp(x1);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x3 = 2.;
	x4 = x[i__];
	if (i__ == 1) {
	    x3 = 2e-8;
	}
	if (i__ == 4) {
	    x3 = 8.;
	}
	if (i__ == 2 && *ka == 1) {
	    x4 = x[i__] + 2.;
	}
	if (i__ == 2 && *ka == 2) {
	    x4 = x[i__] - 2.;
	}
	ga[i__] = x2 * x3 * x4;
/* L222: */
    }
    return 0;
L160:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x1 = (doublereal) (i__ + *ka - 1) * 1.;
	x2 = x[i__] - sin((doublereal) ((i__ << 1) + *ka - 3));
/* Computing 2nd power */
	d__1 = x2;
	ga[i__] = x1 * 2. * x2 * exp(d__1 * d__1);
/* L161: */
    }
    return 0;
L240:
    if (*ka <= 2) {
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    ga[i__] = 0.;
/* L241: */
	}
	if (*ka == 1) {
	    ga[1] = 1.;
	} else if (*ka == 2) {
	    ga[1] = x[1] * -2.;
	    ga[2] = 1.;
	}
    } else {
	ga[1] = 0.;
	t = (doublereal) (*ka - 2) / 29.;
	x2 = x[1];
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    i__2 = i__ - 1;
	    x2 += x[i__] * pow_di(&t, &i__2);
/* L242: */
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ > 1) {
		i__2 = i__ - 2;
		ga[i__] = (doublereal) (i__ - 1) * pow_di(&t, &i__2);
	    }
	    i__2 = i__ - 1;
	    ga[i__] -= x2 * 2. * pow_di(&t, &i__2);
/* L243: */
	}
    }
    return 0;
L250:
    t = (doublereal) (*ka - 1) * .1;
    x1 = exp(-x[5] * t);
/* Computing 2nd power */
    d__1 = t - x[9];
    x2 = exp(-x[6] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = t - x[10];
    x3 = exp(-x[7] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = t - x[11];
    x4 = exp(-x[8] * (d__1 * d__1));
    ga[1] = -x1;
    ga[2] = -x2;
    ga[3] = -x3;
    ga[4] = -x4;
    ga[5] = x1 * x[1] * t;
/* Computing 2nd power */
    d__1 = t - x[9];
    ga[6] = x2 * x[2] * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[10];
    ga[7] = x3 * x[3] * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[11];
    ga[8] = x4 * x[4] * (d__1 * d__1);
    ga[9] = x2 * -2. * x[2] * x[6] * (t - x[9]);
    ga[10] = x3 * -2. * x[3] * x[7] * (t - x[10]);
    ga[11] = x4 * -2. * x[4] * x[8] * (t - x[11]);
    return 0;
} /* tagu06_ */

/* SUBROUTINE TAHD06             ALL SYSTEMS                 99/12/01 */
/* PORTABILITY : ALL SYSTEMS */
/* 95/12/01 LU : ORIGINAL VERSION */

/* PURPOSE : */
/*  HESSIAN MATRICES OF PARTIAL FUNCTIONS IN THE MINIMAX CRITERION. */
/*  DENSE VERSION. */

/* PARAMETERS : */
/*  II  N  NUMBER OF VARIABLES. */
/*  II  KA  INDEX OF THE PARTIAL FUNCTION. */
/*  RI  X(N)  VECTOR OF VARIABLES. */
/*  RO  HA(N*(N+1)/2)  HESSIAN MATRIX OF THE PARTIAL FUNCTION */
/*         AT THE SELECTED POINT. */
/*  II  NEXT  NUMBER OF THE TEST PROBLEM. */

/* Subroutine */ int tahd06_(integer *n, integer *ka, doublereal *x, 
	doublereal *ha, integer *next)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6, i__7;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7, z__8, z__9, z__10;

    /* Builtin functions */
    double exp(doublereal), sqrt(doublereal), cos(doublereal), sin(doublereal)
	    ;
    void pow_zi(doublecomplex *, doublecomplex *, integer *), z_div(
	    doublecomplex *, doublecomplex *, doublecomplex *);
    double pow_dd(doublereal *, doublereal *), z_abs(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);
    double pow_di(doublereal *, integer *);

    /* Local variables */
    static integer i__, j, l;
    static doublereal t;
    static doublecomplex c1, c2, c3, s1, s2, s3, s4;
    static doublereal x1, x2, x3, x4, x5, x6, x7, x8;
    static doublecomplex ca[4], cb[4], cc[6];
    static doublereal fa;
    static doublecomplex dd[6];
    static doublereal ga[8];
    static doublecomplex ci;
    static doublereal ct[3], xa[3], xb[3], xc[3], st[3], beta;

    /* Parameter adjustments */
    --ha;
    --x;

    /* Function Body */
    switch (*next) {
	case 1:  goto L10;
	case 2:  goto L190;
	case 3:  goto L180;
	case 4:  goto L80;
	case 5:  goto L20;
	case 6:  goto L170;
	case 7:  goto L120;
	case 8:  goto L90;
	case 9:  goto L230;
	case 10:  goto L210;
	case 11:  goto L140;
	case 12:  goto L150;
	case 13:  goto L200;
	case 14:  goto L30;
	case 15:  goto L130;
	case 16:  goto L100;
	case 17:  goto L40;
	case 18:  goto L110;
	case 19:  goto L50;
	case 20:  goto L60;
	case 21:  goto L70;
	case 22:  goto L220;
	case 23:  goto L160;
	case 24:  goto L240;
	case 25:  goto L250;
    }
L10:
    ha[1] = 2.;
    ha[2] = 0.;
    ha[3] = 2.;
    if (*ka == 1) {
	ha[3] = x[2] * 12. * x[2];
    }
    if (*ka == 3) {
	ha[1] = exp(x[2] - x[1]) * 2.;
	ha[2] = -ha[1];
	ha[3] = ha[1];
    }
    return 0;
L190:
    ha[2] = 0.;
    ha[3] = 2.;
/* Computing 3rd power */
    d__1 = x[1] + .1;
    x1 = 1. / (d__1 * (d__1 * d__1));
    if (*ka == 3) {
	ha[1] = x1;
    } else {
	ha[1] = -x1;
    }
    return 0;
L180:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    x1 = d__1 * d__1 + d__2 * d__2;
    x2 = sqrt(x1);
    x3 = cos(x2);
    x4 = sin(x2);
    if (*ka == 1) {
	x5 = x[1] - x2 * x3;
	x6 = -(x3 / x2 - x4);
	x7 = x2 + 1. / x2;
/* Computing 2nd power */
	d__1 = x2;
	x7 = (x7 * x3 + x4) / (d__1 * d__1);
	x1 = x6 * x[1] + 1.;
	x2 = x6 * x[2];
/* Computing 2nd power */
	d__1 = x1;
/* Computing 2nd power */
	d__2 = x[1];
	ha[1] = (d__1 * d__1 + x5 * (x6 + x7 * (d__2 * d__2))) * 2. + .01;
	ha[2] = x[2] * 2. * (x6 * x1 + x5 * x7 * x[1]);
/* Computing 2nd power */
	d__1 = x2;
/* Computing 2nd power */
	d__2 = x[2];
	ha[3] = (d__1 * d__1 + x5 * (x6 + x7 * (d__2 * d__2))) * 2. + .01;
    } else if (*ka == 2) {
	x5 = x[2] - x2 * x4;
	x6 = -(x4 / x2 + x3);
	x7 = x2 + 1. / x2;
/* Computing 2nd power */
	d__1 = x2;
	x7 = (x7 * x4 - x3) / (d__1 * d__1);
	x1 = x6 * x[1];
	x2 = x6 * x[2] + 1.;
/* Computing 2nd power */
	d__1 = x1;
/* Computing 2nd power */
	d__2 = x[1];
	ha[1] = (d__1 * d__1 + x5 * (x6 + x7 * (d__2 * d__2))) * 2. + .01;
	ha[2] = x[1] * 2. * (x6 * x2 + x5 * x7 * x[2]);
/* Computing 2nd power */
	d__1 = x2;
/* Computing 2nd power */
	d__2 = x[2];
	ha[3] = (d__1 * d__1 + x5 * (x6 + x7 * (d__2 * d__2))) * 2. + .01;
    }
    return 0;
L80:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L81: */
    }
    switch (*ka) {
	case 1:  goto L82;
	case 2:  goto L82;
	case 3:  goto L83;
	case 4:  goto L83;
	case 5:  goto L85;
	case 6:  goto L86;
    }
L82:
    ha[1] = 2.;
    ha[3] = 2.;
    ha[6] = 2.;
    return 0;
L83:
    return 0;
L85:
    ha[1] = 1.6;
    ha[3] = 12.;
    ha[4] = -20.;
    ha[6] = 100.;
    return 0;
L86:
    ha[1] = 2.;
    return 0;
L20:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L21: */
    }
    ha[1] = 2.;
    ha[3] = 2.;
    ha[6] = 4.;
    ha[10] = 2.;
    if (*ka > 1) {
	ha[1] += 20.;
	ha[3] += 20.;
	ha[6] += 20.;
	ha[10] += 20.;
	switch (*ka - 1) {
	    case 1:  goto L24;
	    case 2:  goto L22;
	    case 3:  goto L23;
	}
L22:
	ha[3] += 20.;
	ha[10] += 20.;
	goto L24;
L23:
	ha[10] += -20.;
L24:
	;
    }
    return 0;
L170:
/* Computing 4th power */
    d__1 = x[4] + 1., d__1 *= d__1;
    x1 = x[1] - d__1 * d__1;
    x2 = x1 * x1;
    x3 = x[2] - x2 * x2;
    x4 = x3 * x3;
/* Computing 3rd power */
    d__1 = x[4] + 1.;
    x5 = d__1 * (d__1 * d__1) * -4.;
/* Computing 2nd power */
    d__1 = x[4] + 1.;
    x6 = d__1 * d__1 * -12.;
    x7 = x1 * x6 + x5 * x5;
    x8 = x3 - x1 * 4. * x1;
    ha[1] = 2. - x8 * 8. + 20.;
    ha[2] = x1 * -8.;
    ha[3] = 2.;
    ha[4] = 0.;
    ha[5] = 0.;
    ha[6] = 4.;
    ha[7] = x5 * 2. - x5 * 8. * x8 + x5 * 20.;
    ha[8] = x1 * -8. * x5;
    ha[9] = 0.;
    ha[10] = x7 * 2. - (x5 * x5 * x8 + x1 * x3 * x6) * 8. - (x6 - x7 * 4.) * 
	    5. + 2.;
    switch (*ka) {
	case 1:  goto L171;
	case 2:  goto L172;
	case 3:  goto L173;
	case 4:  goto L174;
    }
L171:
    return 0;
L172:
    ha[1] += (2. - x8 * 8. + 4.) * 10.;
    ha[2] -= x1 * 80.;
    ha[3] += 20.;
    ha[6] += 20.;
    ha[7] += (x5 * 2. - x5 * 8. * x8 + x5 * 4.) * 10.;
    ha[8] -= x1 * 80. * x5;
    ha[10] += (x7 * 2. - (x5 * x5 * x8 + x1 * x3 * x6) * 8. + x6 + x7 * 4. + 
	    2.) * 10.;
    return 0;
L173:
    ha[1] += (2. - x8 * 16.) * 10.;
    ha[2] -= x1 * 160.;
    ha[3] += 40.;
    ha[6] += 20.;
    ha[7] += (x5 * 2. - x5 * 16. * x8) * 10.;
    ha[8] -= x1 * 160. * x5;
    ha[10] += (x7 * 2. - (x5 * x5 * x8 + x1 * x3 * x6) * 16. - x6 + 4.) * 10.;
    return 0;
L174:
    ha[1] += (2. - x8 * 8. + 4.) * 10.;
    ha[2] -= x1 * 80.;
    ha[3] += 20.;
    ha[6] += 20.;
    ha[7] += (x5 * 2. - x5 * 8. * x8 + x5 * 4.) * 10.;
    ha[8] -= x1 * 80. * x5;
    ha[10] += (x7 * 2. - (x5 * x5 * x8 + x1 * x3 * x6) * 8. + x6 * 2. + x7 * 
	    4.) * 10.;
    return 0;
L120:
    t = empr06_1.y[*ka - 1];
    x1 = exp(-x[1] * t) / x[2];
    x2 = sin(x[2] * t);
    x3 = cos(x[2] * t);
    x4 = t * x3 - x2 / x[2];
/* Computing 2nd power */
    d__1 = t;
    ha[1] = x[3] * x1 * x2 * (d__1 * d__1);
    ha[2] = -x[3] * x1 * x4 * t;
/* Computing 2nd power */
    d__1 = t;
/* Computing 2nd power */
    d__2 = x[2];
    ha[3] = -x[3] * x1 * (d__1 * d__1 * x2 + t * 2. * x3 / x[2] - x2 * 2. / (
	    d__2 * d__2));
    ha[4] = -x1 * x2 * t;
    ha[5] = x1 * x4 * t;
    ha[6] = 0.;
    return 0;
L90:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L91: */
    }
    d__1 = (doublereal) (16 - *ka);
    c1.r = d__1, c1.i = 0.;
/* Computing MIN */
    i__1 = *ka, i__2 = 16 - *ka;
    d__1 = (doublereal) min(i__1,i__2);
    c2.r = d__1, c2.i = 0.;
    d__1 = (doublereal) (*ka) * -2.;
    z__2.r = d__1, z__2.i = 0.;
    z__5.r = x[2] * c1.r, z__5.i = x[2] * c1.i;
    z__6.r = x[3] * c2.r, z__6.i = x[3] * c2.i;
    z__4.r = z__5.r + z__6.r, z__4.i = z__5.i + z__6.i;
    pow_zi(&z__3, &z__4, &c__3);
    z_div(&z__1, &z__2, &z__3);
    c3.r = z__1.r, c3.i = z__1.i;
    z__2.r = c1.r * c1.r - c1.i * c1.i, z__2.i = c1.r * c1.i + c1.i * c1.r;
    z__1.r = z__2.r * c3.r - z__2.i * c3.i, z__1.i = z__2.r * c3.i + z__2.i * 
	    c3.r;
    ha[3] = z__1.r;
    z__2.r = c1.r * c2.r - c1.i * c2.i, z__2.i = c1.r * c2.i + c1.i * c2.r;
    z__1.r = z__2.r * c3.r - z__2.i * c3.i, z__1.i = z__2.r * c3.i + z__2.i * 
	    c3.r;
    ha[5] = z__1.r;
    z__2.r = c2.r * c2.r - c2.i * c2.i, z__2.i = c2.r * c2.i + c2.i * c2.r;
    z__1.r = z__2.r * c3.r - z__2.i * c3.i, z__1.i = z__2.r * c3.i + z__2.i * 
	    c3.r;
    ha[6] = z__1.r;
    return 0;
L230:
    t = empr06_1.y[*ka + 10];
    x1 = x[1] * t * (t + x[2]);
    x2 = (t + x[3]) * t + x[4];
/* Computing 2nd power */
    d__1 = x2;
    x3 = 1. / (d__1 * d__1);
    x4 = t * (t + x[2]) * x3;
    x5 = x1 * -2. * x3 / x2;
    ha[1] = 0.;
    ha[2] = -t / x2;
    ha[3] = 0.;
    ha[7] = x4;
    ha[4] = t * x4;
    ha[8] = t * x[1] * x3;
    ha[5] = t * ha[8];
    ha[10] = x5;
    ha[9] = t * x5;
    ha[6] = t * ha[9];
    return 0;
L210:
    t = (doublereal) (*ka) * .2;
    ha[1] = 2.;
    ha[2] = t * 2.;
    ha[3] = t * 2. * t;
    ha[4] = 0.;
    ha[5] = 0.;
    ha[6] = 2.;
    ha[7] = 0.;
    ha[8] = 0.;
    ha[9] = sin(t) * 2.;
/* Computing 2nd power */
    d__1 = sin(t);
    ha[10] = d__1 * d__1 * 2.;
    return 0;
L140:
    t = empr06_1.y[*ka - 1];
    for (i__ = 7; i__ <= 10; ++i__) {
	ha[i__] = 0.;
/* L141: */
    }
    ha[6] = -2.;
    ha[5] = ha[6] * t;
    ha[4] = ha[5] * t;
    ha[3] = ha[4];
    ha[2] = ha[3] * t;
    ha[1] = ha[2] * t;
    return 0;
L150:
    t = empr06_1.y[*ka - 1];
    for (i__ = 1; i__ <= 10; ++i__) {
	ha[i__] = 0.;
/* L151: */
    }
    x1 = exp(x[3] * t);
    x2 = exp(x[4] * t);
    ha[4] = x1 * t;
    ha[8] = x2 * t;
/* Computing 2nd power */
    d__1 = t;
    ha[6] = x[1] * x1 * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t;
    ha[10] = x[2] * x2 * (d__1 * d__1);
    return 0;
L200:
    t = empr06_1.y[*ka - 1];
    x1 = t + x[2] + 1. / (x[3] * t + x[4]);
    if (x1 == 0.) {
	x1 = 1e-30;
    }
    x2 = x1 / ((t + 1.) * empr06_1.y[*ka + 60]);
    x3 = x[3] * t + x[4];
    if (x3 == 0.) {
	x3 = 1e-30;
    }
    ha[1] = 0.;
    d__1 = abs(x2);
    d__2 = t + .5;
    ha[2] = pow_dd(&d__1, &d__2) * (t + .5) / x1;
    beta = x[1] * ha[2] / x1;
    ha[3] = beta * (t - .5);
    ha[7] = -ha[2] / (x3 * x3);
    ha[4] = ha[7] * t;
    ha[8] = -ha[3] / (x3 * x3);
    ha[5] = ha[8] * t;
/* Computing 4th power */
    d__1 = x3, d__1 *= d__1;
    ha[10] = beta * (t + 1.5 + (x3 + x3) * (t + x[2])) / (d__1 * d__1);
    ha[6] = ha[10] * t * t;
    ha[9] = ha[10] * t;
    return 0;
L30:
    t = (doublereal) (*ka - 1) * .1 - 1.;
    ha[1] = 0.;
    ha[2] = 0.;
    ha[3] = 0.;
    x1 = x[1] + t * x[2];
    x2 = 1. / (t * (x[3] + t * (x[4] + t * x[5])) + 1.);
    x3 = -t * x2 * x2;
    x4 = t * -2. * x1 * x2 * x3;
    ha[4] = x3;
    ha[5] = ha[4] * t;
    ha[7] = ha[5];
    ha[8] = ha[7] * t;
    ha[11] = ha[8];
    ha[12] = ha[11] * t;
    ha[6] = x4;
    ha[9] = ha[6] * t;
    ha[10] = ha[9] * t;
    ha[13] = ha[10];
    ha[14] = ha[13] * t;
    ha[15] = ha[14] * t;
    return 0;
L130:
    t = empr06_1.y[*ka - 1];
    for (i__ = 1; i__ <= 6; ++i__) {
	ha[i__] = 0.;
/* L131: */
    }
    x1 = 1. / (1. - t * (x[4] - t * x[5]));
    x2 = x[1] + t * (x[2] + t * x[3]);
    x3 = x1 * x1;
    ha[7] = -t * x3;
    ha[8] = ha[7] * t;
    ha[9] = ha[8] * t;
    ha[10] = t * 2. * t * x1 * x2 * x3;
    ha[11] = ha[8];
    ha[12] = ha[11] * t;
    ha[13] = ha[12] * t;
    ha[14] = ha[10] * t;
    ha[15] = ha[14] * t;
    return 0;
L100:
    t = (doublereal) (*ka - 1) * .1;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L101: */
    }
    x1 = exp(-x[2] * t);
    x2 = cos(x[3] * t + x[4]);
    x3 = sin(x[3] * t + x[4]);
    x4 = exp(-x[6] * t);
    ha[2] = -x1 * x2 * t;
    ha[10] = -x1 * x2 * x[1];
    ha[9] = ha[10] * t;
    ha[3] = -ha[9] * t;
    ha[7] = -x1 * x3;
    ha[4] = ha[7] * t;
    ha[8] = x1 * x3 * x[1] * t;
    ha[5] = ha[8] * t;
    ha[6] = -ha[3];
    ha[20] = -x4 * t;
    ha[21] = -ha[20] * x[5] * t;
    return 0;
L40:
    beta = empr06_1.y[*ka - 1] * 1.5707963267948966;
    for (i__ = 1; i__ <= 3; ++i__) {
	j = i__ + i__;
	xa[i__ - 1] = x[j - 1];
	xb[i__ - 1] = x[j];
/* L41: */
    }
    ca[3].r = 1., ca[3].i = 0.;
    z__1.r = ca[3].r * 10., z__1.i = ca[3].i * 10.;
    cb[3].r = z__1.r, cb[3].i = z__1.i;
    for (j = 1; j <= 3; ++j) {
	i__ = 4 - j;
	xc[i__ - 1] = beta * xa[i__ - 1];
	t = xc[i__ - 1];
	x1 = cos(t);
	x2 = sin(t);
	z__1.r = x1, z__1.i = 0.;
	c1.r = z__1.r, c1.i = z__1.i;
	d__1 = x2 * xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c2.r = z__1.r, c2.i = z__1.i;
	d__1 = x2 / xb[i__ - 1];
	z__1.r = 0., z__1.i = d__1;
	c3.r = z__1.r, c3.i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c1.r * cb[i__2].r - c1.i * cb[i__2].i, z__2.i = c1.r * cb[
		i__2].i + c1.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c2.r * ca[i__3].r - c2.i * ca[i__3].i, z__3.i = c2.r * ca[
		i__3].i + c2.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	cb[i__1].r = z__1.r, cb[i__1].i = z__1.i;
	i__1 = i__ - 1;
	i__2 = i__;
	z__2.r = c3.r * cb[i__2].r - c3.i * cb[i__2].i, z__2.i = c3.r * cb[
		i__2].i + c3.i * cb[i__2].r;
	i__3 = i__;
	z__3.r = c1.r * ca[i__3].r - c1.i * ca[i__3].i, z__3.i = c1.r * ca[
		i__3].i + c1.i * ca[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	ca[i__1].r = z__1.r, ca[i__1].i = z__1.i;
/* L42: */
    }
    z__1.r = -ca[0].r, z__1.i = -ca[0].i;
    c1.r = z__1.r, c1.i = z__1.i;
    z__1.r = cb[0].r - c1.r, z__1.i = cb[0].i - c1.i;
    c2.r = z__1.r, c2.i = z__1.i;
    z__3.r = c1.r * 2., z__3.i = c1.i * 2.;
    z_div(&z__2, &z__3, &c2);
    z__1.r = z__2.r + 1., z__1.i = z__2.i;
    c3.r = z__1.r, c3.i = z__1.i;
    fa = z_abs(&c3);
    d_cnjg(&z__1, &c3);
    c3.r = z__1.r, c3.i = z__1.i;
    z__2.r = c2.r * c2.r - c2.i * c2.i, z__2.i = c2.r * c2.i + c2.i * c2.r;
    z_div(&z__1, &c_b168, &z__2);
    c1.r = z__1.r, c1.i = z__1.i;
    t = beta;
    for (i__ = 1; i__ <= 3; ++i__) {
	st[i__ - 1] = sin(xc[i__ - 1]);
	ct[i__ - 1] = cos(xc[i__ - 1]);
	i__1 = i__ + i__ - 1;
	i__2 = i__ - 1;
	i__3 = i__ - 1;
	z__3.r = cb[i__2].r * ca[i__3].r - cb[i__2].i * ca[i__3].i, z__3.i = 
		cb[i__2].r * ca[i__3].i + cb[i__2].i * ca[i__3].r;
	i__4 = i__;
	i__5 = i__;
	z__4.r = cb[i__4].r * ca[i__5].r - cb[i__4].i * ca[i__5].i, z__4.i = 
		cb[i__4].r * ca[i__5].i + cb[i__4].i * ca[i__5].r;
	z__2.r = z__3.r - z__4.r, z__2.i = z__3.i - z__4.i;
	i__6 = i__ - 1;
	z__1.r = z__2.r / xb[i__6], z__1.i = z__2.i / xb[i__6];
	cc[i__1].r = z__1.r, cc[i__1].i = z__1.i;
	i__1 = i__ + i__ - 2;
	i__2 = i__ - 1;
	i__3 = i__;
	z__4.r = cb[i__2].r * ca[i__3].r - cb[i__2].i * ca[i__3].i, z__4.i = 
		cb[i__2].r * ca[i__3].i + cb[i__2].i * ca[i__3].r;
	i__4 = i__;
	i__5 = i__ - 1;
	z__5.r = cb[i__4].r * ca[i__5].r - cb[i__4].i * ca[i__5].i, z__5.i = 
		cb[i__4].r * ca[i__5].i + cb[i__4].i * ca[i__5].r;
	z__3.r = z__4.r - z__5.r, z__3.i = z__4.i - z__5.i;
	z__2.r = t * z__3.r, z__2.i = t * z__3.i;
	i__6 = i__ - 1;
	z__1.r = z__2.r / st[i__6], z__1.i = z__2.i / st[i__6];
	cc[i__1].r = z__1.r, cc[i__1].i = z__1.i;
/* L43: */
    }
    for (i__ = 1; i__ <= 6; ++i__) {
	z__2.r = c3.r * c1.r - c3.i * c1.i, z__2.i = c3.r * c1.i + c3.i * 
		c1.r;
	i__1 = i__ - 1;
	z__1.r = z__2.r * cc[i__1].r - z__2.i * cc[i__1].i, z__1.i = z__2.r * 
		cc[i__1].i + z__2.i * cc[i__1].r;
	ga[i__ - 1] = z__1.r / fa;
/* L44: */
    }
    ci.r = 0., ci.i = 1.;
    x2 = x[2];
    x4 = x[4];
    x3 = x[6];
    z__3.r = c1.r + c1.r, z__3.i = c1.i + c1.i;
    z__2.r = -z__3.r, z__2.i = -z__3.i;
    z_div(&z__1, &z__2, &c2);
    c2.r = z__1.r, c2.i = z__1.i;
    d__1 = -st[0] * x2;
    z__1.r = d__1, z__1.i = ct[0];
    s1.r = z__1.r, s1.i = z__1.i;
    d__1 = st[0] / x2;
    d__2 = -ct[0];
    z__1.r = d__1, z__1.i = d__2;
    s2.r = z__1.r, s2.i = z__1.i;
    z__2.r = ct[1] * s1.r, z__2.i = ct[1] * s1.i;
    z__5.r = ci.r * s2.r - ci.i * s2.i, z__5.i = ci.r * s2.i + ci.i * s2.r;
    z__4.r = st[1] * z__5.r, z__4.i = st[1] * z__5.i;
    z__3.r = x4 * z__4.r, z__3.i = x4 * z__4.i;
    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
    s3.r = z__1.r, s3.i = z__1.i;
    z__2.r = ct[1] * s2.r, z__2.i = ct[1] * s2.i;
    z__5.r = ci.r * s1.r - ci.i * s1.i, z__5.i = ci.r * s1.i + ci.i * s1.r;
    z__4.r = st[1] * z__5.r, z__4.i = st[1] * z__5.i;
    z__3.r = z__4.r / x4, z__3.i = z__4.i / x4;
    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
    s4.r = z__1.r, s4.i = z__1.i;
    z__3.r = t * ci.r, z__3.i = t * ci.i;
    z__5.r = cb[0].r / x2, z__5.i = cb[0].i / x2;
    z__6.r = x2 * ca[0].r, z__6.i = x2 * ca[0].i;
    z__4.r = z__5.r + z__6.r, z__4.i = z__5.i + z__6.i;
    z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i = z__3.r * z__4.i + 
	    z__3.i * z__4.r;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[0].r = z__1.r, dd[0].i = z__1.i;
    d__1 = -st[0];
    z__3.r = d__1 * ci.r, z__3.i = d__1 * ci.i;
    d__2 = x2 * x2;
    z__5.r = cb[1].r / d__2, z__5.i = cb[1].i / d__2;
    z__4.r = z__5.r - ca[1].r, z__4.i = z__5.i - ca[1].i;
    z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i = z__3.r * z__4.i + 
	    z__3.i * z__4.r;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[1].r = z__1.r, dd[1].i = z__1.i;
    z__5.r = cb[1].r * s1.r - cb[1].i * s1.i, z__5.i = cb[1].r * s1.i + cb[1]
	    .i * s1.r;
    z__4.r = z__5.r / x4, z__4.i = z__5.i / x4;
    z__7.r = ca[1].r * s2.r - ca[1].i * s2.i, z__7.i = ca[1].r * s2.i + ca[1]
	    .i * s2.r;
    z__6.r = x4 * z__7.r, z__6.i = x4 * z__7.i;
    z__3.r = z__4.r - z__6.r, z__3.i = z__4.i - z__6.i;
    z__2.r = t * z__3.r, z__2.i = t * z__3.i;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[2].r = z__1.r, dd[2].i = z__1.i;
    d__1 = -st[1];
    z__5.r = cb[2].r * s1.r - cb[2].i * s1.i, z__5.i = cb[2].r * s1.i + cb[2]
	    .i * s1.r;
    d__2 = x4 * x4;
    z__4.r = z__5.r / d__2, z__4.i = z__5.i / d__2;
    z__6.r = ca[2].r * s2.r - ca[2].i * s2.i, z__6.i = ca[2].r * s2.i + ca[2]
	    .i * s2.r;
    z__3.r = z__4.r + z__6.r, z__3.i = z__4.i + z__6.i;
    z__2.r = d__1 * z__3.r, z__2.i = d__1 * z__3.i;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[3].r = z__1.r, dd[3].i = z__1.i;
    z__5.r = cb[2].r * s3.r - cb[2].i * s3.i, z__5.i = cb[2].r * s3.i + cb[2]
	    .i * s3.r;
    z__4.r = z__5.r / x3, z__4.i = z__5.i / x3;
    z__7.r = ca[2].r * s4.r - ca[2].i * s4.i, z__7.i = ca[2].r * s4.i + ca[2]
	    .i * s4.r;
    z__6.r = x3 * z__7.r, z__6.i = x3 * z__7.i;
    z__3.r = z__4.r - z__6.r, z__3.i = z__4.i - z__6.i;
    z__2.r = t * z__3.r, z__2.i = t * z__3.i;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[4].r = z__1.r, dd[4].i = z__1.i;
    d__1 = -st[2];
    z__5.r = s3.r * 10., z__5.i = s3.i * 10.;
    d__2 = x3 * x3;
    z__4.r = z__5.r / d__2, z__4.i = z__5.i / d__2;
    z__3.r = z__4.r + s4.r, z__3.i = z__4.i + s4.i;
    z__2.r = d__1 * z__3.r, z__2.i = d__1 * z__3.i;
    z__1.r = c2.r * z__2.r - c2.i * z__2.i, z__1.i = c2.r * z__2.i + c2.i * 
	    z__2.r;
    dd[5].r = z__1.r, dd[5].i = z__1.i;
    l = 0;
    for (i__ = 1; i__ <= 6; ++i__) {
	l = l + i__ - 1;
	i__1 = i__;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = i__ - 1;
	    i__3 = j - 1;
	    z__3.r = c3.r * dd[i__3].r - c3.i * dd[i__3].i, z__3.i = c3.r * 
		    dd[i__3].i + c3.i * dd[i__3].r;
	    i__4 = j - 1;
	    z__6.r = c1.r * cc[i__4].r - c1.i * cc[i__4].i, z__6.i = c1.r * 
		    cc[i__4].i + c1.i * cc[i__4].r;
	    d_cnjg(&z__5, &z__6);
	    z__4.r = c1.r * z__5.r - c1.i * z__5.i, z__4.i = c1.r * z__5.i + 
		    c1.i * z__5.r;
	    z__2.r = z__3.r + z__4.r, z__2.i = z__3.i + z__4.i;
	    z__1.r = cc[i__2].r * z__2.r - cc[i__2].i * z__2.i, z__1.i = cc[
		    i__2].r * z__2.i + cc[i__2].i * z__2.r;
	    ha[l + j] = (z__1.r - ga[i__ - 1] * ga[j - 1]) / fa;
/* L45: */
	}
    }
    for (i__ = 1; i__ <= 3; ++i__) {
	j = i__ * (i__ + i__ + 1);
	z__4.r = c3.r * c1.r - c3.i * c1.i, z__4.i = c3.r * c1.i + c3.i * 
		c1.r;
	z__3.r = z__4.r * ci.r - z__4.i * ci.i, z__3.i = z__4.r * ci.i + 
		z__4.i * ci.r;
	z__2.r = t * z__3.r, z__2.i = t * z__3.i;
	i__1 = i__ - 1;
	i__2 = i__ - 1;
	z__6.r = ca[i__1].r * ca[i__2].r - ca[i__1].i * ca[i__2].i, z__6.i = 
		ca[i__1].r * ca[i__2].i + ca[i__1].i * ca[i__2].r;
	i__3 = i__ - 1;
	i__4 = i__ - 1;
	z__8.r = cb[i__3].r / xb[i__4], z__8.i = cb[i__3].i / xb[i__4];
	pow_zi(&z__7, &z__8, &c__2);
	z__5.r = z__6.r + z__7.r, z__5.i = z__6.i + z__7.i;
	z__1.r = z__2.r * z__5.r - z__2.i * z__5.i, z__1.i = z__2.r * z__5.i 
		+ z__2.i * z__5.r;
	ha[j - 1] += z__1.r / fa;
	z__3.r = c3.r * c1.r - c3.i * c1.i, z__3.i = c3.r * c1.i + c3.i * 
		c1.r;
	i__1 = i__ - 1;
	z__2.r = z__3.r / xb[i__1], z__2.i = z__3.i / xb[i__1];
	i__2 = i__ - 1;
	z__6.r = st[i__2] * ci.r, z__6.i = st[i__2] * ci.i;
	i__3 = i__ - 1;
	i__4 = i__;
	z__8.r = ca[i__3].r * ca[i__4].r - ca[i__3].i * ca[i__4].i, z__8.i = 
		ca[i__3].r * ca[i__4].i + ca[i__3].i * ca[i__4].r;
	i__5 = i__ - 1;
	i__6 = i__;
	z__10.r = cb[i__5].r * cb[i__6].r - cb[i__5].i * cb[i__6].i, z__10.i =
		 cb[i__5].r * cb[i__6].i + cb[i__5].i * cb[i__6].r;
	d__1 = xb[i__ - 1] * xb[i__ - 1];
	z__9.r = z__10.r / d__1, z__9.i = z__10.i / d__1;
	z__7.r = z__8.r - z__9.r, z__7.i = z__8.i - z__9.i;
	z__5.r = z__6.r * z__7.r - z__6.i * z__7.i, z__5.i = z__6.r * z__7.i 
		+ z__6.i * z__7.r;
	i__7 = i__ + i__ - 1;
	z__4.r = z__5.r - cc[i__7].r, z__4.i = z__5.i - cc[i__7].i;
	z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * z__4.i 
		+ z__2.i * z__4.r;
	ha[j] += z__1.r / fa;
/* L46: */
    }
    return 0;
L110:
    t = empr06_1.y[*ka + 40];
    beta = empr06_1.y[*ka + 81];
/* Computing 2nd power */
    d__1 = x[1] + (x[2] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[2]) * beta;
    x1 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[3] + (x[4] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[4]) * beta;
    x2 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[5] + (x[6] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[6]) * beta;
    x3 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = x[7] + (x[8] + 1.) * t;
/* Computing 2nd power */
    d__2 = (1. - x[8]) * beta;
    x4 = d__1 * d__1 + d__2 * d__2;
    if (x1 == 0.) {
	x1 = 1e-30;
    }
    if (x2 == 0.) {
	x2 = 1e-30;
    }
    if (x3 == 0.) {
	x3 = 1e-30;
    }
    if (x4 == 0.) {
	x4 = 1e-30;
    }
    fa = sqrt(x1 / x2) * sqrt(x3 / x4);
    ha[37] = fa / x1 * (x[1] + t * (x[2] + 1.));
    ha[38] = fa / x1 * (x[2] + t * 2. * t - 1. + x[1] * t);
    ha[39] = -fa / x2 * (x[3] + t * (x[4] + 1.));
    ha[40] = -fa / x2 * (x[4] + t * 2. * t - 1. + x[3] * t);
    ha[41] = fa / x3 * (x[5] + t * (x[6] + 1.));
    ha[42] = fa / x3 * (x[6] + t * 2. * t - 1. + x[5] * t);
    ha[43] = -fa / x4 * (x[7] + t * (x[8] + 1.));
    ha[44] = -fa / x4 * (x[8] + t * 2. * t - 1. + x[7] * t);
    ha[45] = 0.;
    fa = x[9] * fa;
    for (i__ = 1; i__ <= 8; ++i__) {
/* L111: */
	ga[i__ - 1] = ha[i__ + 36] * x[9];
    }
    for (j = 1; j <= 8; ++j) {
	i__1 = j;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* L112: */
	    ha[(j - 1) * j / 2 + i__] = ga[i__ - 1] * ga[j - 1] / fa;
	}
    }
    ha[1] = fa / x1 - ha[1];
    ha[2] = fa * t / x1 - ha[2];
    ha[3] = fa / x1 - ha[3];
    ha[6] = ha[6] * 3. - fa / x2;
    ha[9] = ha[9] * 3. - fa * t / x2;
    ha[10] = ha[10] * 3. - fa / x2;
    ha[15] = fa / x3 - ha[15];
    ha[20] = fa * t / x3 - ha[20];
    ha[21] = fa / x3 - ha[21];
    ha[28] = ha[28] * 3. - fa / x4;
    ha[35] = ha[35] * 3. - fa * t / x4;
    ha[36] = ha[36] * 3. - fa / x4;
    return 0;
L50:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L51: */
    }
    ha[1] = 2.;
    ha[3] = 10.;
/* Computing 2nd power */
    d__1 = x[3];
    ha[6] = d__1 * d__1 * 12.;
    ha[10] = 6.;
/* Computing 4th power */
    d__1 = x[5], d__1 *= d__1;
    ha[15] = d__1 * d__1 * 300.;
    ha[21] = 14.;
    ha[27] = -4.;
/* Computing 2nd power */
    d__1 = x[7];
    ha[28] = d__1 * d__1 * 12.;
    switch (*ka) {
	case 1:  goto L56;
	case 2:  goto L52;
	case 3:  goto L53;
	case 4:  goto L54;
	case 5:  goto L55;
    }
L52:
    ha[1] += 40.;
/* Computing 2nd power */
    d__1 = x[2];
    ha[3] += d__1 * d__1 * 360.;
    ha[10] += 80.;
    return 0;
L53:
    ha[6] += 200.;
    return 0;
L54:
    ha[3] += 20.;
    ha[21] += 120.;
    return 0;
L55:
    ha[1] += 80.;
    ha[2] += -30.;
    ha[3] += 20.;
    ha[6] += 40.;
L56:
    return 0;
L60:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L61: */
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
	case 1:  goto L69;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L69;
	case 7:  goto L69;
	case 8:  goto L68;
	case 9:  goto L69;
    }
L62:
    ha[1] += 60.;
    ha[3] += 80.;
    ha[6] += 40.;
    return 0;
L63:
    ha[1] += 100.;
    ha[6] += 20.;
    return 0;
L64:
    ha[1] += 10.;
    ha[3] += 40.;
    ha[15] += 60.;
    return 0;
L65:
    ha[1] += 10.;
    ha[2] += -20.;
    ha[3] += 40.;
    return 0;
L68:
    ha[45] += 240.;
L69:
    return 0;
L70:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L71: */
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
	case 1:  goto L69;
	case 2:  goto L62;
	case 3:  goto L63;
	case 4:  goto L64;
	case 5:  goto L65;
	case 6:  goto L69;
	case 7:  goto L69;
	case 8:  goto L68;
	case 9:  goto L69;
	case 10:  goto L69;
	case 11:  goto L72;
	case 12:  goto L73;
	case 13:  goto L74;
	case 14:  goto L75;
	case 15:  goto L76;
	case 16:  goto L77;
	case 17:  goto L78;
	case 18:  goto L79;
    }
L72:
    ha[1] += 20.;
    return 0;
L73:
    ha[91] += 100.;
    return 0;
L74:
    ha[91] += 60.;
    return 0;
L75:
    ha[1] += 280.;
    return 0;
L76:
    ha[3] += 300.;
    return 0;
L77:
    ha[1] += 100.;
/* Computing 2nd power */
    d__1 = x[17];
    ha[153] += d__1 * d__1 * 1080.;
    return 0;
L78:
    ha[1] += 20.;
    return 0;
L79:
    ha[1] += 140.;
    ha[3] += 100.;
    ha[190] += 20.;
    return 0;
L220:
    x1 = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x3 = 1.;
	x4 = x[i__];
	if (i__ == 1) {
	    x3 = 1e-8;
	}
	if (i__ == 4) {
	    x3 = 4.;
	}
	if (i__ == 2 && *ka == 1) {
	    x4 = x[i__] + 2.;
	}
	if (i__ == 2 && *ka == 2) {
	    x4 = x[i__] - 2.;
	}
/* Computing 2nd power */
	d__1 = x4;
	x1 += x3 * (d__1 * d__1);
/* L221: */
    }
    x2 = exp(x1);
    l = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x3 = 2.;
	x4 = x[i__];
	if (i__ == 1) {
	    x3 = 2e-8;
	}
	if (i__ == 4) {
	    x3 = 8.;
	}
	if (i__ == 2 && *ka == 1) {
	    x4 = x[i__] + 2.;
	}
	if (i__ == 2 && *ka == 2) {
	    x4 = x[i__] - 2.;
	}
	i__2 = i__;
	for (j = 1; j <= i__2; ++j) {
	    ++l;
	    x5 = 2.;
	    x6 = x[j];
	    if (j == 1) {
		x5 = 2e-8;
	    }
	    if (j == 4) {
		x5 = 8.;
	    }
	    if (j == 2 && *ka == 1) {
		x6 = x[j] + 2.;
	    }
	    if (j == 2 && *ka == 2) {
		x6 = x[j] - 2.;
	    }
	    ha[l] = x2 * x3 * x4 * x5 * x6;
/* L222: */
	}
	ha[l] += x2 * x3;
/* L223: */
    }
    return 0;
L160:
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L161: */
    }
    l = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	l += i__;
	x1 = (doublereal) (i__ + *ka - 1) * 1.;
/* Computing 2nd power */
	d__1 = x[i__] - sin((doublereal) ((i__ << 1) + *ka - 3));
	x2 = d__1 * d__1;
	ha[l] = x1 * 2. * exp(x2) * (x2 * 2. + 1.);
/* L162: */
    }
    return 0;
L240:
    if (*ka <= 2) {
	i__1 = *n * (*n + 1) / 2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ha[i__] = 0.;
/* L241: */
	}
	if (*ka == 1) {
	} else if (*ka == 2) {
	    ha[1] = -2.;
	}
    } else {
	t = (doublereal) (*ka - 2) / 29.;
	l = 0;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    for (j = 1; j <= i__2; ++j) {
		++l;
		i__3 = i__ + j - 2;
		ha[l] = pow_di(&t, &i__3) * -2.;
/* L242: */
	    }
/* L243: */
	}
    }
    return 0;
L250:
    t = (doublereal) (*ka - 1) * .1;
    i__1 = *n * (*n + 1) / 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ha[i__] = 0.;
/* L251: */
    }
    x1 = exp(-x[5] * t);
/* Computing 2nd power */
    d__1 = t - x[9];
    x2 = exp(-x[6] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = t - x[10];
    x3 = exp(-x[7] * (d__1 * d__1));
/* Computing 2nd power */
    d__1 = t - x[11];
    x4 = exp(-x[8] * (d__1 * d__1));
    ha[11] = x1 * t;
/* Computing 2nd power */
    d__1 = t;
    ha[15] = -x1 * x[1] * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[9];
    ha[17] = x2 * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[9];
    ha[21] = -ha[17] * x[2] * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[10];
    ha[24] = x3 * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[10];
    ha[28] = -ha[24] * x[3] * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[11];
    ha[32] = x4 * (d__1 * d__1);
/* Computing 2nd power */
    d__1 = t - x[11];
    ha[36] = -ha[32] * x[4] * (d__1 * d__1);
    ha[38] = x2 * -2. * x[6] * (t - x[9]);
/* Computing 3rd power */
    d__1 = t - x[9];
    ha[42] = x2 * -2. * x[2] * (t - x[9]) + x2 * 2. * x[2] * x[6] * (d__1 * (
	    d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[6] * (t - x[9]);
    ha[45] = x2 * 2. * x[2] * x[6] - x2 * 4. * x[2] * (d__1 * d__1);
    ha[48] = x3 * -2. * x[7] * (t - x[10]);
/* Computing 3rd power */
    d__1 = t - x[10];
    ha[52] = x3 * -2. * x[3] * (t - x[10]) + x3 * 2. * x[3] * x[7] * (d__1 * (
	    d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[7] * (t - x[10]);
    ha[55] = x3 * 2. * x[3] * x[7] - x3 * 4. * x[3] * (d__1 * d__1);
    ha[59] = x4 * -2. * x[8] * (t - x[11]);
/* Computing 3rd power */
    d__1 = t - x[11];
    ha[63] = x4 * -2. * x[4] * (t - x[11]) + x4 * 2. * x[4] * x[8] * (d__1 * (
	    d__1 * d__1));
/* Computing 2nd power */
    d__1 = x[8] * (t - x[11]);
    ha[66] = x4 * 2. * x[4] * x[8] - x4 * 4. * x[4] * (d__1 * d__1);
    return 0;
} /* tahd06_ */

