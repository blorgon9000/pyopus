/* dfoxs.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int dfoxs_(integer *n, doublereal *x, integer *nprob, 
	doublereal *factor)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double sqrt(doublereal), log(doublereal), sin(doublereal), cos(doublereal)
	    ;
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);

    /* Local variables */
    static integer i__, j;
    static doublereal sum, temp;

    /* Fortran I/O blocks */
    static cilist io___5 = { 0, 6, 0, 0, 0 };


/*     ********** */

/*     Subroutine dfoxs */

/*     This subroutine specifies the standard starting points for the */
/*     functions defined by subroutine dfovec as used in: */

/*     Benchmarking Derivative-Free Optimization Algorithms */
/*     Jorge J. More' and Stefan M. Wild */
/*     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009. */

/*     The latest version of this subroutine is always available at */
/*     http://www.mcs.anl.gov/~more/dfo/ */
/*     The authors would appreciate feedback and experiences from numerical */
/*     studies conducted using this subroutine. */

/*     The subroutine returns */
/*     in x a multiple (factor) of the standard starting point. */

/*     The subroutine statement is */

/*       subroutine dfoxs(n,x,nprob,factor) */

/*     where */

/*       n is a positive integer input variable. */

/*       x is an output array of length n which contains the standard */
/*         starting point for problem nprob multiplied by factor. */

/*       nprob is a positive integer input variable which defines the */
/*         number of the problem. nprob must not exceed 22. */

/*       factor is an input variable which specifies the multiple of */
/*         the standard starting point. */

/*     Argonne National Laboratory. */
/*     Jorge More' and Stefan Wild. September 2007. */

/*     ********** */
/*     Selection of initial point. */
    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*nprob <= 3) {
/*        Linear function - full rank or rank 1. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = 1.;
	}
    } else if (*nprob == 4) {
/*        Rosenbrock function. */
	x[1] = -1.2;
	x[2] = 1.;
    } else if (*nprob == 5) {
/*        Helical valley function. */
	x[1] = -1.;
	x[2] = 0.;
	x[3] = 0.;
    } else if (*nprob == 6) {
/*        Powell singular function. */
	x[1] = 3.;
	x[2] = -1.;
	x[3] = 0.;
	x[4] = 1.;
    } else if (*nprob == 7) {
/*        Freudenstein and Roth function. */
	x[1] = .5;
	x[2] = -2.;
    } else if (*nprob == 8) {
/*        Bard function. */
	x[1] = 1.;
	x[2] = 1.;
	x[3] = 1.;
    } else if (*nprob == 9) {
/*        Kowalik and Osborne function. */
	x[1] = .25;
	x[2] = .39;
	x[3] = .415;
	x[4] = .39;
    } else if (*nprob == 10) {
/*        Meyer function. */
	x[1] = .02;
	x[2] = 4e3;
	x[3] = 250.;
    } else if (*nprob == 11) {
/*        Watson function. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = .5;
	}
    } else if (*nprob == 12) {
/*        Box 3-dimensional function. */
	x[1] = 0.;
	x[2] = 10.;
	x[3] = 20.;
    } else if (*nprob == 13) {
/*        Jennrich and Sampson function. */
	x[1] = .3;
	x[2] = .4;
    } else if (*nprob == 14) {
/*        Brown and Dennis function. */
	x[1] = 25.;
	x[2] = 5.;
	x[3] = -5.;
	x[4] = -1.;
    } else if (*nprob == 15) {
/*        Chebyquad function. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = j / (doublereal) (*n + 1);
	}
    } else if (*nprob == 16) {
/*        Brown almost-linear function. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = .5;
	}
    } else if (*nprob == 17) {
/*        Osborne 1 function. */
	x[1] = .5;
	x[2] = 1.5;
	x[3] = 1.;
	x[4] = .01;
	x[5] = .02;
    } else if (*nprob == 18) {
/*        Osborne 2 function. */
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
    } else if (*nprob == 19) {
/*        Bdqrtic function. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = 1.;
	}
    } else if (*nprob == 20) {
/*        Cube function. */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    x[j] = .5;
	}
    } else if (*nprob == 21) {
/*        Mancino function. */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sum = 0.;
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		temp = sqrt((doublereal) i__ / (doublereal) j);
/* Computing 5th power */
		d__1 = sin(log(temp)), d__2 = d__1, d__1 *= d__1;
/* Computing 5th power */
		d__3 = cos(log(temp)), d__4 = d__3, d__3 *= d__3;
		sum += temp * (d__2 * (d__1 * d__1) + d__4 * (d__3 * d__3));
	    }
/* Computing 3rd power */
	    i__2 = i__ - 50;
	    x[i__] = (i__2 * (i__2 * i__2) + sum) * -8.710996e-4;
	}
    } else if (*nprob == 22) {
/*        Heart8 function. */
	x[1] = -.3;
	x[2] = -.39;
	x[3] = .3;
	x[4] = -.344;
	x[5] = -1.2;
	x[6] = 2.69;
	x[7] = 1.59;
	x[8] = -1.5;
    } else {
	s_wsle(&io___5);
	do_lio(&c__9, &c__1, "Parameter nprob > 22 in subroutine dfoxs", (
		ftnlen)40);
	e_wsle();
    }
/*     Compute multiple of initial point. */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	x[j] = *factor * x[j];
    }
    return 0;
} /* dfoxs_ */

