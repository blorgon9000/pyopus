/* dfovec.f -- translated by f2c (version 20100827).
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

static doublereal c_b2 = .25;
static integer c__9 = 9;
static integer c__1 = 1;

/* Subroutine */ int dfovec_(integer *m, integer *n, doublereal *x, 
	doublereal *fvec, integer *nprob)
{
    /* Initialized data */

    static doublereal zero = 0.;
    static doublereal one = 1.;
    static doublereal y[11] = { 4.,2.,1.,.5,.25,.167,.125,.1,.0833,.0714,
	    .0625 };
    static doublereal y1[15] = { .14,.18,.22,.25,.29,.32,.35,.39,.37,.58,.73,
	    .96,1.34,2.1,4.39 };
    static doublereal y2[11] = { .1957,.1947,.1735,.16,.0844,.0627,.0456,
	    .0342,.0323,.0235,.0246 };
    static doublereal y3[16] = { 34780.,28610.,23650.,19630.,16370.,13720.,
	    11540.,9744.,8261.,7030.,6005.,5147.,4427.,3820.,3307.,2872. };
    static doublereal y4[33] = { .844,.908,.932,.936,.925,.908,.881,.85,.818,
	    .784,.751,.718,.685,.658,.628,.603,.58,.558,.538,.522,.506,.49,
	    .478,.467,.457,.448,.438,.431,.424,.42,.414,.411,.406 };
    static doublereal y5[65] = { 1.366,1.191,1.112,1.013,.991,.885,.831,.847,
	    .786,.725,.746,.679,.608,.655,.616,.606,.602,.626,.651,.724,.649,
	    .649,.694,.644,.624,.661,.612,.558,.533,.495,.5,.423,.395,.375,
	    .372,.391,.396,.405,.428,.429,.523,.562,.607,.653,.672,.708,.633,
	    .668,.645,.632,.591,.559,.597,.625,.739,.71,.729,.72,.636,.581,
	    .428,.292,.162,.098,.054 };
    static doublereal y6[8] = { -.69,-.044,-1.57,-1.31,-2.65,2.,-12.6,9.48 };

    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8;

    /* Builtin functions */
    double atan(doublereal), d_sign(doublereal *, doublereal *), sqrt(
	    doublereal), exp(doublereal), sin(doublereal), cos(doublereal), 
	    log(doublereal);
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);

    /* Local variables */
    static integer i__, j;
    static doublereal dx, sum, tmp1, tmp2, tmp3, tmp4, prod, temp;

    /* Fortran I/O blocks */
    static cilist io___20 = { 0, 6, 0, 0, 0 };


/*     ********** */

/*     Subroutine dfovec */

/*     This subroutine specifies the nonlinear benchmark problems in */

/*     Benchmarking Derivative-Free Optimization Algorithms */
/*     Jorge J. More' and Stefan M. Wild */
/*     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009. */

/*     The latest version of this subroutine is always available at */
/*     http://www.mcs.anl.gov/~more/dfo/ */
/*     The authors would appreciate feedback and experiences from numerical */
/*     studies conducted using this subroutine. */

/*     The data file dfo.dat defines suitable values of m and n */
/*     for each problem number nprob. */

/*     The code for the first 18 functions in dfovec is derived */
/*     from the MINPACK-1 subroutine ssqfcn */

/*     The subroutine statement is */

/*       subroutine dfovec(m,n,x,fvec,nprob) */

/*     where */

/*       m and n are positive integer input variables. */
/*         n must not exceed m. */

/*       x is an input array of length n. */

/*       fvec is an output array of length m that contains the nprob */
/*         function evaluated at x. */

/*       nprob is a positive integer input variable which defines the */
/*         number of the problem. nprob must not exceed 22. */

/*     Argonne National Laboratory */
/*     Jorge More' and Stefan Wild. January 2008. */

/*     ********** */
    /* Parameter adjustments */
    --fvec;
    --x;

    /* Function Body */
    if (*nprob == 1) {
/*        Linear function - full rank. */
	sum = zero;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum += x[j];
	}
	temp = sum * 2 / (doublereal) (*m) + one;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = -temp;
	    if (i__ <= *n) {
		fvec[i__] += x[i__];
	    }
	}
    } else if (*nprob == 2) {
/*        Linear function - rank 1. */
	sum = zero;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum += (doublereal) j * x[j];
	}
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = (doublereal) i__ * sum - one;
	}
    } else if (*nprob == 3) {
/*        Linear function - rank 1 with zero columns and rows. */
	sum = zero;
	i__1 = *n - 1;
	for (j = 2; j <= i__1; ++j) {
	    sum += (doublereal) j * x[j];
	}
	i__1 = *m - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = (doublereal) (i__ - 1) * sum - one;
	}
	fvec[*m] = -one;
    } else if (*nprob == 4) {
/*        Rosenbrock function. */
/* Computing 2nd power */
	d__1 = x[1];
	fvec[1] = (x[2] - d__1 * d__1) * 10;
	fvec[2] = one - x[1];
    } else if (*nprob == 5) {
/*        Helical valley function. */
	temp = atan(one) * 8;
	tmp1 = d_sign(&c_b2, &x[2]);
	if (x[1] > zero) {
	    tmp1 = atan(x[2] / x[1]) / temp;
	}
	if (x[1] < zero) {
	    tmp1 = atan(x[2] / x[1]) / temp + .5;
	}
/* Computing 2nd power */
	d__1 = x[1];
/* Computing 2nd power */
	d__2 = x[2];
	tmp2 = sqrt(d__1 * d__1 + d__2 * d__2);
	fvec[1] = (x[3] - tmp1 * 10) * 10;
	fvec[2] = (tmp2 - one) * 10;
	fvec[3] = x[3];
    } else if (*nprob == 6) {
/*        Powell singular function. */
	fvec[1] = x[1] + x[2] * 10;
	fvec[2] = sqrt(5.) * (x[3] - x[4]);
/* Computing 2nd power */
	d__1 = x[2] - x[3] * 2;
	fvec[3] = d__1 * d__1;
/* Computing 2nd power */
	d__1 = x[1] - x[4];
	fvec[4] = sqrt(10.) * (d__1 * d__1);
    } else if (*nprob == 7) {
/*        Freudenstein and Roth function. */
	fvec[1] = x[1] - 13 + ((5 - x[2]) * x[2] - 2) * x[2];
	fvec[2] = x[1] - 29 + ((one + x[2]) * x[2] - 14) * x[2];
    } else if (*nprob == 8) {
/*        Bard function. */
	for (i__ = 1; i__ <= 15; ++i__) {
	    tmp1 = (doublereal) i__;
	    tmp2 = (doublereal) (16 - i__);
	    tmp3 = tmp1;
	    if (i__ > 8) {
		tmp3 = tmp2;
	    }
	    fvec[i__] = y1[i__ - 1] - (x[1] + tmp1 / (x[2] * tmp2 + x[3] * 
		    tmp3));
	}
    } else if (*nprob == 9) {
/*        Kowalik and Osborne function. */
	for (i__ = 1; i__ <= 11; ++i__) {
	    tmp1 = y[i__ - 1] * (y[i__ - 1] + x[2]);
	    tmp2 = y[i__ - 1] * (y[i__ - 1] + x[3]) + x[4];
	    fvec[i__] = y2[i__ - 1] - x[1] * tmp1 / tmp2;
	}
    } else if (*nprob == 10) {
/*        Meyer function. */
	for (i__ = 1; i__ <= 16; ++i__) {
	    temp = (doublereal) i__ * 5 + 45 + x[3];
	    tmp1 = x[2] / temp;
	    tmp2 = exp(tmp1);
	    fvec[i__] = x[1] * tmp2 - y3[i__ - 1];
	}
    } else if (*nprob == 11) {
/*        Watson function. */
	for (i__ = 1; i__ <= 29; ++i__) {
	    temp = (doublereal) i__ / 29;
	    sum = zero;
	    dx = one;
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		sum += (doublereal) (j - 1) * dx * x[j];
		dx = temp * dx;
	    }
	    fvec[i__] = sum;
	    sum = zero;
	    dx = one;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum += dx * x[j];
		dx = temp * dx;
	    }
/* Computing 2nd power */
	    d__1 = sum;
	    fvec[i__] = fvec[i__] - d__1 * d__1 - one;
	}
	fvec[30] = x[1];
/* Computing 2nd power */
	d__1 = x[1];
	fvec[31] = x[2] - d__1 * d__1 - one;
    } else if (*nprob == 12) {
/*        Box 3-dimensional function. */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = (doublereal) i__;
	    tmp1 = temp / 10;
	    fvec[i__] = exp(-tmp1 * x[1]) - exp(-tmp1 * x[2]) + (exp(-temp) - 
		    exp(-tmp1)) * x[3];
	}
    } else if (*nprob == 13) {
/*        Jennrich and Sampson function. */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = (doublereal) i__;
	    fvec[i__] = (one + temp) * 2 - (exp(temp * x[1]) + exp(temp * x[2]
		    ));
	}
    } else if (*nprob == 14) {
/*        Brown and Dennis function. */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = (doublereal) i__ / 5;
	    tmp1 = x[1] + temp * x[2] - exp(temp);
	    tmp2 = x[3] + sin(temp) * x[4] - cos(temp);
/* Computing 2nd power */
	    d__1 = tmp1;
/* Computing 2nd power */
	    d__2 = tmp2;
	    fvec[i__] = d__1 * d__1 + d__2 * d__2;
	}
    } else if (*nprob == 15) {
/*        Chebyquad function. */
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = zero;
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    tmp1 = one;
	    tmp2 = x[j] * 2 - one;
	    temp = tmp2 * 2;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		fvec[i__] += tmp2;
		tmp3 = temp * tmp2 - tmp1;
		tmp1 = tmp2;
		tmp2 = tmp3;
	    }
	}
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] /= *n;
	    if (i__ % 2 == 0) {
/* Computing 2nd power */
		i__2 = i__;
		fvec[i__] += one / (i__2 * i__2 - one);
	    }
	}
    } else if (*nprob == 16) {
/*        Brown almost-linear function. */
	sum = -((doublereal) (*n + 1));
	prod = one;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum += x[j];
	    prod = x[j] * prod;
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = x[i__] + sum;
	}
	fvec[*n] = prod - one;
    } else if (*nprob == 17) {
/*        Osborne 1 function. */
	for (i__ = 1; i__ <= 33; ++i__) {
	    temp = (doublereal) (i__ - 1) * 10;
	    tmp1 = exp(-x[4] * temp);
	    tmp2 = exp(-x[5] * temp);
	    fvec[i__] = y4[i__ - 1] - (x[1] + x[2] * tmp1 + x[3] * tmp2);
	}
    } else if (*nprob == 18) {
/*        Osborne 2 function. */
	for (i__ = 1; i__ <= 65; ++i__) {
	    temp = (doublereal) (i__ - 1) / 10;
	    tmp1 = exp(-x[5] * temp);
/* Computing 2nd power */
	    d__1 = temp - x[9];
	    tmp2 = exp(-x[6] * (d__1 * d__1));
/* Computing 2nd power */
	    d__1 = temp - x[10];
	    tmp3 = exp(-x[7] * (d__1 * d__1));
/* Computing 2nd power */
	    d__1 = temp - x[11];
	    tmp4 = exp(-x[8] * (d__1 * d__1));
	    fvec[i__] = y5[i__ - 1] - (x[1] * tmp1 + x[2] * tmp2 + x[3] * 
		    tmp3 + x[4] * tmp4);
	}
    } else if (*nprob == 19) {
/*        Bdqrtic function. */
	i__1 = *n - 4;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fvec[i__] = x[i__] * -4 + 3;
/* Computing 2nd power */
	    d__1 = x[i__];
/* Computing 2nd power */
	    d__2 = x[i__ + 1];
/* Computing 2nd power */
	    d__3 = x[i__ + 2];
/* Computing 2nd power */
	    d__4 = x[i__ + 3];
/* Computing 2nd power */
	    d__5 = x[*n];
	    fvec[*n - 4 + i__] = d__1 * d__1 + d__2 * d__2 * 2 + d__3 * d__3 *
		     3 + d__4 * d__4 * 4 + d__5 * d__5 * 5;
	}
    } else if (*nprob == 20) {
/*        Cube function. */
	fvec[1] = x[1] - one;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing 3rd power */
	    d__1 = x[i__ - 1];
	    fvec[i__] = (x[i__] - d__1 * (d__1 * d__1)) * 10;
	}
    } else if (*nprob == 21) {
/*        Mancino function. */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sum = zero;
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
/* Computing 2nd power */
		d__1 = x[i__];
		temp = sqrt(d__1 * d__1 + (doublereal) i__ / (doublereal) j);
/* Computing 5th power */
		d__1 = sin(log(temp)), d__2 = d__1, d__1 *= d__1;
/* Computing 5th power */
		d__3 = cos(log(temp)), d__4 = d__3, d__3 *= d__3;
		sum += temp * (d__2 * (d__1 * d__1) + d__4 * (d__3 * d__3));
	    }
/* Computing 3rd power */
	    i__2 = i__ - 50;
	    fvec[i__] = x[i__] * 1400 + i__2 * (i__2 * i__2) + sum;
	}
    } else if (*nprob == 22) {
/*       Heart 8 function. */
	fvec[1] = x[1] + x[2] - y6[0];
	fvec[2] = x[3] + x[4] - y6[1];
	fvec[3] = x[5] * x[1] + x[6] * x[2] - x[7] * x[3] - x[8] * x[4] - y6[
		2];
	fvec[4] = x[7] * x[1] + x[8] * x[2] + x[5] * x[3] + x[6] * x[4] - y6[
		3];
/* Computing 2nd power */
	d__1 = x[5];
/* Computing 2nd power */
	d__2 = x[7];
/* Computing 2nd power */
	d__3 = x[6];
/* Computing 2nd power */
	d__4 = x[8];
	fvec[5] = x[1] * (d__1 * d__1 - d__2 * d__2) - x[3] * 2 * x[5] * x[7] 
		+ x[2] * (d__3 * d__3 - d__4 * d__4) - x[4] * 2 * x[6] * x[8] 
		- y6[4];
/* Computing 2nd power */
	d__1 = x[5];
/* Computing 2nd power */
	d__2 = x[7];
/* Computing 2nd power */
	d__3 = x[6];
/* Computing 2nd power */
	d__4 = x[8];
	fvec[6] = x[3] * (d__1 * d__1 - d__2 * d__2) + x[1] * 2 * x[5] * x[7] 
		+ x[4] * (d__3 * d__3 - d__4 * d__4) + x[2] * 2 * x[6] * x[8] 
		- y6[5];
/* Computing 2nd power */
	d__1 = x[5];
/* Computing 2nd power */
	d__2 = x[7];
/* Computing 2nd power */
	d__3 = x[7];
/* Computing 2nd power */
	d__4 = x[5];
/* Computing 2nd power */
	d__5 = x[6];
/* Computing 2nd power */
	d__6 = x[8];
/* Computing 2nd power */
	d__7 = x[8];
/* Computing 2nd power */
	d__8 = x[6];
	fvec[7] = x[1] * x[5] * (d__1 * d__1 - d__2 * d__2 * 3) + x[3] * x[7] 
		* (d__3 * d__3 - d__4 * d__4 * 3) + x[2] * x[6] * (d__5 * 
		d__5 - d__6 * d__6 * 3) + x[4] * x[8] * (d__7 * d__7 - d__8 * 
		d__8 * 3) - y6[6];
/* Computing 2nd power */
	d__1 = x[5];
/* Computing 2nd power */
	d__2 = x[7];
/* Computing 2nd power */
	d__3 = x[7];
/* Computing 2nd power */
	d__4 = x[5];
/* Computing 2nd power */
	d__5 = x[6];
/* Computing 2nd power */
	d__6 = x[8];
/* Computing 2nd power */
	d__7 = x[8];
/* Computing 2nd power */
	d__8 = x[6];
	fvec[8] = x[3] * x[5] * (d__1 * d__1 - d__2 * d__2 * 3) - x[1] * x[7] 
		* (d__3 * d__3 - d__4 * d__4 * 3) + x[4] * x[6] * (d__5 * 
		d__5 - d__6 * d__6 * 3) - x[2] * x[8] * (d__7 * d__7 - d__8 * 
		d__8 * 3) - y6[7];
    } else {
	s_wsle(&io___20);
	do_lio(&c__9, &c__1, "Parameter nprob > 22 in subroutine dfovec", (
		ftnlen)41);
	e_wsle();
    }
    return 0;
} /* dfovec_ */

