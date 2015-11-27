/* tnsiec.f -- translated by f2c (version 20100827).
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
static doublereal c_b296 = 1.;
static doublereal c_b311 = 3.5;

/* *********************************************************************** */


/*     Test problems for NonSmooth InEquality Constrained minimization */


/*     TNSIEC includes the following subroutines */

/*     S   XINIT3          Initiation of variables. */
/*     S   FUNC            Computation of the value and the subgradient */
/*                           of the objective function. */
/*     S   CINEQ           Computation of the value and the subgradient */
/*                           of the constraint functions. */


/*     Napsu Karmitsa (2003, inequality constrained version 2006-2007) */

/*     Haarala M., Miettinen K. and Mäkelä M.M.: New Limited Memory */
/*     Bundle Method for Large-Scale Nonsmooth Optimization, Optimization */
/*     Methods and Software, Vol. 19, No. 6, 2004, 673-692. */

/*     Karmitsa N.: Test Problems for Large-Scale Nonsmooth Minimization, */
/*     Reports of the Department of Mathematical Information Technology, */
/*     Series B, Scientific Computing, B 4/2007, University of Jyväskylä, */
/*     Jyväskylä, 2007. */


/* *********************************************************************** */

/*     * SUBROUTINE XINIT3 * */


/*     * Purpose * */

/*     Initiation of variables for inequality constrained minimization. */


/*     * Calling sequence * */

/*     CALL XINIT3(N,MG,X,NEXT,NCONS) */


/*     * Parameters * */

/*     II  N          Number of variables. */
/*     IU  MG         Number of inequality constraints. */
/*     RO  X(N)       Vector of variables. */
/*     RI  NEXT       Problem number. */
/*     RI  NCONS      Constraint number. */


/*     * Problems * */

/*     1.  Generalization of MAXQ. */
/*     2.  Generalization of MXHILB. */
/*     3.  Chained LQ. */
/*     4.  Chained CB3 I. */
/*     5.  Chained CB3 II. */
/*     6.  Number of active faces. */
/*     7.  Nonsmooth generalization of Brown function 2. */
/*     8.  Chained Mifflin 2. */
/*     9.  Chained crescent I. */
/*     10. Chained crescent II. */


/*     * Constraints * */

/*     1.  Modification of Broyden tridiagonal constraints I for */
/*     problems 1,2,6,7,9, and 10. */
/*     1'. Modification of Broyden tridiagonal constraints I for */
/*     problems 3,4,5, and 8. */
/*     2.  Modification of Broyden tridiagonal constraints II for */
/*     problems 1,2,6,7,9, and 10. */
/*     2'. Modification of Broyden tridiagonal constraints II for */
/*     problems 3,4,5, and 8. */
/*     3.  Modification of MAD1 I for problems 1 - 10. */
/*     4.  Modification of MAD1 II for problems 1 - 10. */
/*     5.  Simple modification of MAD1 I for problems 1,2,6,7,9, and 10. */
/*     5'. Simple modification of MAD1 I for problems 3,4,5, and 8. */
/*     6.  Simple modification of MAD1 II for problems 1,2,6,7,9, and 10. */
/*     6'. Simple modification of MAD1 II for problems 3,4,5, and 8. */
/*     7.  Modification of Problem 20 from UFO collection I for */
/*     problems 1 - 10. */
/*     8.  Modification of Problem 20 from UFO collection II for */
/*     problems 1 - 10. */


/*     Napsu Karmitsa (2006-2007) */


/* Subroutine */ int xinit3_(integer *n, integer *mg, doublereal *x, integer *
	next, integer *ncons)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);

    /* Local variables */
    static integer i__, mgtmp;

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 6, 0, 0, 0 };
    static cilist io___4 = { 0, 6, 0, 0, 0 };
    static cilist io___5 = { 0, 6, 0, 0, 0 };
    static cilist io___6 = { 0, 6, 0, 0, 0 };
    static cilist io___7 = { 0, 6, 0, 0, 0 };
    static cilist io___8 = { 0, 6, 0, 0, 0 };
    static cilist io___9 = { 0, 6, 0, 0, 0 };
    static cilist io___10 = { 0, 6, 0, 0, 0 };
    static cilist io___11 = { 0, 6, 0, 0, 0 };
    static cilist io___12 = { 0, 6, 0, 0, 0 };
    static cilist io___13 = { 0, 6, 0, 0, 0 };
    static cilist io___14 = { 0, 6, 0, 0, 0 };
    static cilist io___15 = { 0, 6, 0, 0, 0 };
    static cilist io___16 = { 0, 6, 0, 0, 0 };
    static cilist io___17 = { 0, 6, 0, 0, 0 };
    static cilist io___18 = { 0, 6, 0, 0, 0 };
    static cilist io___19 = { 0, 6, 0, 0, 0 };
    static cilist io___20 = { 0, 6, 0, 0, 0 };
    static cilist io___21 = { 0, 6, 0, 0, 0 };
    static cilist io___22 = { 0, 6, 0, 0, 0 };
    static cilist io___23 = { 0, 6, 0, 0, 0 };
    static cilist io___24 = { 0, 6, 0, 0, 0 };
    static cilist io___25 = { 0, 6, 0, 0, 0 };
    static cilist io___26 = { 0, 6, 0, 0, 0 };
    static cilist io___27 = { 0, 6, 0, 0, 0 };
    static cilist io___28 = { 0, 6, 0, 0, 0 };
    static cilist io___29 = { 0, 6, 0, 0, 0 };
    static cilist io___30 = { 0, 6, 0, 0, 0 };
    static cilist io___31 = { 0, 6, 0, 0, 0 };
    static cilist io___32 = { 0, 6, 0, 0, 0 };
    static cilist io___33 = { 0, 6, 0, 0, 0 };
    static cilist io___34 = { 0, 6, 0, 0, 0 };
    static cilist io___35 = { 0, 6, 0, 0, 0 };
    static cilist io___36 = { 0, 6, 0, 0, 0 };
    static cilist io___37 = { 0, 6, 0, 0, 0 };
    static cilist io___38 = { 0, 6, 0, 0, 0 };
    static cilist io___39 = { 0, 6, 0, 0, 0 };
    static cilist io___40 = { 0, 6, 0, 0, 0 };
    static cilist io___41 = { 0, 6, 0, 0, 0 };
    static cilist io___42 = { 0, 6, 0, 0, 0 };
    static cilist io___43 = { 0, 6, 0, 0, 0 };
    static cilist io___44 = { 0, 6, 0, 0, 0 };
    static cilist io___45 = { 0, 6, 0, 0, 0 };
    static cilist io___46 = { 0, 6, 0, 0, 0 };
    static cilist io___47 = { 0, 6, 0, 0, 0 };
    static cilist io___48 = { 0, 6, 0, 0, 0 };
    static cilist io___49 = { 0, 6, 0, 0, 0 };
    static cilist io___50 = { 0, 6, 0, 0, 0 };
    static cilist io___51 = { 0, 6, 0, 0, 0 };
    static cilist io___52 = { 0, 6, 0, 0, 0 };
    static cilist io___53 = { 0, 6, 0, 0, 0 };
    static cilist io___54 = { 0, 6, 0, 0, 0 };
    static cilist io___55 = { 0, 6, 0, 0, 0 };
    static cilist io___56 = { 0, 6, 0, 0, 0 };
    static cilist io___57 = { 0, 6, 0, 0, 0 };
    static cilist io___58 = { 0, 6, 0, 0, 0 };
    static cilist io___59 = { 0, 6, 0, 0, 0 };
    static cilist io___60 = { 0, 6, 0, 0, 0 };


/*     Scalar Arguments */
/*     Array Arguments */
/*     Local Arguments */
    /* Parameter adjustments */
    --x;

    /* Function Body */
    mgtmp = *mg;
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
    s_wsle(&io___2);
    do_lio(&c__9, &c__1, "Error: Not such a problem.", (ftnlen)26);
    e_wsle();
    *next = -1;
    return 0;

/*     Generalization of MAXQ */

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
    switch (*ncons) {
	case 1:  goto L13;
	case 2:  goto L17;
	case 3:  goto L14;
	case 4:  goto L15;
	case 5:  goto L19;
	case 6:  goto L16;
	case 7:  goto L13;
	case 8:  goto L17;
    }
L13:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint I or P20 from UFO collection I: */
/*     NCONS = 1 or 7. */
    if (*mg + 2 > *n / 2) {
	i__1 = *mg + 2;
	for (i__ = *n / 2 + 1; i__ <= i__1; ++i__) {
	    x[i__] = (doublereal) i__;
/* L18: */
	}
    }
    if (*mg > *n - 2) {
	s_wsle(&io___4);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    return 0;
L14:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I. NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___5);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L15:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II. NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___6);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L16:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II. NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___7);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
	return 0;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L101: */
    }
    return 0;
L17:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II or  P20 from UFO collection II: */
/*     NCONS = 2 or 8. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___8);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L19:
/*     Feasible starting point if the constraint is a simple */
/*     modification of MAD1 I. NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L111: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___9);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Generalization of MXHILB */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L21: */
    }
    switch (*ncons) {
	case 1:  goto L23;
	case 2:  goto L29;
	case 3:  goto L25;
	case 4:  goto L26;
	case 5:  goto L121;
	case 6:  goto L27;
	case 7:  goto L24;
	case 8:  goto L28;
    }
L23:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___10);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    return 0;
L24:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection I: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___11);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L103: */
    }
    return 0;
L25:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I (for 2 constraints): NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___12);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L26:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II (for 4 constraints): NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___13);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L27:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___14);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
	return 0;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L102: */
    }
    return 0;
L28:
/*     Feasible starting point if the constraint is a modification */
/*     of Problem 20 from UFO collection II: NCONS = 8. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L22: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___15);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L29:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II: NCONS = 2. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___16);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L121:
/*     Feasible starting point if the constraint is a simple */
/*     modification of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L122: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___17);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Chained LQ */

L30:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -.5;
/* L31: */
    }
    switch (*ncons) {
	case 1:  goto L33;
	case 2:  goto L38;
	case 3:  goto L35;
	case 4:  goto L36;
	case 5:  goto L37;
	case 6:  goto L39;
	case 7:  goto L34;
	case 8:  goto L38;
    }
L33:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraints I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___18);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L104: */
    }
    return 0;
L34:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___19);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -2.;
/* L105: */
    }
    return 0;
L35:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___20);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L36:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___21);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L37:
/*     Feasible starting point if the constraint is a simple */
/*     modification of MAD1 I: NCONS = 5. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___22);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L39:
/*     Feasible starting point if the constraint is a simple */
/*     modification of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___23);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    return 0;
L38:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II or Problem 20 from UFO */
/*     collection II: NCONS = 2 or 8. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L32: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___24);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Chained CB3 I and II */

L40:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L41: */
    }
    switch (*ncons) {
	case 1:  goto L42;
	case 2:  goto L46;
	case 3:  goto L43;
	case 4:  goto L44;
	case 5:  goto L47;
	case 6:  goto L45;
	case 7:  goto L42;
	case 8:  goto L46;
    }
L42:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint I or Problem 20 from UFO */
/*     collection I: NCONS = 1 or 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___25);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    return 0;
L43:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___26);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L44:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___27);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L45:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___28);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
	return 0;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L106: */
    }
    return 0;
L46:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II or Problem 20 from UFO */
/*     collection II: NCONS = 2 or 8. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___29);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L47:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L48: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___30);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Number of active faces */

L60:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 1.;
/* L61: */
    }
    switch (*ncons) {
	case 1:  goto L63;
	case 2:  goto L69;
	case 3:  goto L65;
	case 4:  goto L66;
	case 5:  goto L161;
	case 6:  goto L67;
	case 7:  goto L64;
	case 8:  goto L68;
    }
L63:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___31);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    return 0;
L64:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection I: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___32);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L107: */
    }
    return 0;
L65:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___33);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L66:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___34);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L67:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___35);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L108: */
    }
    return 0;
L68:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection II: NCONS = 8. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L62: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___36);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L69:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II: NCONS = 2. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___37);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L161:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L162: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___38);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Nonsmooth generalization of Brown function 2 */

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
    switch (*ncons) {
	case 1:  goto L73;
	case 2:  goto L79;
	case 3:  goto L75;
	case 4:  goto L76;
	case 5:  goto L171;
	case 6:  goto L77;
	case 7:  goto L74;
	case 8:  goto L78;
    }
L73:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraints I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___39);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
	x[i__] = -1.;
/* L109: */
    }
    return 0;
L74:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection I: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___40);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L110: */
    }
    return 0;
L75:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___41);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L76:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___42);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L77:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___43);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L112: */
    }
    return 0;
L78:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection II: NCONS = 8. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L72: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___44);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L79:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraints II: NCONS = 2. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___45);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L171:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L172: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___46);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Chained Mifflin 2 */

L80:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L81: */
    }
    switch (*ncons) {
	case 1:  goto L83;
	case 2:  goto L88;
	case 3:  goto L85;
	case 4:  goto L86;
	case 5:  goto L89;
	case 6:  goto L87;
	case 7:  goto L84;
	case 8:  goto L88;
    }
L83:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraints I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___47);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L113: */
    }
    return 0;
L84:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection I: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___48);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -2.;
/* L114: */
    }
    return 0;
L85:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___49);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L86:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___50);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L87:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___51);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L115: */
    }
    return 0;
L88:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraint II or Problem 20 from UFO */
/*     collection II: NCONS = 2 or 8. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L82: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___52);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L89:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 0.;
/* L181: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___53);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;

/*     Chained crescent I and II */

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
    switch (*ncons) {
	case 1:  goto L92;
	case 2:  goto L97;
	case 3:  goto L94;
	case 4:  goto L95;
	case 5:  goto L98;
	case 6:  goto L96;
	case 7:  goto L93;
	case 8:  goto L97;
    }
L92:
/*     Feasible starting point if the constraint is a modification of */
/*     Broyden tridiagonal constraints I: NCONS = 1. */
    if (*mg > *n - 2) {
	s_wsle(&io___54);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = -1.;
/* L116: */
    }
    return 0;
L93:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection I: NCONS = 7. */
    if (*mg > *n - 2) {
	s_wsle(&io___55);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = 2.;
/* L117: */
    }
    return 0;
L94:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 I: NCONS = 3. */
    *mg = 2;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___56);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L95:
/*     Feasible starting point if the constraint is a modification of */
/*     MAD1 II: NCONS = 4. */
    *mg = 4;
    x[1] = -.5;
    x[2] = 1.1;
    if (*mg > mgtmp) {
	s_wsle(&io___57);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L96:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 II: NCONS = 6. */
    if (*mg > *n - 1) {
	s_wsle(&io___58);
	do_lio(&c__9, &c__1, "Error: Too many constraints.", (ftnlen)28);
	e_wsle();
	*next = -1;
    }
    i__1 = *mg + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L118: */
    }
    return 0;
L97:
/*     Feasible starting point if the constraint is a modification of */
/*     Problem 20 from UFO collection II: NCONS = 8. */
    *mg = 1;
    if (*mg > mgtmp) {
	s_wsle(&io___59);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
L98:
/*     Feasible starting point if the constraint is a simple modification */
/*     of MAD1 I: NCONS = 5. */
    *mg = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = .5;
/* L99: */
    }
    if (*mg > mgtmp) {
	s_wsle(&io___60);
	do_lio(&c__9, &c__1, "Error: Not enough space for constraints.", (
		ftnlen)40);
	e_wsle();
	*next = -1;
    }
    return 0;
} /* xinit3_ */

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
/*     4.  Chained CB3 I (convex). */
/*     5.  Chained CB3 II (convex). */
/*     6.  Number of active faces (nonconvex). */
/*     7.  Nonsmooth generalization of Brown function 2 (nonconvex). */
/*     8.  Chained Mifflin 2 (nonconvex). */
/*     9.  Chained crescent I (nonconvex). */
/*     10. Chained crescent II (nonconvex). */


/*     Napsu Haarala (2003) */


/* Subroutine */ int funci_(integer *n, doublereal *x, doublereal *f, 
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
    static cilist io___61 = { 0, 6, 0, 0, 0 };


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
    s_wsle(&io___61);
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
    g[1] = d_sign(&c_b296, f);
    *f = abs(*f);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	temp2 = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    temp2 += x[j] / (doublereal) (i__ + j - 1);
/* L23: */
	}
	g[i__] = d_sign(&c_b296, &temp2);
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
	y = d_sign(&c_b311, &y) + 4.;
	g[i__] = g[i__] + y * x[i__] - 1.;
	g[i__ + 1] = y * x[i__ + 1];
/* L81: */
    }
    return 0;

/*     Chained crescent I (nonconvex) */

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

/*     Chained crescent II (nonconvex) */

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
} /* funci_ */

/* *********************************************************************** */

/*     * SUBROUTINE CINEQ * */


/*     * Purpose * */


/*     Computation of the value and the subgradient of the constraint */
/*     functions. */


/*     * Calling sequence * */

/*     CALL CINEQ(N,MG,X,G,DG,NEXT,NCONS) */


/*     * Parameters * */

/*     II  N          Number of variables. */
/*     II  MG         Number of inequality constraints. */
/*     RI  X(N)       Vector of variables. */
/*     RI  NEXT       Problem number. */
/*     RI  NCONS      Constraint number. */
/*     RO  G(MG)      Values of the inequality constraints. */
/*     RO  DG(N*MG)   Jacobian of the inequality constraints. */


/*     * Problems * */

/*     1.  Generalization of MAXQ. */
/*     2.  Generalization of MXHILB. */
/*     3.  Chained LQ. */
/*     4.  Chained CB3 I. */
/*     5.  Chained CB3 II. */
/*     6.  Number of active faces. */
/*     7.  Nonsmooth generalization of Brown function 2. */
/*     8.  Chained Mifflin 2. */
/*     9.  Chained crescent I. */
/*     10. Chained crescent II. */


/*     * Constraints * */

/*     1.  Modification of Broyden tridiagonal constraints I for */
/*     problems 1,2,6,7,9, and 10. */
/*     1'. Modification of Broyden tridiagonal constraints I for */
/*     problems 3,4,5, and 8. */
/*     2.  Modification of Broyden tridiagonal constraints II for */
/*     problems 1,2,6,7,9, and 10. */
/*     2'. Modification of Broyden tridiagonal constraints II for */
/*     problems 3,4,5, and 8. */
/*     3.  Modification of MAD1 I for problems 1 - 10. */
/*     4.  Modification of MAD1 II for problems 1 - 10. */
/*     5.  Simple modification of MAD1 I for problems 1,2,6,7,9, and 10. */
/*     5'. Simple modification of MAD1 I for problems 3,4,5, and 8. */
/*     6.  Simple modification of MAD1 II for problems 1,2,6,7,9, and 10. */
/*     6'. Simple modification of MAD1 II for problems 3,4,5, and 8. */
/*     7.  Modification of Problem 20 from UFO collection I for */
/*     problems 1 - 10. */
/*     8.  Modification of Problem 20 from UFO collection II for */
/*     problems 1 - 10. */


/*     Napsu Karmitsa (2006-2007) */


/* Subroutine */ int cineq_(integer *n, integer *mg, doublereal *x, 
	doublereal *g, doublereal *dg, integer *next, integer *ncons)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    double sin(doublereal), cos(doublereal);

    /* Local variables */
    static integer i__, hit;
    static doublereal tmp1;

    /* Fortran I/O blocks */
    static cilist io___75 = { 0, 6, 0, 0, 0 };


/*     Scalar Arguments */
/*     Array Arguments */
/*      DOUBLE PRECISION X(N),G(MG),DG(N*MG) */
/*     Local Arguments */
/*     Intrinsic Functions */
    /* Parameter adjustments */
    --dg;
    --g;
    --x;

    /* Function Body */
    i__1 = *n * *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dg[i__] = 0.;
/* L1: */
    }
    switch (*ncons) {
	case 1:  goto L10;
	case 2:  goto L90;
	case 3:  goto L40;
	case 4:  goto L50;
	case 5:  goto L100;
	case 6:  goto L60;
	case 7:  goto L30;
	case 8:  goto L80;
    }
    s_wsle(&io___75);
    do_lio(&c__9, &c__1, "Error: Not such a problem.", (ftnlen)26);
    e_wsle();
    *next = -1;
    return 0;

/*     Modification of Broyden tridiagonal constraint I. */

L10:
    switch (*next) {
	case 1:  goto L11;
	case 2:  goto L11;
	case 3:  goto L20;
	case 4:  goto L20;
	case 5:  goto L20;
	case 6:  goto L11;
	case 7:  goto L11;
	case 8:  goto L20;
	case 9:  goto L11;
	case 10:  goto L11;
    }
L11:
/*     MG <= N-2 */
/*     for problems 1,2,6,7,9, and 10. */
    i__1 = *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__] = (3. - x[i__ + 1] * 2.) * x[i__ + 1] - x[i__] - x[i__ + 2] * 
		2. + 1.;
	dg[(i__ - 1) * *n + i__] += -1.;
	dg[(i__ - 1) * *n + i__ + 1] = dg[(i__ - 1) * *n + i__ + 1] + 3. - x[
		i__ + 1] * 4.;
	dg[(i__ - 1) * *n + i__ + 2] += -2.;
/* L12: */
    }
    return 0;
/*     Modification of Broyden tridiagonal constraint I'. */
L20:
/*     MG <= N-2 */
/*     for problems 3,4,5, and 8. */
    i__1 = *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__] = (3. - x[i__ + 1] * 2.) * x[i__ + 1] - x[i__] - x[i__ + 2] * 
		2. + 2.5;
	dg[(i__ - 1) * *n + i__] += -1.;
	dg[(i__ - 1) * *n + i__ + 1] = dg[(i__ - 1) * *n + i__ + 1] + 3. - x[
		i__ + 1] * 4.;
	dg[(i__ - 1) * *n + i__ + 2] += -2.;
/* L21: */
    }
    return 0;

/*     Modification of Problem 20 from UFO collection I. */

L30:
/*     MG <= N-2 */
    i__1 = *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
	g[i__] = (3. - x[i__ + 1] * .5) * x[i__ + 1] - x[i__] - x[i__ + 2] * 
		2. + 1.;
	dg[(i__ - 1) * *n + i__] += -1.;
	dg[(i__ - 1) * *n + i__ + 1] = dg[(i__ - 1) * *n + i__ + 1] + 3. - x[
		i__ + 1] * 1.;
	dg[(i__ - 1) * *n + i__ + 2] += -2.;
/* L31: */
    }
    return 0;

/*     Modification of MAD1 I. */

L40:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    g[1] = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - 1.;
    hit = 1;
    tmp1 = sin(x[1]);
    if (tmp1 > g[1]) {
	g[1] = tmp1;
	hit = 2;
    }
    tmp1 = -cos(x[2]);
    if (tmp1 > g[1]) {
	g[1] = tmp1;
	hit = 3;
    }
    g[2] = -x[1] - x[2] + .5;
    dg[*n + 1] = -1.;
    dg[*n + 2] = -1.;
    switch (hit) {
	case 1:  goto L41;
	case 2:  goto L42;
	case 3:  goto L43;
    }
L41:
    dg[1] = x[1] * 2. + x[2];
    dg[2] = x[2] * 2. + x[1];
    return 0;
L42:
    dg[1] = cos(x[1]);
    return 0;
L43:
    dg[2] = sin(x[2]);
    return 0;

/*     Modification of MAD1 II. */

L50:
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    g[1] = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - 1.;
    dg[1] = x[1] * 2. + x[2];
    dg[2] = x[2] * 2. + x[1];
    g[2] = sin(x[1]);
    dg[*n + 1] = cos(x[1]);
    g[3] = -cos(x[2]);
    dg[(*n << 1) + 2] = sin(x[2]);
    g[4] = -x[1] - x[2] + .5;
    dg[*n * 3 + 1] = -1.;
    dg[*n * 3 + 2] = -1.;
    return 0;

/*     Simple modification of MAD1 II. */

L60:
    switch (*next) {
	case 1:  goto L70;
	case 2:  goto L70;
	case 3:  goto L61;
	case 4:  goto L61;
	case 5:  goto L61;
	case 6:  goto L70;
	case 7:  goto L70;
	case 8:  goto L61;
	case 9:  goto L70;
	case 10:  goto L70;
    }
L61:
/*     for problems 3,4,5, and 8. */
    i__1 = *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
/* Computing 2nd power */
	d__2 = x[i__ + 1];
	g[i__] = d__1 * d__1 + d__2 * d__2 + x[i__] * x[i__ + 1] - 1.;
	dg[(i__ - 1) * *n + i__] = x[i__] * 2. + x[i__ + 1];
	dg[(i__ - 1) * *n + i__ + 1] = x[i__ + 1] * 2. + x[i__];
/* L62: */
    }
    return 0;
/*     Simple modification of MAD1 II'. */
L70:
/*     for problems 1,2,6,7,9, and 10. */
    i__1 = *mg;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
/* Computing 2nd power */
	d__2 = x[i__ + 1];
	g[i__] = d__1 * d__1 + d__2 * d__2 + x[i__] * x[i__ + 1] - x[i__] * 
		2. - x[i__ + 1] * 2. + 1.;
	dg[(i__ - 1) * *n + i__] = x[i__] * 2. + x[i__ + 1] - 2.;
	dg[(i__ - 1) * *n + i__ + 1] = x[i__ + 1] * 2. + x[i__] - 2.;
/* L71: */
    }
    return 0;

/*     Modification of Problem 20 from UFO collection II. */

L80:
    g[1] = (3. - x[2] * .5) * x[2] - x[1] - x[3] * 2. + 1.;
    dg[1] = -1.;
    dg[2] = 3. - x[2] * 1.;
    dg[3] = -2.;
    i__1 = *n - 2;
    for (i__ = 2; i__ <= i__1; ++i__) {
	g[1] = g[1] + (3. - x[i__ + 1] * .5) * x[i__ + 1] - x[i__] - x[i__ + 
		2] * 2. + 1.;
	dg[i__] += -1.;
	dg[i__ + 1] = dg[i__ + 1] + 3. - x[i__ + 1] * 1.;
	dg[i__ + 2] += -2.;
/* L81: */
    }
    return 0;

/*     Modification of Broyden tridiagonal constraint II. */

L90:
    switch (*next) {
	case 1:  goto L91;
	case 2:  goto L91;
	case 3:  goto L93;
	case 4:  goto L93;
	case 5:  goto L93;
	case 6:  goto L91;
	case 7:  goto L91;
	case 8:  goto L93;
	case 9:  goto L91;
	case 10:  goto L91;
    }
L91:
/*     for problems 1,2,6,7,9, and 10. */
    g[1] = (3. - x[2] * 2.) * x[2] - x[1] - x[3] * 2. + 1.;
    dg[1] = -1.;
    dg[2] = 3. - x[2] * 4.;
    dg[3] = -2.;
    i__1 = *n - 2;
    for (i__ = 2; i__ <= i__1; ++i__) {
	g[1] = g[1] + (3. - x[i__ + 1] * 2.) * x[i__ + 1] - x[i__] - x[i__ + 
		2] * 2. + 1.;
	dg[i__] += -1.;
	dg[i__ + 1] = dg[i__ + 1] + 3. - x[i__ + 1] * 4.;
	dg[i__ + 2] += -2.;
/* L92: */
    }
    return 0;
/*     Modification of Broyden tridiagonal constraint II'. */
L93:
/*     for problems 3,4,5, and 8. */
    g[1] = (3. - x[2] * 2.) * x[2] - x[1] - x[3] * 2. + 2.5;
    dg[1] = -1.;
    dg[2] = 3. - x[2] * 4.;
    dg[3] = -2.;
    i__1 = *n - 2;
    for (i__ = 2; i__ <= i__1; ++i__) {
	g[1] = g[1] + (3. - x[i__ + 1] * 2.) * x[i__ + 1] - x[i__] - x[i__ + 
		2] * 2. + 2.5;
	dg[i__] += -1.;
	dg[i__ + 1] = dg[i__ + 1] + 3. - x[i__ + 1] * 4.;
	dg[i__ + 2] += -2.;
/* L94: */
    }
    return 0;

/*     Simple modification of MAD1 I. */

L100:
    switch (*next) {
	case 1:  goto L110;
	case 2:  goto L110;
	case 3:  goto L111;
	case 4:  goto L111;
	case 5:  goto L111;
	case 6:  goto L110;
	case 7:  goto L110;
	case 8:  goto L111;
	case 9:  goto L110;
	case 10:  goto L110;
    }
L111:
/*     for problems 3,4,5, and 8. */
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    g[1] = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - 1.;
    dg[1] = x[1] * 2. + x[2];
    dg[2] = x[2] * 2. + x[1];
    i__1 = *n - 1;
    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
/* Computing 2nd power */
	d__2 = x[i__ + 1];
	g[1] = g[1] + d__1 * d__1 + d__2 * d__2 + x[i__] * x[i__ + 1] - 1.;
	dg[i__] = dg[i__] + x[i__] * 2. + x[i__ + 1];
	dg[i__ + 1] = dg[i__ + 1] + x[i__ + 1] * 2. + x[i__];
/* L112: */
    }
    return 0;
/*     Simple modification of MAD1 I'. */
L110:
/*     for problems 1,2,6,7,9, and 10. */
/* Computing 2nd power */
    d__1 = x[1];
/* Computing 2nd power */
    d__2 = x[2];
    g[1] = d__1 * d__1 + d__2 * d__2 + x[1] * x[2] - x[1] * 2. - x[2] * 2. + 
	    1.;
    dg[1] = x[1] * 2. + x[2] - 2.;
    dg[2] = x[2] * 2. + x[1] - 2.;
    i__1 = *n - 1;
    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = x[i__];
/* Computing 2nd power */
	d__2 = x[i__ + 1];
	g[1] = g[1] + d__1 * d__1 + d__2 * d__2 + x[i__] * x[i__ + 1] - x[i__]
		 * 2. - x[i__ + 1] * 2. + 1.;
	dg[i__] = dg[i__] + x[i__] * 2. + x[i__ + 1] - 2.;
	dg[i__ + 1] = dg[i__ + 1] + x[i__ + 1] * 2. + x[i__] - 2.;
/* L113: */
    }
    return 0;
} /* cineq_ */

