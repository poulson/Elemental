/**
	C++ template version of LAPACK routine xerbla.
	Based on C code translated by f2c (version 20061008).
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

namespace pmrrr { namespace lapack {

	/* Subroutine */ 
	int oerbla(const char *srname, int *info)
	{
		
		/* Table of constant values */
		static int c__1 = 1;

		/*  -- LAPACK auxiliary routine (version 3.2) -- */
		/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
		/*     November 2006 */

		/*     .. Scalar Arguments .. */
		/*     .. */

		/*  Purpose */
		/*  ======= */

		/*  OERBLA  is an error handler for the LAPACK routines. */
		/*  It is called by an LAPACK routine if an input parameter has an */
		/*  invalid value.  A message is printed and execution stops. */

		/*  Installers may consider modifying the STOP statement in order to */
		/*  call system-specific exception-handling facilities. */

		/*  Arguments */
		/*  ========= */

		/*  SRNAME  (input) CHARACTER*(*) */
		/*          The name of the routine which called OERBLA. */

		/*  INFO    (input) INT */
		/*          The position of the invalid parameter in the parameter list */
		/*          of the calling routine. */

		/* ===================================================================== */

		/*     .. Intrinsic Functions .. */
		/*     .. */
		/*     .. Executable Statements .. */

			printf("** On entry to %6s, parameter number %2i had an illegal value\n",
				srname, *info);


		/*     End of OERBLA */

		return 0;
	} /* oerbla_ */

}

}
