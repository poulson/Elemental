/* Copyright (c) 2010, RWTH Aachen University
 * All rights reserved.
 *
 * Copyright (c) 2015 Jack Poulson
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or 
 * without modification, are permitted provided that the following
 * conditions are met:
 *   * Redistributions of source code must retain the above 
 *     copyright notice, this list of conditions and the following
 *     disclaimer.
 *   * Redistributions in binary form must reproduce the above 
 *     copyright notice, this list of conditions and the following 
 *     disclaimer in the documentation and/or other materials 
 *     provided with the distribution.
 *   * Neither the name of the RWTH Aachen University nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL RWTH 
 * AACHEN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE.
 *
 * Coded by Matthias Petschow (petschow@aices.rwth-aachen.de),
 * August 2010, Version 0.6
 *
 * This code was the result of a collaboration between 
 * Matthias Petschow and Paolo Bientinesi. When you use this 
 * code, kindly reference a paper related to this work.
 *
 */

#ifndef PPMRRR_H
#define PPMRRR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <ctype.h>
#include <mpi.h>

#include <pmrrr/definitions/global.h>

/* Parallel computation of all or a subset of eigenvalues and 
 * optionally eigenvectors of a symmetric tridiagonal matrix based on 
 * the algorithm of Multiple Relatively Robust Representations (MRRR). 
 * The routine targets hybrid architectures consisting of multiple SMP 
 * nodes. It also runs in fully distributed mode, with each node 
 * having only one processor, and fully SMP mode, in which case no 
 * message passing is required. The implementation is based on 
 * LAPACK's routine 'dstemr'.
 */

/* Set the number of threads in case PMR_NUM_THREADS is not 
 * specified */
#define DEFAULT_NUM_THREADS 1

/* Call LAPACK's dstemr in every process to compute all desiered 
 * eigenpairs redundantly (and discard the once that would usually 
 * not be computed by the process) if n < DSTEMR_IF_SMALLER; 
 * default: 4 */ 
#define DSTEMR_IF_SMALLER   4

/* Make sure that eigenpairs are sorted globally; if set to false
 * they are in most cases sorted, but it is not double checked and 
 * can therefore not be guaranteed; default: true */
#define ASSERT_SORTED_EIGENPAIRS false

/* Set flag if Rayleigh Quotient Correction should be used, 
 * which is usually faster; default: true */
#define TRY_RQC          true

/* Maximum numver of iterations of inverse iteration;
 * default: 10 */
#define MAXITER            10

/* Set the min. relative gap for an eigenvalue to be considered 
 * well separated, that is a singleton; this is a very important 
 * parameter of the computation; default: 10e-3 */
#define MIN_RELGAP       1e-3

/* Set the maximal allowed element growth for being accepted as 
 * an RRR, that is if max. pivot < MAX_GROWTH * 'spectral diameter'
 * the RRR is accepted; default: 64.0 */
#define MAX_GROWTH         64.0

/* Set how many iterations should be executed to find the root 
 * representation; default: 6 */
#define MAX_TRY_RRR       10

#endif /* End of header file */
