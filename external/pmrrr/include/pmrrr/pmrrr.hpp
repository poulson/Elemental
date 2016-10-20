/* Computation of eigenvalues and eigenvectors of a symmetric
 * tridiagonal matrix T, given by its diagonal elements D
 * and its super-/subdiagonal elements E.
 *
 * See INCLUDE/pmrrr.h for more information.
 *
 * Copyright (c) 2010, RWTH Aachen University
 * All rights reserved.
 *
 * Copyright (c) 2015, Jack Poulson
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

#ifndef __PMRRR_HPP__
#define __PMRRR_HPP__

#include <pmrrr/definitions/pmrrr.h>
#include <pmrrr/definitions/global.h>
#include <pmrrr/definitions/structs.h>
#include <pmrrr/plarrv.hpp>
#include <pmrrr/plarre.hpp>
#include <pmrrr/lapack/odstmr.hpp>
#include <pmrrr/lapack/odrrr.hpp>
#include <pmrrr/lapack/odnst.hpp>
#include <pmrrr/lapack/odrrj.hpp>
#include <pmrrr/blas/odscal.hpp>

using std::sort;

/*
 * Computation of eigenvalues and eigenvectors of a symmetric
 * tridiagonal matrix T, given by its diagonal elements D
 * and its super-/subdiagonal elements E.
 * See README or 'pmrrr.h' for details.
 */
namespace pmrrr {

	namespace detail{
		template<typename FloatingType>
		FloatingType scale_matrix(in_t<FloatingType> *Dstruct, val_t<FloatingType> *Wstruct, bool valeig);

		template<typename FloatingType>
		int handle_small_cases(char *jobz, char *range, int *np, FloatingType  *D,
				   FloatingType *E, FloatingType *vlp, FloatingType *vup, int *ilp,
				   int *iup, int *tryracp, MPI_Comm comm, int *nzp,
				   int *myfirstp, FloatingType *W, FloatingType *Z, int *ldzp,
				   int *Zsupp);

		template<typename FloatingType>
		bool cmp(const FloatingType & a1, const FloatingType & a2);

		/*
		 * Template template parameter required because of the sort function not catching overload properly.
		 */
		template<typename FloatingType>
		bool cmp_sort_struct(const sort_struct_t<FloatingType> & a1, const sort_struct_t<FloatingType> & a2);


		template<typename FloatingType>
		void clean_up(MPI_Comm comm, FloatingType *Werr, FloatingType *Wgap,
			  FloatingType *gersch, int *iblock, int *iproc,
			  int *Windex, int *isplit, int *Zindex,
			  proc_t *procinfo, in_t<FloatingType> *Dstruct,
			  val_t<FloatingType> *Wstruct, vec_t<FloatingType> *Zstruct,
			  tol_t<FloatingType> *tolstruct);

		template<typename FloatingType>
		int sort_eigenpairs(proc_t *procinfo, val_t<FloatingType> *Wstruct, vec_t<FloatingType> *Zstruct);

		template<typename FloatingType>
		int refine_to_highrac(proc_t *procinfo, char *jobz, FloatingType *D,
					  FloatingType *E2, in_t<FloatingType> *Dstruct, int *nzp,
					  val_t<FloatingType> *Wstruct, tol_t<FloatingType> *tolstruct);

		template<typename FloatingType>
		void invscale_eigenvalues(val_t<FloatingType> *Wstruct, FloatingType scale,
					  int size);
	}

    /* Arguments:
     * ----------
     *
     * INPUTS: 
     * -------
     * jobz              "N" or "n" - compute only eigenvalues
     *                   "V" or "v" - compute also eigenvectors
     *                   "C" or "c" - count the maximal number of 
     *                                locally computed eigenvectors
     * range             "A" or "a" - all
     *                   "V" or "v" - by interval: (VL,VU]
     *                   "I" or "i" - by index:     IL-IU
     * n                 matrix size
     * ldz               must be set on input to the leading dimension 
     *                   of of eigenvector matrix Z; this is often equal 
     *                   to matrix size n (not changed on output)
     *
     * INPUT + OUTPUT: 
     * ---------------
     * D (double[n])     Diagonal elements of tridiagonal T.
     *                   (On output the array will be overwritten).
     * E (double[n])     Off-diagonal elements of tridiagonal T.
     *                   First n-1 elements contain off-diagonals,
     *                   the last element can have an abitrary value. 
     *                   (On output the array will be overwritten.)
     * vl                If range="V", lower bound of interval
     *                   (vl,vu], on output refined.
     *                   If range="A" or "I" not referenced as input.
     *                   On output the interval (vl,vu] contains ALL
     *                   the computed eigenvalues.
     * vu                If range="V", upper bound of interval
     *                   (vl,vu], on output refined.
     *                   If range="A" or "I" not referenced as input.
     *                   On output the interval (vl,vu] contains ALL
     *                   the computed eigenvalues.
     * il                If range="I", lower index (1-based indexing) of 
     *                   the subset 'il' to 'iu'.
     *                   If range="A" or "V" not referenced as input.
     *                   On output the eigenvalues with index il to iu are 
     *                   computed by ALL processes.
     * iu                If range="I", upper index (1-based indexing) of 
     *                   the subset 'il' to 'iu'.
     *                   If range="A" or "V" not referenced as input.
     *                   On output the eigenvalues with index il to iu are 
     *                   computed by ALL processes.
     * tryrac            0 - do not try to achieve high relative accuracy.
     *                   NOTE: this should be the default in context of  
     *                         dense eigenproblems.
     *                   1 - relative accuracy will be attempted; 
     *                       on output it is set to zero if high relative 
     *                       accuracy is not achieved.
     * comm              MPI communicator; commonly: MPI_COMM_WORLD.
     *
     * OUTPUT: 
     * -------
     * nz                Number of eigenvalues and eigenvectors computed 
     *                   locally.
     *                   If jobz="C", 'nz' will be set to the maximal
     *                   number of locally computed eigenvectors such 
     *                   that double[n*nz] will provide enough memory 
     *                   for the local eigenvectors;  this is only 
     *                   important in case of range="V" since 
     *                   '#eigenpairs' are not known in advance
     * offset            Index, relative to the computed eigenvalues, of 
     *                   the smallest eigenvalue computed locally
     *                   (0-based indexing).
     * W (double[n])     Locally computed eigenvalues;
     *                   The first nz entries contain the eigenvalues 
     *                   computed locally; the first entry contains the 
     *                   'offset + 1'-th computed eigenvalue, which is the 
     *                   'offset + il'-th eigenvalue of the input matrix 
     *                   (1-based indexing in both cases).
     *                   In some situations it is desirable to have all 
     *                   computed eigenvalues in W, instead of only 
     *                   those computed locally. In this case, call 
     *                   routine 'PMR_comm_eigvals' after 
     *                   'pmrrr' returns (see example and interface below).
     * Z                 Locally computed eigenvectors.
     * (double[n*nz])    Enough space must be provided to store the
     *                   vectors. 'nz' should be bigger or equal 
     *                   to ceil('#eigenpairs'/'#processes'), where 
     *                   '#eigenpairs' is 'n' in case of range="A" and
     *                   'iu-il+1' in case of range="I". Alternatively, 
     *                   and for range="V" 'nz' can be obtained 
     *                   by running the routine with jobz="C". 
     * Zsupp             Support of eigenvectors, which is given by
     * (double[2*n])     i1=Zsupp[2*i] to i2=Zsupp[2*i+1] for the i-th local eigenvector
     *                   (returns 1-based indexing; e.g. in C Z[i1-1:i2-1] are non-zero and
     *                   in Fotran Z(i1:i2) are non-zero).
     *
     * RETURN VALUE: 
     * -------------
     *                 0 - success  
     *                 1 - wrong input parameter
     *                 2 - misc errors  
     *
     * The Fortran interface takes an additinal integer argument INFO
     * to retrieve the return value. 
     * An example call in Fortran looks therefore like
     *
     * CALL PMRRR('V', 'A', N, D, E, VL, VU, IL, IU, TRYRAC, 
     *            MPI_COMM_WORLD, NZ, MYFIRST, W, Z, LDZ, ZSUPP, INFO)
     *
     *
     * EXAMPLE CALL: 
     * -------------
     * char    *jobz, *range;
     * int     n, il, iu, tryRAC=0, nz, offset, ldz, *Zsupp;
     * double  *D, *E, *W, *Z, vl, vu;
     *
     * // allocate space for D, E, W, Z
     * // initialize D, E
     * // set jobz, range, ldz, and if necessary, il, iu or vl, vu  
     * 
     * info = pmrrr(jobz, range, &n, D, E, &vl, &vu, &il, &iu,
     *              &tryRAC, MPI_COMM_WORLD, &nz, &myfirst, W,
     *          Z, &ldz , Zsupp);
     *
     * // optional: 
     * PMR_comm_eigvals(MPI_COMM_WORLD, &nz, &myfirst, W);
     *
     */
    template<typename FloatingType>
    int pmrrr(char *jobz, char *range, int *np, FloatingType  *D,
          FloatingType *E, FloatingType *vl, FloatingType *vu, int *il,
          int *iu, int *tryracp, MPI_Comm comm, int *nzp,
          int *offsetp, FloatingType *W, FloatingType *Z, int *ldz,
          int *Zsupp)
    {
      /* Input parameter */
      int         n      = *np;
      bool onlyW = toupper(jobz[0]) == 'N';
      bool wantZ = toupper(jobz[0]) == 'V';
      bool cntval = toupper(jobz[0]) == 'C';
      bool alleig = toupper(range[0]) == 'A';
      bool valeig = toupper(range[0]) == 'V';
      bool indeig = toupper(range[0]) == 'I';

      /* Check input parameters */
      if(!(onlyW  || wantZ  || cntval)) return 1;
      if(!(alleig || valeig || indeig)) return 1;
      if(n <= 0) return 1;
      if (valeig) {
        if(*vu<=*vl) return 1;
      } else if (indeig) {
        if (*il<1 || *il>n || *iu<*il || *iu>n) return 1;
      }

      /* MPI & multithreading info */
      int is_init, is_final;
      MPI_Initialized(&is_init);
      MPI_Finalized(&is_final);
      if (is_init!=1 || is_final==1) {
        fprintf(stderr, "ERROR: MPI is not active! (init=%d, final=%d) \n",
          is_init, is_final);
        return 1;
      }
      MPI_Comm comm_dup;
      MPI_Comm_dup(comm, &comm_dup);
      int nproc, pid, thread_support;
      MPI_Comm_size(comm_dup, &nproc);
      MPI_Comm_rank(comm_dup, &pid);
      MPI_Query_thread(&thread_support);

      int nthreads;
      if ( !(thread_support == MPI_THREAD_MULTIPLE ||
             thread_support == MPI_THREAD_FUNNELED) ) {
        /* Disable multithreading; note: to support multithreading with 
         * MPI_THREAD_SERIALIZED the code must be changed slightly; this 
         * is not supported at the moment */
        nthreads = 1;
      } else {
        char *ompvar = getenv("PMR_NUM_THREADS");
        if (ompvar == NULL) {
          nthreads = DEFAULT_NUM_THREADS;
        } else {
          nthreads = atoi(ompvar);
        }
      }

    #if defined(MVAPICH2_VERSION)
      if (nthreads>1) {
        int           mv2_affinity=1;
        char        *mv2_string = getenv("MV2_ENABLE_AFFINITY");
        if (mv2_string != NULL) {
          mv2_affinity = atoi(mv2_string);
        }    
        if (mv2_affinity!=0) {
          nthreads = 1;
          if (pid==0) {
            fprintf(stderr, "WARNING: PMRRR incurs a significant performance penalty when multithreaded with MVAPICH2 with affinity enabled. The number of threads has been reduced to one; please rerun with MV2_ENABLE_AFFINITY=0 or PMR_NUM_THREADS=1 in the future.\n");
            fflush(stderr);
          }
        } 
      }
    #endif

      /* If only maximal number of local eigenvectors are queried
       * return if possible here */
      *nzp     = 0;
      *offsetp = 0;
      if (cntval) {
        if ( alleig || n < DSTEMR_IF_SMALLER ) {
          *nzp = iceil(n,nproc);
          MPI_Comm_free(&comm_dup);
          return 0;
        } else if (indeig) {
          *nzp = iceil(*iu-*il+1,nproc);
          MPI_Comm_free(&comm_dup);
          return 0;
        }
      }

      /* Check if computation should be done by multiple processes */
      int info;
      if (n < DSTEMR_IF_SMALLER) {
        info = detail::handle_small_cases(jobz, range, np, D, E, vl, vu, il,
                      iu, tryracp, comm, nzp, offsetp, W,
                      Z, ldz, Zsupp);
        MPI_Comm_free(&comm_dup);
        return info;
      }

      /* Allocate memory */
      FloatingType *Werr = (FloatingType *) malloc( n * sizeof(FloatingType) );
      assert(Werr != NULL);
      FloatingType *Wgap = (FloatingType *) malloc( n * sizeof(FloatingType) );
      assert(Wgap != NULL);
      FloatingType *gersch = (FloatingType *) malloc( 2*n*sizeof(FloatingType) );
      assert(gersch != NULL);
      int *iblock  = (int *) calloc( n , sizeof(int) );
      assert(iblock != NULL);
      int *iproc = (int *)  malloc( n * sizeof(int) );
      assert(iproc != NULL);
      int *Windex = (int *) malloc( n * sizeof(int) );
      assert(Windex != NULL);
      int *isplit = (int *) malloc( n * sizeof(int) );
      assert(isplit != NULL);
      int *Zindex = (int *) malloc( n * sizeof(int) );
      assert(Zindex != NULL);
      detail::proc_t *procinfo = (detail::proc_t *) malloc( sizeof(detail::proc_t) );
      assert(procinfo != NULL);
      detail::in_t<FloatingType> *Dstruct = (detail::in_t<FloatingType> *) malloc( sizeof(detail::in_t<FloatingType>) );
      assert(Dstruct != NULL);
      detail::val_t<FloatingType> *Wstruct = (detail::val_t<FloatingType> *) malloc( sizeof(detail::val_t<FloatingType>) );
      assert(Wstruct != NULL);
      detail::vec_t<FloatingType> *Zstruct = (detail::vec_t<FloatingType> *) malloc( sizeof(detail::vec_t<FloatingType>) );
      assert(Zstruct != NULL);
      detail::tol_t<FloatingType> *tolstruct = (detail::tol_t<FloatingType> *) malloc( sizeof(detail::tol_t<FloatingType>) );
      assert(tolstruct != NULL);

      /* Bundle variables into a structures */
      procinfo->pid            = pid;
      procinfo->nproc          = nproc;
      procinfo->comm           = comm_dup;
      procinfo->nthreads       = nthreads;
      procinfo->thread_support = thread_support;

      Dstruct->n               = n;
      Dstruct->D               = D;
      Dstruct->E               = E;
      Dstruct->isplit          = isplit;

      Wstruct->n               = n;
      Wstruct->vl              = vl;
      Wstruct->vu              = vu;
      Wstruct->il              = il;
      Wstruct->iu              = iu;
      Wstruct->W               = W;
      Wstruct->Werr            = Werr;
      Wstruct->Wgap            = Wgap;
      Wstruct->Windex          = Windex;
      Wstruct->iblock          = iblock;
      Wstruct->iproc           = iproc;
      Wstruct->gersch          = gersch;

      Zstruct->ldz             = *ldz;
      Zstruct->nz              = 0;
      Zstruct->Z               = Z;
      Zstruct->Zsupp           = Zsupp;
      Zstruct->Zindex          = Zindex;

      /* Scale matrix to allowable range, returns 1.0 if not scaled */
      FloatingType scale = detail::scale_matrix<FloatingType>(Dstruct, Wstruct, valeig);

      /*  Test if matrix warrants more expensive computations which
       *  guarantees high relative accuracy */
      if (*tryracp)
        lapack::odrrr(&n, D, E, &info); /* 0 - rel acc */
      else info = -1;

      int i;
      FloatingType *Dcopy, *E2copy;
      if (info == 0) {
        /* This case is extremely rare in practice */ 
        tolstruct->split = std::numeric_limits<FloatingType>::epsilon();
        /* Copy original data needed for refinement later */
        Dcopy  = (FloatingType *) malloc( n * sizeof(FloatingType) );
        assert(Dcopy != NULL);
        memcpy(Dcopy, D, n*sizeof(FloatingType));  
        E2copy = (FloatingType *) malloc( n * sizeof(FloatingType) );
        assert(E2copy != NULL);
        for (i=0; i<n-1; i++)
          E2copy[i] = E[i]*E[i];
      } else {
        /* Neg. threshold forces old splitting criterion */
        tolstruct->split = -std::numeric_limits<FloatingType>::epsilon(); 
        *tryracp = 0;
      }

      if (!wantZ) {
        /* Compute eigenvalues to full precision */
        tolstruct->rtol1 = 4.0 * std::numeric_limits<FloatingType>::epsilon();
        tolstruct->rtol2 = 4.0 * std::numeric_limits<FloatingType>::epsilon();
      } else {
        /* Do not compute to full accuracy first, but refine later */
        tolstruct->rtol1 = sqrt(std::numeric_limits<FloatingType>::epsilon());
        tolstruct->rtol1 = fmin(1e-2*MIN_RELGAP, tolstruct->rtol1);
        tolstruct->rtol2 = sqrt(std::numeric_limits<FloatingType>::epsilon())*5.0E-3;
        tolstruct->rtol2 = fmin(5e-6*MIN_RELGAP, tolstruct->rtol2);
        tolstruct->rtol2 = fmax(4.0 * std::numeric_limits<FloatingType>::epsilon(), tolstruct->rtol2);
      }

      /*  Compute all eigenvalues: sorted by block */
      // TODO: change later the casting
      info = detail::plarre(procinfo, jobz, range, Dstruct, Wstruct, tolstruct, nzp, offsetp);
      assert(info == 0);

      /* If just number of local eigenvectors are queried */
      if (cntval & valeig) {    
        detail::clean_up(comm_dup, Werr, Wgap, gersch, iblock, iproc, Windex,
             isplit, Zindex, procinfo, Dstruct, Wstruct, Zstruct,
             tolstruct);
        return 0;
      }

      /* If only eigenvalues are to be computed */
      if (!wantZ) {

        /* Refine to high relative with respect to input T */
        if (*tryracp) {
          info = detail::refine_to_highrac(procinfo, jobz, Dcopy, E2copy, 
                                        Dstruct, nzp, Wstruct, tolstruct);
          assert(info == 0);
        }

        /* Sort eigenvalues */
        sort(W, W + n, detail::cmp<FloatingType>);

        /* Only keep subset ifirst:ilast */
        int ifirst, ilast, isize;
        int iil = *il;
        int iiu = *iu;    
        int ifirst_tmp = iil;
        for (i=0; i<nproc; i++) {
          int chunk  = (iiu-iil+1)/nproc + (i < (iiu-iil+1)%nproc);
          int ilast_tmp;
          if (i == nproc-1) {
            ilast_tmp = iiu;
          } else {
            ilast_tmp = ifirst_tmp + chunk - 1;
            ilast_tmp = imin(ilast_tmp, iiu);
          }
          if (i == pid) {
            ifirst    = ifirst_tmp;
            ilast     = ilast_tmp;
            isize     = ilast - ifirst + 1;
            *offsetp = ifirst - iil;
            *nzp      = isize;
          }
          ifirst_tmp = ilast_tmp + 1;
          ifirst_tmp = imin(ifirst_tmp, iiu + 1);
        }
        if (isize > 0) {
          memmove(W, &W[ifirst-1], *nzp * sizeof(FloatingType));
        }

        /* If matrix was scaled, rescale eigenvalues */
        detail::invscale_eigenvalues(Wstruct, scale, *nzp);

        detail::clean_up(comm_dup, Werr, Wgap, gersch, iblock, iproc, Windex,
             isplit, Zindex, procinfo, Dstruct, Wstruct, Zstruct,
             tolstruct);

        return 0;
      } /* end of only eigenvalues to compute */

      /* Compute eigenvectors */
      info = detail::plarrv(procinfo, Dstruct, Wstruct, Zstruct, tolstruct, 
            nzp, offsetp);
      assert(info == 0);

      /* Refine to high relative with respect to input matrix */
      if (*tryracp) {
        info = detail::refine_to_highrac(procinfo, jobz, Dcopy, E2copy, 
                     Dstruct, nzp, Wstruct, tolstruct);
        assert(info == 0);
      }

      /* If matrix was scaled, rescale eigenvalues */
      detail::invscale_eigenvalues(Wstruct, scale, n);

      /* Sort eigenvalues and eigenvectors of process */
      detail::sort_eigenpairs<FloatingType>(procinfo, Wstruct, Zstruct);

      detail::clean_up(comm_dup, Werr, Wgap, gersch, iblock, iproc, Windex,
           isplit, Zindex, procinfo, Dstruct, Wstruct, Zstruct,
           tolstruct);
      if (*tryracp) {
        free(Dcopy);
        free(E2copy);
      }

      return 0;
    } /* end pmrrr */


    /*
     * Routine to communicate eigenvalues such that every process has
     * all computed eigenvalues (iu-il+1) in W; this routine is designed 
     * to be called right after 'pmrrr'.
     *
     * Arguments:
     * ----------
     *
     * INPUTS: 
     * -------
     * jobz              "N" or "n" - compute only eigenvalues
     *                   "V" or "v" - compute also eigenvectors
     *                   "C" or "c" - count the maximal number of 
     *                                locally computed eigenvectors
     * range             "A" or "a" - all
     *                   "V" or "v" - by interval: (VL,VU]
     *                   "I" or "i" - by index:     IL-IU
     * n                 matrix size
     * ldz               must be set on input to the leading dimension 
     *                   of of eigenvector matrix Z; this is often equal 
     *                   to matrix size n (not changed on output)
     *
     * INPUT + OUTPUT: 
     * ---------------
     * D (double[n])     Diagonal elements of tridiagonal T.
     *                   (On output the array will be overwritten).
     * E (double[n])     Off-diagonal elements of tridiagonal T.
     *                   First n-1 elements contain off-diagonals,
     *                   the last element can have an abitrary value. 
     *                   (On output the array will be overwritten.)
     * vl                If range="V", lower bound of interval
     *                   (vl,vu], on output refined.
     *                   If range="A" or "I" not referenced as input.
     *                   On output the interval (vl,vu] contains ALL
     *                   the computed eigenvalues.
     * vu                If range="V", upper bound of interval
     *                   (vl,vu], on output refined.
     *                   If range="A" or "I" not referenced as input.
     *                   On output the interval (vl,vu] contains ALL
     *                   the computed eigenvalues.
     * il                If range="I", lower index (1-based indexing) of 
     *                   the subset 'il' to 'iu'.
     *                   If range="A" or "V" not referenced as input.
     *                   On output the eigenvalues with index il to iu are 
     *                   computed by ALL processes.
     * iu                If range="I", upper index (1-based indexing) of 
     *                   the subset 'il' to 'iu'.
     *                   If range="A" or "V" not referenced as input.
     *                   On output the eigenvalues with index il to iu are 
     *                   computed by ALL processes.
     * tryrac            0 - do not try to achieve high relative accuracy.
     *                   NOTE: this should be the default in context of  
     *                         dense eigenproblems.
     *                   1 - relative accuracy will be attempted; 
     *                       on output it is set to zero if high relative 
     *                       accuracy is not achieved.
     * comm              MPI communicator; commonly: MPI_COMM_WORLD.
     *
     * OUTPUT: 
     * -------
     * nz                Number of eigenvalues and eigenvectors computed 
     *                   locally.
     *                   If jobz="C", 'nz' will be set to the maximal
     *                   number of locally computed eigenvectors such 
     *                   that double[n*nz] will provide enough memory 
     *                   for the local eigenvectors;  this is only 
     *                   important in case of range="V" since 
     *                   '#eigenpairs' are not known in advance
     * offset            Index, relative to the computed eigenvalues, of 
     *                   the smallest eigenvalue computed locally
     *                   (0-based indexing).
     * W (double[n])     Locally computed eigenvalues;
     *                   The first nz entries contain the eigenvalues 
     *                   computed locally; the first entry contains the 
     *                   'offset + 1'-th computed eigenvalue, which is the 
     *                   'offset + il'-th eigenvalue of the input matrix 
     *                   (1-based indexing in both cases).
     *                   In some situations it is desirable to have all 
     *                   computed eigenvalues in W, instead of only 
     *                   those computed locally. In this case, call 
     *                   routine 'PMR_comm_eigvals' after 
     *                   'pmrrr' returns (see example and interface below).
     * Z                 Locally computed eigenvectors.
     * (double[n*nz])    Enough space must be provided to store the
     *                   vectors. 'nz' should be bigger or equal 
     *                   to ceil('#eigenpairs'/'#processes'), where 
     *                   '#eigenpairs' is 'n' in case of range="A" and
     *                   'iu-il+1' in case of range="I". Alternatively, 
     *                   and for range="V" 'nz' can be obtained 
     *                   by running the routine with jobz="C". 
     * Zsupp             Support of eigenvectors, which is given by
     * (double[2*n])     i1=Zsupp[2*i] to i2=Zsupp[2*i+1] for the i-th local eigenvector
     *                   (returns 1-based indexing; e.g. in C Z[i1-1:i2-1] are non-zero and
     *                   in Fotran Z(i1:i2) are non-zero).
     *
     * RETURN VALUE: 
     * -------------
     *                 0 - success  
     *                 1 - wrong input parameter
     *                 2 - misc errors  
     *
     * The Fortran interface takes an additinal integer argument INFO
     * to retrieve the return value. 
     * An example call in Fortran looks therefore like
     *
     * CALL PMRRR('V', 'A', N, D, E, VL, VU, IL, IU, TRYRAC, 
     *            MPI_COMM_WORLD, NZ, MYFIRST, W, Z, LDZ, ZSUPP, INFO)
     *
     *
     * EXAMPLE CALL: 
     * -------------
     * char    *jobz, *range;
     * int     n, il, iu, tryRAC=0, nz, offset, ldz, *Zsupp;
     * double  *D, *E, *W, *Z, vl, vu;
     *
     * // allocate space for D, E, W, Z
     * // initialize D, E
     * // set jobz, range, ldz, and if necessary, il, iu or vl, vu  
     * 
     * info = pmrrr(jobz, range, &n, D, E, &vl, &vu, &il, &iu,
     *              &tryRAC, MPI_COMM_WORLD, &nz, &myfirst, W,
     *          Z, &ldz , Zsupp);
     *
     * // optional: 
     * PMR_comm_eigvals(MPI_COMM_WORLD, &nz, &myfirst, W);
     *
     */
    template<typename FloatingType>
    int PMR_comm_eigvals(MPI_Comm comm, int *nz, int *myfirstp, FloatingType *W)
    {
      MPI_Comm comm_dup;
      MPI_Comm_dup(comm, &comm_dup);
      int nproc;
      MPI_Comm_size(comm_dup, &nproc);

      int *rcount = (int *) malloc( nproc * sizeof(int) );
      assert(rcount != NULL);
      int *rdispl = (int *) malloc( nproc * sizeof(int) );
      assert(rdispl != NULL);
      FloatingType *work = (FloatingType *) malloc((*nz+1) * sizeof(FloatingType));
      assert(work != NULL);

      if (*nz > 0)
        memcpy(work, W, (*nz) * sizeof(FloatingType) );

      MPI_Allgather(nz, 1, MPI_INT, rcount, 1, MPI_INT, comm_dup);

      MPI_Allgather(myfirstp, 1, MPI_INT, rdispl, 1, MPI_INT, comm_dup);
      
      MPI_Allgatherv(work, *nz, float_traits<FloatingType>::mpi_type(), W, rcount, rdispl,
             float_traits<FloatingType>::mpi_type(), comm_dup);

      MPI_Comm_free(&comm_dup);
      free(rcount);
      free(rdispl);
      free(work);

      return 0;
    }

    /**
        Helper methods
     **/

    namespace detail{

        /*
         * Free's on allocated memory of pmrrr routine
         */
        template<typename FloatingType>
        void clean_up(MPI_Comm comm, FloatingType *Werr, FloatingType *Wgap,
                  FloatingType *gersch, int *iblock, int *iproc,
                  int *Windex, int *isplit, int *Zindex,
                  proc_t *procinfo, in_t<FloatingType> *Dstruct,
                  val_t<FloatingType> *Wstruct, vec_t<FloatingType> *Zstruct,
                  tol_t<FloatingType> *tolstruct)
        {
          MPI_Comm_free(&comm);
          free(Werr);
          free(Wgap);
          free(gersch);
          free(iblock);
          free(iproc);
          free(Windex);
          free(isplit);
          free(Zindex);
          free(procinfo);
          free(Dstruct);
          free(Wstruct);
          free(Zstruct);
          free(tolstruct);
        }

        /*
         * Wrapper to call LAPACKs DSTEMR for small matrices
         */
        template<typename FloatingType>
        int handle_small_cases(char *jobz, char *range, int *np, FloatingType  *D,
                       FloatingType *E, FloatingType *vlp, FloatingType *vup, int *ilp,
                       int *iup, int *tryracp, MPI_Comm comm, int *nzp,
                       int *myfirstp, FloatingType *W, FloatingType *Z, int *ldzp,
                       int *Zsupp)
        {
          bool   cntval  = (jobz[0]  == 'C' || jobz[0]  == 'c');
          bool   onlyW   = (jobz[0]  == 'N' || jobz[0]  == 'n');
          bool   wantZ   = (jobz[0]  == 'V' || jobz[0]  == 'v');
          bool   indeig  = (range[0] == 'I' || range[0] == 'i');
          int    n       = *np;
          int    ldz_tmp = *np;
          int    ldz     = *ldzp;

          int    nproc, pid;
          MPI_Comm_size(comm, &nproc);
          MPI_Comm_rank(comm, &pid);

          int lwork, liwork;
          FloatingType *Z_tmp; 
          if (onlyW) {
            lwork  = 12*n;
            liwork =  8*n;
          } else if (cntval) {
            lwork  = 18*n;
            liwork = 10*n;
          } else if (wantZ) {
            lwork  = 18*n;
            liwork = 10*n;
            int itmp;
            if (indeig) itmp = *iup-*ilp+1;
            else        itmp = n;
            Z_tmp = (FloatingType *) malloc(n*itmp * sizeof(FloatingType));
            assert(Z_tmp != NULL);
          } else {
            return 1;
          }

          FloatingType *work = (FloatingType *) malloc( lwork  * sizeof(FloatingType));
          assert(work != NULL);
          int *iwork = (int *)   malloc( liwork * sizeof(int));
          assert(iwork != NULL);

          if (cntval) {
            /* Note: at the moment, jobz="C" should never get here, since
             * it is blocked before. */
            int m, info, MINUSONE = -1;
            FloatingType cnt;
            lapack::odstmr("V", "V", np, D, E, vlp, vup, ilp, iup, &m, W, &cnt,
                &ldz_tmp, &MINUSONE, Zsupp, tryracp, work, &lwork, iwork,
                &liwork, &info);
            assert(info == 0);
            
            *nzp = (int) ceil(cnt/nproc);
            free(work); free(iwork);
            return 0;
          }

          int m, info;
          lapack::odstmr(jobz, range, np, D, E, vlp, vup, ilp, iup, &m, W, Z_tmp,
              &ldz_tmp, np, Zsupp, tryracp, work, &lwork, iwork,
              &liwork, &info);
          assert(info == 0);

          int chunk = iceil(m,nproc);
          int myfirst = imin(pid * chunk, m);
          int mylast = imin((pid+1)*chunk - 1, m - 1);
          int mysize = mylast - myfirst + 1;

          if (mysize > 0) {
            memmove(W, &W[myfirst], mysize*sizeof(FloatingType));
            if (wantZ) {
              if (ldz == ldz_tmp) {
                /* copy everything in one chunk */
                memcpy(Z, &Z_tmp[myfirst*ldz_tmp], n*mysize*sizeof(FloatingType));
              } else {
                int i;
                /* copy each vector seperately */
                for (i=0; i<mysize; i++)
                  memcpy(&Z[i*ldz], &Z_tmp[(myfirst+i)*ldz_tmp], 
                     n*sizeof(FloatingType));
              } 
            } /* if (wantZ) */
          } 
          
          *myfirstp = myfirst;
          *nzp      = mysize;

          if (wantZ) free(Z_tmp);
          free(work);
          free(iwork);

          return 0;
        }

        /*
         * Scale matrix to allowable range, returns 1.0 if not scaled
         */
        template<typename FloatingType>
        FloatingType scale_matrix(in_t<FloatingType> *Dstruct, val_t<FloatingType> *Wstruct, bool valeig)
        {
          int              n  = Dstruct->n;
          FloatingType *restrict D  = Dstruct->D;
          FloatingType *restrict E  = Dstruct->E;
          FloatingType          *vl = Wstruct->vl;
          FloatingType          *vu = Wstruct->vu;

          /* Set some machine dependent constants */
          FloatingType smlnum = std::numeric_limits<FloatingType>::min() / std::numeric_limits<FloatingType>::epsilon();
          FloatingType bignum = 1.0 / smlnum;
          FloatingType rmin = sqrt(smlnum);
          FloatingType rmax = fmin(sqrt(bignum), 1.0 / sqrt(sqrt(std::numeric_limits<FloatingType>::min())));

          /*  Scale matrix to allowable range */
          FloatingType scale = 1.0;
          FloatingType T_norm = lapack::odnst("M", &n, D, E);  /* returns max(|T(i,j)|) */
          if (T_norm > 0 && T_norm < rmin) {
            scale = rmin / T_norm;
          } else if (T_norm > rmax) {
            scale = rmax / T_norm;
          }

          if (scale != 1.0) {  /* FP cmp okay */
            /* Scale matrix and matrix norm */
            int itmp = n-1;
            int IONE = 1;
            blas::odscal(&n,    &scale, D, &IONE);
            blas::odscal(&itmp, &scale, E, &IONE);
            if (valeig == true) {
              /* Scale eigenvalue bounds */
              *vl *= scale;
              *vu *= scale;
            }
          } /* end scaling */

          return scale;
        }

        /*
         * If matrix scaled, rescale eigenvalues
         */
        template<typename FloatingType>
        void invscale_eigenvalues(val_t<FloatingType> *Wstruct, FloatingType scale,
                      int size)
        {
          if (scale != 1.0) {  /* FP cmp okay */
            FloatingType *vl = Wstruct->vl;
            FloatingType *vu = Wstruct->vu;
            FloatingType *W  = Wstruct->W;

            FloatingType invscale = 1.0 / scale;
            int IONE = 1;
            *vl *= invscale;
            *vu *= invscale;
            blas::odscal(&size, &invscale, W, &IONE);
          }

        }
        
        template<typename FloatingType>
        int sort_eigenpairs_local(proc_t *procinfo, int m, val_t<FloatingType> *Wstruct, vec_t<FloatingType> *Zstruct)
        {
          int              pid        = procinfo->pid;
          int              n        = Wstruct->n;
          FloatingType *restrict W        = Wstruct->W;
          FloatingType *restrict work     = Wstruct->gersch;
          int              ldz      = Zstruct->ldz;
          FloatingType *restrict Z        = Zstruct->Z;
          int    *restrict Zsupp    = Zstruct->Zsupp;
         
          bool             sorted;
          int              j;
          FloatingType           tmp;
          int              itmp1, itmp2;
          
          /* Make sure that sorted correctly; ineffective implementation,
           * but usually no or very little swapping should be done here */
          sorted = false;
          while (sorted == false) {
            sorted = true;
            for (j=0; j<m-1; j++) {
              if (W[j] > W[j+1]) {
                sorted = false;
                /* swap eigenvalue */
                tmp    = W[j];
                W[j]   = W[j+1];
                W[j+1] = tmp;
                /* swap eigenvalue support */
                itmp1 = Zsupp[2*j];
                Zsupp[2*j] = Zsupp[2*(j+1)];
                Zsupp[2*(j+1)] = itmp1;
                itmp2 = Zsupp[2*j + 1];
                Zsupp[2*j + 1] = Zsupp[2*(j+1) + 1];
                Zsupp[2*(j+1) +1 ] = itmp2;
                /* swap eigenvector */
                memcpy(work, &Z[j*ldz], n*sizeof(FloatingType));
                memcpy(&Z[j*ldz], &Z[(j+1)*ldz], n*sizeof(FloatingType));
                memcpy(&Z[(j+1)*ldz], work, n*sizeof(FloatingType));
              }
            }
          } /* end while */

          return(0);
        }

        template<typename FloatingType>
        int sort_eigenpairs_global(proc_t *procinfo, int m, val_t<FloatingType> *Wstruct, 
                       vec_t<FloatingType> *Zstruct)
        {
          int              pid   = procinfo->pid;
          int              nproc = procinfo->nproc;
          int              n     = Wstruct->n;
          FloatingType *restrict W     = Wstruct->W;
          FloatingType *restrict work  = Wstruct->gersch;
          int              ldz   = Zstruct->ldz;
          FloatingType *restrict Z     = Zstruct->Z;
          int    *restrict Zsupp = Zstruct->Zsupp;

          FloatingType           *minW, *maxW, *minmax; 
          int              i, p, lp, itmp[2];
          bool             sorted;
          MPI_Status       status;
          FloatingType  nan_value = 0.0/0.0;
          
          minW   = (FloatingType *) malloc(  nproc*sizeof(FloatingType));
          assert(minW != NULL);
          maxW   = (FloatingType *) malloc(  nproc*sizeof(FloatingType));
          assert(maxW != NULL);
          minmax = (FloatingType *) malloc(2*nproc*sizeof(FloatingType));
          assert(minmax != NULL);

          if (m == 0) {
            MPI_Allgather(&nan_value, 1, float_traits<FloatingType>::mpi_type(), minW, 1, float_traits<FloatingType>::mpi_type(), 
                  procinfo->comm); 
            MPI_Allgather(&nan_value, 1, float_traits<FloatingType>::mpi_type(), maxW, 1, float_traits<FloatingType>::mpi_type(), 
                  procinfo->comm); 
          } else {
            MPI_Allgather(&W[0], 1, float_traits<FloatingType>::mpi_type(), minW, 1, float_traits<FloatingType>::mpi_type(), 
                  procinfo->comm); 
            MPI_Allgather(&W[m-1], 1, float_traits<FloatingType>::mpi_type(), maxW, 1, float_traits<FloatingType>::mpi_type(), 
                  procinfo->comm); 
          }

          for (i=0; i<nproc; i++) {
            minmax[2*i]   = minW[i];
            minmax[2*i+1] = maxW[i];
          }

          sorted = true;
          for (i=0; i<2*nproc-1; i++) {
            if (minmax[i] > minmax[i+1]) sorted = false;
          }

          /* Make sure that sorted correctly; ineffective implementation,
           * but usually no or very little swapping should be done here */
          while (sorted == false) {

            sorted = true;

            for (p=1; p<nproc; p++) {

              lp =  p - 1;

              /* swap one pair of eigenvalues and eigenvectors */
              if ((pid == lp || pid == p) && minW[p] < maxW[lp]) {
                if (pid == lp) {
                  W[m-1] = minW[p];
                      MPI_Sendrecv(&Z[(m-1)*ldz], n, float_traits<FloatingType>::mpi_type(), p, lp, 
                           work, n, float_traits<FloatingType>::mpi_type(), p, p, 
                           procinfo->comm, &status);
                  memcpy(&Z[(m-1)*ldz], work, n*sizeof(FloatingType));
                }
                if (pid == p) {
                  W[0]   = maxW[p-1];
                      MPI_Sendrecv(&Z[0], n, float_traits<FloatingType>::mpi_type(), lp, p, 
                           work,  n, float_traits<FloatingType>::mpi_type(), lp, lp, 
                           procinfo->comm, &status);
                  memcpy(&Z[0], work, n*sizeof(FloatingType));
                }
              }

              /* swap eigenvector support as well; 
               * (would better be recomputed here though) */
              if ((pid == lp || pid == p) && minW[p] < maxW[lp]) {
                if (pid == lp) {
                      MPI_Sendrecv(&Zsupp[2*(m-1)], 2, MPI_INT, p, lp, 
                           itmp, 2, MPI_INT, p, p, 
                           procinfo->comm, &status);
                  Zsupp[2*(m-1)]     = itmp[0];
                  Zsupp[2*(m-1) + 1] = itmp[1];
                }
                if (pid == p) {
                      MPI_Sendrecv(&Zsupp[0], 2, MPI_INT, lp, p, 
                           itmp,  2, MPI_INT, lp, lp, 
                           procinfo->comm, &status);
                  Zsupp[0] = itmp[0];
                  Zsupp[1] = itmp[1];
                }
              }
            }

            /* sort local again */
            sort_eigenpairs_local<FloatingType>(procinfo, m, Wstruct, Zstruct);
            
            /* check again if globally sorted */
            if (m == 0) {
              MPI_Allgather(&nan_value, 1, float_traits<FloatingType>::mpi_type(), minW, 1, float_traits<FloatingType>::mpi_type(), 
                    procinfo->comm); 
              MPI_Allgather(&nan_value, 1, float_traits<FloatingType>::mpi_type(), maxW, 1, float_traits<FloatingType>::mpi_type(), 
                    procinfo->comm);       
            } else {
              MPI_Allgather(&W[0], 1, float_traits<FloatingType>::mpi_type(), minW, 1, float_traits<FloatingType>::mpi_type(), 
                    procinfo->comm); 
              MPI_Allgather(&W[m-1], 1, float_traits<FloatingType>::mpi_type(), maxW, 1, float_traits<FloatingType>::mpi_type(), 
                    procinfo->comm); 
            }
            
            for (i=0; i<nproc; i++) {
              minmax[2*i]   = minW[i];
              minmax[2*i+1] = maxW[i];
            }
            
            for (i=0; i<2*nproc-1; i++) {
              if (minmax[i] > minmax[i+1]) sorted = false;
            }
            
          } /* end while not sorted */

          free(minW);
          free(maxW);
          free(minmax);

          return 0;
        }

        /* Routine to sort the eigenpairs */
        template<typename FloatingType>
        int sort_eigenpairs(proc_t *procinfo, val_t<FloatingType> *Wstruct, vec_t<FloatingType> *Zstruct)
        {
          /* From inputs */
          int              pid      = procinfo->pid;
          int              n        = Wstruct->n;
          FloatingType *restrict W        = Wstruct->W;
          int    *restrict Windex   = Wstruct->Windex;
          int    *restrict iproc    = Wstruct->iproc;
          int    *restrict Zindex   = Zstruct->Zindex;

          /* Others */
          int           im, j;
          sort_struct_t<FloatingType> *sort_array;

          /* Make the first nz elements of W contains the eigenvalues
           * associated to the process */
          im = 0;
          for (j=0; j<n; j++) {
            if (iproc[j] == pid) {
              W[im]      = W[j];
              Windex[im] = Windex[j];
              Zindex[im] = Zindex[j];
              im++;
            }
          }

          sort_array = (sort_struct_t<FloatingType> *) malloc(im*sizeof(sort_struct_t<FloatingType>));

          for (j=0; j<im; j++) {
            sort_array[j].lambda    = W[j]; 
            sort_array[j].local_ind = Windex[j];
            sort_array[j].block_ind = 0;
            sort_array[j].ind       = Zindex[j];
          }

          /* Sort according to Zindex */
          sort(sort_array, sort_array + im, detail::cmp_sort_struct<FloatingType>);

          for (j=0; j<im; j++) {
            W[j]      = sort_array[j].lambda; 
            Windex[j] = sort_array[j].local_ind;
          }

          /* Make sure eigenpairs are sorted locally; this is a very 
           * inefficient way sorting, but in general no or very little 
           * swapping of eigenpairs is expected here */
          sort_eigenpairs_local<FloatingType>(procinfo, im, Wstruct, Zstruct);

          /* Make sure eigenpairs are sorted globally; this is a very 
           * inefficient way sorting, but in general no or very little 
           * swapping of eigenpairs is expected here */
          if (ASSERT_SORTED_EIGENPAIRS == true)
            sort_eigenpairs_global<FloatingType>(procinfo, im, Wstruct, Zstruct);

          free(sort_array);

          return(0);
        }

        /*
         * Refines the eigenvalue to high relative accuracy with
         * respect to the input matrix;
         * Note: In principle this part could be fully parallel too,
         * but it will only rarely be called and not much work
         * is involved, if the eigenvalues are not small in magnitude
         * even no work at all is not uncommon. 
         */
        template<typename FloatingType>
        int refine_to_highrac(proc_t *procinfo, char *jobz, FloatingType *D,
                      FloatingType *E2, in_t<FloatingType> *Dstruct, int *nzp,
                      val_t<FloatingType> *Wstruct, tol_t<FloatingType> *tolstruct)
        {
          int              n      = Dstruct->n;
          int              nsplit = Dstruct->nsplit;
          int    *restrict isplit = Dstruct->isplit;
          FloatingType     spdiam = Dstruct->spdiam;
          FloatingType *restrict W      = Wstruct->W;
          FloatingType *restrict Werr   = Wstruct->Werr;

          FloatingType *work  = (FloatingType *) malloc( 2*n * sizeof(FloatingType) );
          assert (work != NULL);
          int *iwork = (int *)    malloc( 2*n * sizeof(int)    );
          assert (iwork != NULL);

          int j, ibegin  = 0;
          for (j=0; j<nsplit; j++) {
            
            int iend   = isplit[j] - 1;
            int isize  = iend - ibegin + 1;
            int nbl    = isize;
            
            if (nbl == 1) {
              ibegin = iend + 1;
              continue;
            }
            
            int ifirst = 1, ilast = nbl, offset = 0, info;

            FloatingType pivmin = tolstruct->pivmin;
            FloatingType tol    = 4 * std::numeric_limits<FloatingType>::epsilon(); 
            lapack::odrrj(&isize, &D[ibegin], &E2[ibegin], &ifirst, &ilast, &tol,
                &offset, &W[ibegin], &Werr[ibegin], work, iwork, &pivmin,
                &spdiam, &info);
            assert(info == 0);
            
            ibegin = iend + 1;
          } /* end j */
          
          free(work);
          free(iwork);
          return 0;
        }

        /*
         * Compare function for using qsort() on an array
         * of FloatingTypes
         */
        template<typename FloatingType>
        bool cmp(const FloatingType & arg1, const FloatingType & arg2)
        {
            return arg1 < arg2;
        }

        /*
         * Compare function for using qsort() on an array of 
         * sort_structs
         */
        template<typename FloatingType>
        bool cmp_sort_struct(const sort_struct_t<FloatingType> & arg1, const sort_struct_t<FloatingType> & arg2)
        {
          /* Within block local index decides */
          return arg1.ind < arg2.ind;
        }
    }
/* Fortran function prototype */
/*void pmrrr_(char *jobz, char *range, int *n, FloatingType  *D,
	    FloatingType *E, FloatingType *vl, FloatingType *vu, int *il, int *iu,
	    int *tryracp, MPI_Fint *comm, int *nz, int *myfirst,
	    FloatingType *W, FloatingType *Z, int *ldz, int *Zsupp, int* info)
{
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);

  *info = pmrrr(jobz, range, n, D, E, vl, vu, il, iu, tryracp, 
		c_comm, nz, myfirst, W, Z, ldz, Zsupp);
}

void pmr_comm_eigvals_(MPI_Fint *comm, int *nz, int *myfirstp, 
		       FloatingType *W, int *info)
{
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);

  *info = PMR_comm_eigvals(c_comm, nz, myfirstp, W);
}*/

}

#endif
