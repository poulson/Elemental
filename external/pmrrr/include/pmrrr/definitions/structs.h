/* Copyright (c) 2010, RWTH Aachen University
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

#ifndef SSTRUCTS_H
#define SSTRUCTS_H

#include "mpi.h"
#include "global.h"
#include "counter.h"
#include "queue.h"

namespace pmrrr { namespace detail {

	template<typename FloatingType>
	struct in_t {
	  int              n;
	  FloatingType *restrict D;
	  FloatingType *restrict E;
	  int              nsplit;
	  int    *restrict isplit ;
	  FloatingType           spdiam;
	};

	template<typename FloatingType>
	struct val_t {
	  int              n;
	  FloatingType           *vl;
	  FloatingType           *vu;
	  int              *il;
	  int              *iu;
	  FloatingType *restrict W;
	  FloatingType *restrict Werr;
	  FloatingType *restrict Wgap;
	  int    *restrict Windex;
	  int    *restrict iblock;
	  int    *restrict iproc;
	  FloatingType *restrict Wshifted;
	  FloatingType *restrict gersch;
	};

	template<typename FloatingType>
	struct vec_t {
	  int              ldz;
	  int              nz;
	  FloatingType *restrict Z;
	  int    *restrict Zsupp;
	  int    *restrict Zindex;
	};

	struct proc_t {
	  int      pid;
	  int      nproc;
	  MPI_Comm comm;
	  int      nthreads;
	  int      thread_support;
	};

	template<typename FloatingType>
	struct tol_t {
	  FloatingType split;
	  FloatingType rtol1;
	  FloatingType rtol2;
	  FloatingType pivmin;
	};

	typedef struct {
	  int         num_messages;
	  MPI_Request *requests;
	  MPI_Status  *stats;
	} comm_t;

	typedef struct {
	  queue_t *r_queue;
	  queue_t *s_queue;
	  queue_t *c_queue;
	} workQ_t;

	template<typename FloatingType>
	struct sort_struct_t{
	  FloatingType lambda;
	  int    local_ind;
	  int    block_ind;
	  int    ind;
	};

	template<typename FloatingType>
	struct auxarg1_t {
	  int    n;
	  FloatingType *D;
	  FloatingType *E;
	  FloatingType *E2;
	  int    il;
	  int    iu;
	  int    my_il;
	  int    my_iu;
	  int    nsplit;
	  int    *isplit;
	  FloatingType bsrtol;
	  FloatingType pivmin;
	  FloatingType *gersch;
	  FloatingType *W;
	  FloatingType *Werr;
	  int    *Windex;
	  int   	 *iblock;
	};

	template<typename FloatingType>
	struct auxarg2_t {
	  int          bl_size;
	  FloatingType       *D;
	  FloatingType       *DE2;
	  int          rf_begin;
	  int          rf_end;
	  FloatingType        *W;
	  FloatingType        *Werr;
	  FloatingType        *Wgap;
	  int            *Windex;
	  FloatingType       rtol1;
	  FloatingType       rtol2;
	  FloatingType       pivmin;
	  FloatingType       bl_spdiam;
	};

	template<typename FloatingType>
	struct auxarg3_t {
	  int          tid;
	  proc_t       *procinfo;
	  val_t<FloatingType>        *Wstruct;
	  vec_t<FloatingType>        *Zstruct;
	  tol_t<FloatingType>        *tolstruct;
	  workQ_t      *workQ;
	  counter_t    *num_left;
	};

}	// namespace detail

}	// namespace pmrrr

#endif
