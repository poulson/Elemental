/* Copyright (c) 2010, RWTH Aachen University
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

#ifndef TTASKS_H
#define TTASKS_H

#ifndef DISABLE_PTHREADS
 #include <semaphore.h>
#endif

#include <pmrrr/definitions/global.h>
#include <pmrrr/definitions/queue.h>
#include <pmrrr/definitions/structs.h>
#include <pmrrr/definitions/rrr.h>

#define SINGLETON_TASK_FLAG  0
#define CLUSTER_TASK_FLAG    1
#define REFINE_TASK_FLAG     2

namespace pmrrr { namespace detail {

	template<typename FloatingType>
	struct singleton_t {
	  int        begin;
	  int        end;
	  int        depth;
	  int        bl_begin;
	  int        bl_end;
	  FloatingType     bl_spdiam;
	  FloatingType     lgap;
	  rrr_t<FloatingType>      *RRR;
	};

	template<typename FloatingType>
	struct cluster_t {
	  int        begin;
	  int        end;
	  int        depth;
	  int        bl_begin;   /* In priciple not needed since info */
	  int        bl_end;     /* also contained in iblock+isplit */
	  FloatingType     bl_spdiam;
	  FloatingType     lgap;
	  int        proc_W_begin;
	  int        proc_W_end;
	  int        left_pid;
	  int        right_pid;
	  rrr_t<FloatingType>      *RRR;
	  bool       wait_until_refined;
	  comm_t     *messages;
	};

	template<typename FloatingType>
	struct refine_t{
	  int        begin;
	  int        end;
	  FloatingType     *D;
	  FloatingType     *DLL;
	  int        p;
	  int        q;
	  int        bl_size;
	  FloatingType     bl_spdiam;
	  int        producer_tid; // not longer needed
	#ifndef DISABLE_PTHREADS
	  sem_t      *sem; /* since semt_t is a handle could also store it
				  instead of pointer to it, but pointer is all
		                  that is needed */
	#endif
	};	

} }
#endif
