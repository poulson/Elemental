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

#ifndef __PROCESS_R_TASK_HPP__
#define __PROCESS_R_TASK_HPP__

#include <pmrrr/definitions/pmrrr.h>
#include <pmrrr/definitions/plarrv.h>
#include <pmrrr/definitions/global.h>
#include <pmrrr/definitions/queue.h>
#include <pmrrr/definitions/counter.h>
#include <pmrrr/definitions/rrr.h>
#include <pmrrr/definitions/structs.h>
#include <pmrrr/definitions/tasks.h>

#include <pmrrr/lapack/odrrb.hpp>

namespace pmrrr { namespace detail {

	/*
	 * Executes all tasks which are in the r-queue at the moment of the 
	 * call. This routine is called to make sure that all tasks in the 
	 * queue are dequeued before continueing with other tasks.
	 */
	template<typename FloatingType>
	void PMR_process_r_queue(int tid, proc_t *procinfo, val_t<FloatingType> *Wstruct, 
				 vec_t<FloatingType> *Zstruct, tol_t<FloatingType> *tolstruct, 
				 workQ_t *workQ, counter_t *num_left, 
				 FloatingType *work, int *iwork)
	{
	  int thread_support = procinfo->thread_support;
	  int num_tasks = PMR_get_num_tasks(workQ->r_queue);

      int t;
	  for (t=0; t<num_tasks; t++) {
		
		task_t *task = PMR_remove_task_at_front(workQ->r_queue);

		if ( task != NULL ) {
		
		  if (task->flag == CLUSTER_TASK_FLAG) {

            if (thread_support != MPI_THREAD_FUNNELED || tid == 0) {
              /* if MPI_THREAD_FUNNELED only tid==0 should process 
                   * these tasks, otherwise any thread can do it */
              int status = PMR_process_c_task((cluster_t<FloatingType> *) task->data,
                              tid, procinfo, Wstruct,
                              Zstruct, tolstruct, workQ,
                              num_left, work, iwork);
              
              if (status == C_TASK_PROCESSED) {
                free(task);
              } else {
                PMR_insert_task_at_back(workQ->r_queue, task);
              }
            } else {
                PMR_insert_task_at_back(workQ->r_queue, task);
            }

		  } /* end if cluster task */
          else if (task->flag == REFINE_TASK_FLAG) {
            PMR_process_r_task((refine_t<FloatingType> *) task->data, procinfo,
                       Wstruct, tolstruct, work, iwork);
            free(task);
		  } 
		} /* end if task removed */
	  } /* end for t */
	} /* end process_entire_r_queue */
  
	/*
	 * Process the task of refining a subset of eigenvalues.
	 */
	template<typename FloatingType>
	int PMR_process_r_task(refine_t<FloatingType> *rf, proc_t *procinfo, 
				   val_t<FloatingType> *Wstruct, tol_t<FloatingType> *tolstruct, 
				   FloatingType *work, int *iwork)
	{
	  /* From inputs */
	  int              		 ts_begin  = rf->begin;
	  FloatingType *restrict D         = rf->D;
	  FloatingType *restrict DLL       = rf->DLL;
	  int              		 p         = rf->p;
	  int              		 q         = rf->q;
	  int              		 bl_size   = rf->bl_size;
	  FloatingType           bl_spdiam = rf->bl_spdiam;

	  FloatingType *restrict Werr      = Wstruct->Werr;
	  FloatingType *restrict Wgap      = Wstruct->Wgap;
	  int    *restrict 		 Windex    = Wstruct->Windex;
	  FloatingType *restrict Wshifted  = Wstruct->Wshifted;
	  
	  FloatingType savegap;

	  if (p == q) {
		savegap = Wgap[ts_begin];
		Wgap[ts_begin] = 0.0;
	  }

      int info;
      int offset = Windex[ts_begin]-1;
      FloatingType rtol1 = tolstruct->rtol1;
      FloatingType rtol2 = tolstruct->rtol2;
      FloatingType pivmin = tolstruct->pivmin;
	  lapack::odrrb(&bl_size, D, DLL, &p, &q, &rtol1, &rtol2, &offset, 
		  &Wshifted[ts_begin], &Wgap[ts_begin], &Werr[ts_begin],
		  work, iwork, &pivmin, &bl_spdiam, &bl_size, &info);
	  assert(info == 0);

	  if (p == q) {
		Wgap[ts_begin] = savegap;
	  }  

	  PMR_refine_sem_post(rf);
	  free(rf);

	  return 0;
	}

} //namespace detail

} //namespace pmrrr

#endif
