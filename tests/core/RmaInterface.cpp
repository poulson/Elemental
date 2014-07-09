/*
   Copyright (c) 2009-2014, Jack Poulson
   Copyright (c) 2011, The University of Texas at Austin
   Copyright (c) 2014, Sayan Ghosh, University of Houston
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
using namespace El;

#define ITER 		10
//#define DIM 		1000
//#define AXPY_DIM 	20
#define DIM 		20
#define AXPY_DIM 	4
#define ALPHA		2.0
#include <assert.h>
    int
main (int argc, char *argv[])
{
    Initialize (argc, argv);
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::Rank (comm);
    const Int commSize = mpi::Size (comm);

    assert (AXPY_DIM < DIM);

    try
    {
	Grid grid (comm);

	// Create an 8 x 8 distributed matrix over the given grid
	DistMatrix < double, MC, MR > A (DIM, DIM, grid);

	// Set every entry of A to zero
	Zeros (A, DIM, DIM);

	// Print the original A
	if (DIM <= 20)
	    Print (A, "Original distributed A");

	for (Int k = 0; k < ITER; ++k)
	{
	    if (commRank == 0)
		std::cout << "Iteration " << k << std::endl;

	    RmaInterface < double > Rmaint;
	    Rmaint.Attach (A);

	    // If we are process 0, then create a 3 x 3 identity matrix, B,
	    // and Axpy it into the bottom-right of A (using alpha=2)
	    // NOTE: The bottom-right 3 x 3 submatrix starts at the (5,5)
	    //       entry of A.
	    // NOTE: Every process is free to Axpy as many submatrices as they
	    //       desire at this point.
	    if (grid.VCRank () == 0)
	    {
		mpi::Op op;
		op.op = MPI_SUM;
		Matrix < double >B (AXPY_DIM, AXPY_DIM);
		Identity (B, AXPY_DIM, AXPY_DIM);
		// AXPY is scaled accumulate as in ARMCI 
		Rmaint.Acc (ALPHA, B, op, (DIM - AXPY_DIM), (DIM - AXPY_DIM));
		Rmaint.Flush (B, (DIM - AXPY_DIM), (DIM - AXPY_DIM));
	    }
	    if (DIM <= 20)
		Print (A, "Updated distributed A");
	    // Have process 0 request a copy of the entire distributed matrix
	    //
	    // NOTE: Every process is free to Axpy as many submatrices as they
	    //       desire at this point.
	    Matrix < double >C;
	    if (grid.VCRank () == 0)
	    {
		Zeros (C, DIM, DIM);
		Rmaint.Get (C, 0, 0);
		Rmaint.Flush (C);
	    }

	    // Process 0 can now locally print its copy of A
	    if (grid.VCRank () == 0 && DIM <= 20)
		Print (C, "Process 0's local copy of A");
	    // Collectively detach in order to finish filling process 0's request
	    Rmaint.Detach ();
	}
    }
    catch (std::exception & e)
    {
	ReportException (e);
    }

    Finalize ();
    return 0;
}
