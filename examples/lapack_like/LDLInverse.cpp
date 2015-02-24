/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
using namespace std;
using namespace El;

// Typedef our real and complex types to 'Real' and 'C' for convenience
typedef double Real;
typedef Complex<Real> C;

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );

    try 
    {
        const Int n = Input("--size","size of matrix to factor",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const double realMean = Input("--realMean","real mean",0.);
        const double imagMean = Input("--imagMean","imag mean",0.);
        const double stddev = Input("--stddev","standard dev.",1.);
        const bool conjugate = Input("--conjugate","LDL^H?",false);
        ProcessInput();
        PrintInputReport();

        SetBlocksize( nb );

        C mean( realMean, imagMean );
        DistMatrix<C> A;
        if( conjugate )
        {
            Wigner( A, n, mean, stddev );
            //HermitianUniformSpectrum( A, n, 1, 2 );
        }
        else
        {
            Gaussian( A, n, n, mean, stddev );
            MakeSymmetric( LOWER, A );
        }

        // Make a copy of A and then overwrite it with its inverse
        DistMatrix<C> invA( A );
        SymmetricInverse( LOWER, invA, conjugate );

        // Form I - invA*A and print the relevant norms
        DistMatrix<C> E;
        Identity( E, n, n );
        Symm( LEFT, LOWER, C(-1), invA, A, C(1), E, conjugate );

        const Real frobNormA = FrobeniusNorm( A );
        const Real frobNormInvA = SymmetricFrobeniusNorm( LOWER, invA );
        const Real frobNormError = FrobeniusNorm( E );
        if( mpi::WorldRank() == 0 )
        {
            cout << "|| A          ||_F = " << frobNormA << "\n"
                 << "|| invA       ||_F = " << frobNormInvA << "\n"
                 << "|| I - invA A ||_F = " << frobNormError << "\n" << endl;
        }
    }
    catch( exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
