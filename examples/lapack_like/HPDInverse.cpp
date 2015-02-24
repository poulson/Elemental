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
        const Int n = Input("--size","size of HPD matrix",100);
        const bool upper = Input("--upper","upper storage?",false);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        DistMatrix<C> A;
        HermitianUniformSpectrum( A, n, Real(1), Real(20) );
        if( print )
            Print( A, "A" );

        // Make a copy of A and then overwrite it with its inverse
        const UpperOrLower uplo = ( upper ? UPPER : LOWER );
        DistMatrix<C> invA( A );
        HPDInverse( uplo, invA );
        if( print )
        {
            MakeHermitian( uplo, invA );
            Print( invA, "inv(A)" );
        }

        // Form I - invA*A and print the relevant norms
        DistMatrix<C> E;
        Identity( E, n, n );
        Hemm( LEFT, uplo, C(-1), invA, A, C(1), E );

        const Real frobNormA = HermitianFrobeniusNorm( uplo, A );
        const Real frobNormInvA = HermitianFrobeniusNorm( uplo, invA );
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

