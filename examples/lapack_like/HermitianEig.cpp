/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

using namespace std;
using namespace El;

template<typename Real>
void run_example(Int n, bool print)
{
    typedef Complex<Real> C;

    // Create a 2d process grid from a communicator. In our case, it is
    // mpi::COMM_WORLD. There is another constructor that allows you to 
    // specify the grid dimensions, Grid g( comm, r ), which creates a
    // grid of height r.
    Grid g( mpi::COMM_WORLD );

    // Create an n x n complex distributed matrix, 
    // We distribute the matrix using grid 'g'.
    //
    // There are quite a few available constructors, including ones that 
    // allow you to pass in your own local buffer and to specify the 
    // distribution alignments (i.e., which process row and column owns the
    // top-left element)
    DistMatrix<C> H( n, n, g );

    // Fill the matrix since we did not pass in a buffer. 
    //
    // We will fill entry (i,j) with the complex value (i+j,i-j) so that 
    // the global matrix is Hermitian. However, only one triangle of the 
    // matrix actually needs to be filled, the symmetry can be implicit.
    //
    const Int localHeight = H.LocalHeight();
    const Int localWidth = H.LocalWidth();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        // Our process owns the rows colShift:colStride:n,
        //           and the columns rowShift:rowStride:n
        const Int j = H.GlobalCol(jLoc);
        for( Int iLoc=0; iLoc<localHeight; ++iLoc )
        {
            const Int i = H.GlobalRow(iLoc);
            H.SetLocal( iLoc, jLoc, C(i+j,i-j) );
        }
    }

    // Make a backup of H before we overwrite it within the eigensolver
    auto HCopy( H );

    // Call the eigensolver. We first create an empty complex eigenvector 
    // matrix, Q[MC,MR], and an eigenvalue column vector, w[VR,* ]
    //
    // Optional: set blocksizes and algorithmic choices here. See the 
    //           'Tuning' section of the README for details.
    DistMatrix<Real,VR,STAR> w( g );
    DistMatrix<C> Q( g );
    HermitianEig( LOWER, H, w, Q ); 

    if( print )
    {
        Print( HCopy, "H" );
        Print( Q, "Q" );
        Print( w, "w" );
    }

    // Check the residual, || H Q - Omega Q ||_F
    const Real frobH = HermitianFrobeniusNorm( LOWER, HCopy );
    auto E( Q );
    DiagonalScale( RIGHT, NORMAL, w, E );
    Hemm( LEFT, LOWER, C(-1), HCopy, Q, C(1), E );
    const Real frobResid = FrobeniusNorm( E );

    // Check the orthogonality of Q
    Identity( E, n, n );
    Herk( LOWER, ADJOINT, Real(-1), Q, Real(1), E );
    const Real frobOrthog = HermitianFrobeniusNorm( LOWER, E );

    if( mpi::Rank() == 0 )
        Output
        ("|| H ||_F = ",frobH,"\n",
         "|| H Q - Q Omega ||_F / || A ||_F = ",frobResid/frobH,"\n",
         "|| Q' Q - I ||_F = ",frobOrthog,"\n");
}


int
main( int argc, char* argv[] )
{
    // This detects whether or not you have already initialized MPI and 
    // does so if necessary. 
    Environment env( argc, argv );

    // Surround the Elemental calls with try/catch statements in order to 
    // safely handle any exceptions that were thrown during execution.
    try 
    {
        const Int n = Input("--size","size of matrix",100);
        const bool print = Input("--print","print matrices?",false);
        const bool single_precision = Input("--single", "single precision?", false);
        ProcessInput();
        PrintInputReport();
        if(single_precision) {
            run_example<float>(n, print);
        } else {
            run_example<double>(n, print);
        }
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
