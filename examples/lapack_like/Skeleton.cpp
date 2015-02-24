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

typedef double Real;
typedef Complex<Real> C;

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );

    try 
    {
        const Int m = Input("--height","height of matrix",20);
        const Int n = Input("--width","width of matrix",100);
        const Int r = Input("--rank","rank of matrix",5);
        const Int maxSteps = Input("--maxSteps","max # of steps of QR",10);
        const double tol = Input("--tol","tolerance for ID",-1.);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        DistMatrix<C> U, V;
        Uniform( U, m, r );
        Uniform( V, n, r );
        DistMatrix<C> A;
        Gemm( NORMAL, ADJOINT, C(1), U, V, A );
        const Real frobA = FrobeniusNorm( A );
        if( print )
            Print( A, "A" );

        const Grid& g = A.Grid();
        QRCtrl<double> ctrl;
        ctrl.boundRank = true;
        ctrl.maxRank = maxSteps;
        if( tol != -1. )
        {
            ctrl.adaptive = true;
            ctrl.tol = tol;
        }
        DistMatrix<Int,VR,STAR> permR(g), permC(g);
        DistMatrix<C> Z(g);
        Skeleton( A, permR, permC, Z, ctrl );
        const Int rank = Z.Height();
        if( print )
        {
            Print( permR, "permR" );
            Print( permC, "permC" );
            Print( Z, "Z" );
        }

        // Form the matrices of A's (hopefully) dominant rows and columns
        DistMatrix<C> AR( A );
        InversePermuteRows( AR, permR );
        AR.Resize( rank, A.Width() );
        DistMatrix<C> AC( A );
        InversePermuteCols( AC, permC );
        AC.Resize( A.Height(), rank );
        if( print )
        {
            Print( AC, "AC" );
            Print( AR, "AR" );
        }

        // Check || A - AC Z AR ||_F / || A ||_F
        DistMatrix<C> B(g);
        Gemm( NORMAL, NORMAL, C(1), Z, AR, B );
        Gemm( NORMAL, NORMAL, C(-1), AC, B, C(1), A );
        const Real frobError = FrobeniusNorm( A );
        if( print )
            Print( A, "A - AC Z AR" );

        if( mpi::WorldRank() == 0 )
        {
            cout << "|| A ||_F = " << frobA << "\n\n"
                 << "|| A - AC Z AR ||_F / || A ||_F = " 
                 << frobError/frobA << "\n" << endl;
        }
    }
    catch( exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
