/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
using namespace El;

// Solve
//
//     minimize || A z - y ||_2 such that z >= 0
//        z 
//
// via the Quadratic Program
//
//     minimize    (1/2) x' Q x + c' x 
//     subject to  x >= 0
//
// with Q = A^T A and c = -A^H y.
//

typedef double Real;

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );

    try
    {
        const Int m = Input("--m","matrix height",200);
        const Int n = Input("--n","matrix width",100);
        const Int k = Input("--k","number of right-hand sides",10);
        // TODO: Test both the ADMM and IPM versions
        /*
        const Int maxIter = Input("--maxIter","maximum # of iter's",500);
        const Real rho = Input("--rho","augmented Lagrangian param.",1.);
        const Real alpha = Input("--alpha","over-relaxation",1.2);
        const Real absTol = Input("--absTol","absolute tolerance",1e-6);
        const Real relTol = Input("--relTol","relative tolerance",1e-4);
        const bool inv = Input("--inv","form inv(LU) to avoid trsv?",true);
        */
        const bool progress = Input("--progress","print progress?",true);
        const bool display = Input("--display","display matrices?",false);
        const bool print = Input("--print","print matrices",false);
        ProcessInput();
        PrintInputReport();

        DistMatrix<Real> A, B;
        Uniform( A, m, n );
        Uniform( B, m, k );
        if( print )
        {
            Print( A, "A" );
            Print( B, "B" );
        }
        if( display )
            Display( A, "A" );

        /*
        qp::box::ADMMCtrl<Real> ctrl;
        ctrl.rho = rho;
        ctrl.alpha = alpha;
        ctrl.maxIter = maxIter;
        ctrl.absTol = absTol;
        ctrl.relTol = relTol;
        ctrl.inv = inv;
        ctrl.print = progress;

        DistMatrix<Real> X;
        nnls::ADMM( A, B, X, ctrl );
        if( print )
            Print( X, "X" );
        */

        DistMatrix<Real> X;
        NNLS( A, B, X );
        if( print )
            Print( X, "X" );

        const double BNorm = FrobeniusNorm( B );
        Gemm( NORMAL, NORMAL, Real(-1), A, X, Real(1), B );
        const double ENorm = FrobeniusNorm( B );
        if( mpi::WorldRank() == 0 )
            cout << "|| B - A X ||_2 / || B ||_2 = " << ENorm/BNorm << endl;
    }
    catch( exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
