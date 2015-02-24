/*
   Copyright (c) 2009-2015, Jack Poulson, Lexing Ying,
   The University of Texas at Austin, Stanford University, and the
   Georgia Insitute of Technology.
   All rights reserved.
 
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
using namespace El;

template<typename F>
void MakeFrontsUniform( SymmFront<F>& front )
{
    ChangeFrontType( front, SYMM_2D );
    MakeUniform( front.L );
    for( SymmFront<F>* child : front.children )
        MakeFrontsUniform( *child );
}

template<typename F>
void MakeFrontsUniform( DistSymmFront<F>& front )
{
    ChangeFrontType( front, SYMM_2D );
    MakeUniform( front.L2D );
    if( front.child != nullptr )
        MakeFrontsUniform( *front.child );
    else
        MakeFrontsUniform( *front.duplicate );
}

int main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commRank = mpi::Rank( comm );

    try
    {
        const Int n1 = Input("--n1","first grid dimension",30);
        const Int n2 = Input("--n2","second grid dimension",30);
        const Int n3 = Input("--n3","third grid dimension",30);
        const Int numRepeats = Input
            ("--numRepeats","number of repeated factorizations",5);
        const bool intraPiv = Input("--intraPiv","frontal pivoting?",false);
        const bool sequential = Input
            ("--sequential","sequential partitions?",true);
        const int numDistSeps = Input
            ("--numDistSeps",
             "number of partitions to try per distributed partition",1);
        const int numSeqSeps = Input
            ("--numSeqSeps",
             "number of partitions to try per sequential partition",1);
        const Int cutoff = Input("--cutoff","cutoff for nested dissection",128);
        const bool print = Input("--print","print matrix?",false);
        const bool display = Input("--display","display matrix?",false);
        ProcessInput();

        BisectCtrl ctrl;
        ctrl.sequential = sequential;
        ctrl.numSeqSeps = numSeqSeps;
        ctrl.numDistSeps = numDistSeps;
        ctrl.cutoff = cutoff;

        const int N = n1*n2*n3;
        DistSparseMatrix<double> A(comm);
        Laplacian( A, n1, n2, n3 );
        Scale( -1., A );
        if( display )
        {
            Display( A );
            Display( A.DistGraph() );
        }
        if( print )
        {
            Print( A );
            Print( A.DistGraph() );
        }

        if( commRank == 0 )
            cout << "Running nested dissection..." << endl;
        const double nestedStart = mpi::Time();
        const DistGraph& graph = A.DistGraph();
        DistSymmNodeInfo info;
        DistSeparator sep;
        DistMap map, invMap;
        NestedDissection( graph, map, sep, info, ctrl );
        InvertMap( map, invMap );
        mpi::Barrier( comm );
        const double nestedStop = mpi::Time();
        if( commRank == 0 )
            cout << nestedStop-nestedStart << " seconds" << endl;

        const Int rootSepSize = info.size;
        if( commRank == 0 )
            cout << rootSepSize << " vertices in root separator\n" << endl;

        if( commRank == 0 )
            cout << "Building DistSymmFront tree..." << endl;
        mpi::Barrier( comm );
        const double buildStart = mpi::Time();
        DistSymmFront<double> front( A, map, sep, info, false );
        mpi::Barrier( comm );
        const double buildStop = mpi::Time();
        if( commRank == 0 )
            cout << buildStop-buildStart << " seconds" << endl;

        for( Int repeat=0; repeat<numRepeats; ++repeat )
        {
            if( repeat != 0 )
                MakeFrontsUniform( front );

            if( commRank == 0 )
                cout << "Running LDL^T and redistribution..." << endl;
            mpi::Barrier( comm );
            const double ldlStart = mpi::Time();
            if( intraPiv )
                LDL( info, front, LDL_INTRAPIV_1D );
            else
                LDL( info, front, LDL_1D );
            mpi::Barrier( comm );
            const double ldlStop = mpi::Time();
            if( commRank == 0 )
                cout << ldlStop-ldlStart << " seconds" << endl;

            if( commRank == 0 )
                cout << "Solving against random right-hand side..." << endl;
            const double solveStart = mpi::Time();
            DistMultiVec<double> y( N, 1, comm );
            MakeUniform( y );
            ldl::SolveAfter( invMap, info, front, y );
            mpi::Barrier( comm );
            const double solveStop = mpi::Time();
            if( commRank == 0 )
                cout << "done, " << solveStop-solveStart << " seconds" << endl;

            // TODO: Check residual error
        }
    }
    catch( exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
