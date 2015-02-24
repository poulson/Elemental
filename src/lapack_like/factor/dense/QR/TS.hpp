/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_QR_TS_HPP
#define EL_QR_TS_HPP

namespace El {
namespace qr {
namespace ts {

template<typename F>
void Reduce( const AbstractDistMatrix<F>& A, TreeData<F>& treeData )
{
    DEBUG_ONLY(
        CallStackEntry cse("qr::ts::Reduce");
        if( A.RowDist() != STAR )
            LogicError("Invalid row distribution for TSQR");
    )
    const Int m =  A.Height();
    const Int n = A.Width();
    const mpi::Comm colComm = A.ColComm();
    const Int p = mpi::Size( colComm );
    if( p == 1 )
        return;
    const Int rank = mpi::Rank( colComm );
    if( m < p*n ) 
        LogicError("TSQR currently assumes height >= width*numProcesses");
    if( !PowerOfTwo(p) )
        LogicError("TSQR currently requires power-of-two number of processes");
    const Int logp = Log2(p);
    auto lastZ = treeData.QR0( IR(0,n), IR(0,n) );
    treeData.QRList.resize( logp );
    treeData.tList.resize( logp );
    treeData.dList.resize( logp );

    // Run the binary tree reduction
    Matrix<F> ZTop(n,n,n), ZBot(n,n,n);
    for( Int stage=0; stage<logp; ++stage )
    {
        // Pack, then send and receive n x n matrices
        const Int partner = Unsigned(rank) ^ (Unsigned(1)<<stage);
        const bool top = rank < partner;
        if( top )
        {
            ZTop = lastZ;
            MakeTrapezoidal( UPPER, ZTop );
            mpi::Recv( ZBot.Buffer(), n*n, partner, colComm );
        }
        else
        {
            ZBot = lastZ;
            MakeTrapezoidal( UPPER, ZBot );
            mpi::Send( ZBot.LockedBuffer(), n*n, partner, colComm );
            break;
        }

        auto& Q = treeData.QRList[stage];
        auto& t = treeData.tList[stage];
        auto& d = treeData.dList[stage];
        Q.Resize( 2*n, n, 2*n );
        t.Resize( n, 1 );
        d.Resize( n, 1 );
        auto QTop = Q( IR(0,n),   IR(0,n) );
        auto QBot = Q( IR(n,2*n), IR(0,n) );
        QTop = ZTop;
        QBot = ZBot;

        // Note that the last QR is not performed by this routine, as many
        // higher-level routines, such as TS-SVT, are simplified if the final
        // small matrix is left alone.
        if( stage < logp-1 )
        {
            // TODO: Exploit double-triangular structure
            QR( Q, t, d );
            lastZ = Q( IR(0,n), IR(0,n) );
        }
    }
}

template<typename F>
Matrix<F>&
RootQR( const AbstractDistMatrix<F>& A, TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root QR");
    if( p == 1 )
        return treeData.QR0;
    else
        return treeData.QRList.back();
}

template<typename F>
const Matrix<F>&
RootQR( const AbstractDistMatrix<F>& A, const TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root QR");
    if( p == 1 )
        return treeData.QR0;
    else
        return treeData.QRList.back();
}

template<typename F>
inline Matrix<F>&
RootPhases( const AbstractDistMatrix<F>& A, TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root phases");
    if( p == 1 )
        return treeData.t0;
    else
        return treeData.tList.back();
}

template<typename F>
inline const Matrix<F>&
RootPhases( const AbstractDistMatrix<F>& A, const TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root phases");
    if( p == 1 )
        return treeData.t0;
    else
        return treeData.tList.back();
}

template<typename F>
inline Matrix<Base<F>>&
RootSignature( const AbstractDistMatrix<F>& A, TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root signature");
    if( p == 1 )
        return treeData.d0;
    else
        return treeData.dList.back();
}

template<typename F>
inline const Matrix<Base<F>>&
RootSignature( const AbstractDistMatrix<F>& A, const TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    const Int rank = mpi::Rank( A.ColComm() );
    if( rank != 0 )
        LogicError("This process does not have access to the root signature");
    if( p == 1 )
        return treeData.d0;
    else
        return treeData.dList.back();
}

template<typename F>
void Scatter( AbstractDistMatrix<F>& A, const TreeData<F>& treeData )
{
    DEBUG_ONLY(
        CallStackEntry cse("qr::ts::Scatter");
        if( A.RowDist() != STAR )
            LogicError("Invalid row distribution for TSQR");
    )
    const Int m =  A.Height();
    const Int n = A.Width();
    const mpi::Comm colComm = A.ColComm();
    const Int p = mpi::Size( colComm );
    if( p == 1 )
        return;
    const Int rank = mpi::Rank( colComm );
    if( m < p*n ) 
        LogicError("TSQR currently assumes height >= width*numProcesses");
    if( !PowerOfTwo(p) )
        LogicError("TSQR currently requires power-of-two number of processes");
    const Int logp = Log2(p);

    // Run the binary tree scatter
    Matrix<F> Z(2*n,n,2*n), ZHalf(n,n,n);
    if( rank == 0 )
        Z = RootQR( A, treeData );
    auto ZTop = Z( IR(0,n),   IR(0,n) );
    auto ZBot = Z( IR(n,2*n), IR(0,n) );
    for( Int revStage=0; revStage<logp; ++revStage )
    {
        const Int stage = (logp-1)-revStage;
        // Skip this stage if the first stage bits of our rank are not zero
        if( stage>0 && (Unsigned(rank) & ((Unsigned(1)<<stage)-1)) )
            continue;

        const Int partner = rank ^ (1u<<stage);
        const bool top = rank < partner;
        if( top )
        {
            if( stage < logp-1 )
            {
                // Multiply by the current Q
                ZTop = ZHalf;        
                Zero( ZBot );
                // TODO: Exploit sparsity?
                ApplyQ
                ( LEFT, NORMAL, 
                  treeData.QRList[stage], treeData.tList[stage], 
                  treeData.dList[stage], Z );
            }
            // Send bottom-half to partner and keep top half
            ZHalf = ZBot;
            mpi::Send( ZHalf.LockedBuffer(), n*n, partner, colComm );
            ZHalf = ZTop; 
        }
        else
        {
            // Recv top half from partner
            mpi::Recv( ZHalf.Buffer(), n*n, partner, colComm );
        }
    }

    // Apply the initial Q
    Zero( A.Matrix() );
    auto ATop = A.Matrix()( IR(0,n), IR(0,n) );
    ATop = ZHalf;
    // TODO: Exploit sparsity
    ApplyQ( LEFT, NORMAL, treeData.QR0, treeData.t0, treeData.d0, A.Matrix() );
}

template<typename F>
inline DistMatrix<F,STAR,STAR>
FormR( const AbstractDistMatrix<F>& A, const TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Grid& g = A.Grid();
    DistMatrix<F,CIRC,CIRC> RRoot(g);
    if( A.ColRank() == 0 )
    {
        const Int n = A.Width();
        auto R = RootQR(A,treeData);
        auto RTop = R( IR(0,n), IR(0,n) );
        RRoot.CopyFromRoot( RTop );
        MakeTrapezoidal( UPPER, RRoot );
    }
    else
        RRoot.CopyFromNonRoot();
    DistMatrix<F,STAR,STAR> R(g);
    R = RRoot;
    return R;
}

// NOTE: This is destructive
template<typename F>
inline void
FormQ( AbstractDistMatrix<F>& A, TreeData<F>& treeData )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    const Int p = mpi::Size( A.ColComm() );
    if( p == 1 )
    {
        A.Matrix() = treeData.QR0;
        ExpandPackedReflectors
        ( LOWER, VERTICAL, CONJUGATED, 0,
          A.Matrix(), RootPhases(A,treeData) );
        DiagonalScale( RIGHT, NORMAL, RootSignature(A,treeData), A.Matrix() );
    }
    else
    {
        if( A.ColRank() == 0 )
        {
            ExpandPackedReflectors
            ( LOWER, VERTICAL, CONJUGATED, 0, 
              RootQR(A,treeData), RootPhases(A,treeData) );
            DiagonalScale
            ( RIGHT, NORMAL, RootSignature(A,treeData), RootQR(A,treeData) );
        }
        Scatter( A, treeData );
    }
}

} // namespace ts

template<typename F>
TreeData<F> TS( const AbstractDistMatrix<F>& A )
{
    if( A.RowDist() != STAR )
        LogicError("Invalid row distribution for TSQR");
    TreeData<F> treeData;
    treeData.QR0 = A.LockedMatrix();
    QR( treeData.QR0, treeData.t0, treeData.d0 );

    const Int p = mpi::Size( A.ColComm() );
    if( p != 1 )
    {
        ts::Reduce( A, treeData );
        if( A.ColRank() == 0 )
            QR
            ( ts::RootQR(A,treeData), ts::RootPhases(A,treeData), 
              ts::RootSignature(A,treeData) );
    }
    return treeData;
}

template<typename F>
void ExplicitTS( AbstractDistMatrix<F>& A, AbstractDistMatrix<F>& R )
{
    auto treeData = TS( A );
    Copy( ts::FormR( A, treeData ), R );
    ts::FormQ( A, treeData );
}

} // namespace qr
} // namespace El

#endif // ifndef EL_QR_TS_HPP
