/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_HERMITIANEIG_SDC_HPP
#define EL_HERMITIANEIG_SDC_HPP

#include "../Schur/SDC.hpp"

// TODO: Reference to Yuji's work

namespace El {

namespace herm_eig {

using El::schur::ComputePartition;
using El::schur::SplitGrid;
using El::schur::PushSubproblems;
using El::schur::PullSubproblems;

// TODO: Exploit symmetry in A := Q^H A Q. Routine for A := X^H A X?

// G should be a rational function of A. If returnQ=true, G will be set to
// the computed unitary matrix upon exit.
template<typename F>
inline ValueInt<Base<F>>
QDWHDivide( UpperOrLower uplo, Matrix<F>& A, Matrix<F>& G, bool returnQ=false )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::QDWHDivide"))

    // G := sgn(G)
    // G := 1/2 ( G + I )
    PolarCtrl ctrl;
    ctrl.qdwh = true;
    HermitianPolar( uplo, G, ctrl );
    ShiftDiagonal( G, F(1) );
    Scale( F(1)/F(2), G );

    // Compute the pivoted QR decomposition of the spectral projection 
    Matrix<F> t;
    Matrix<Base<F>> d;
    Matrix<Int> p;
    El::QR( G, t, d, p );

    // A := Q^H A Q
    MakeHermitian( uplo, A );
    const Base<F> oneA = OneNorm( A );
    if( returnQ )
    {
        ExpandPackedReflectors( LOWER, VERTICAL, CONJUGATED, 0, G, t );
        DiagonalScale( RIGHT, NORMAL, d, G );
        Matrix<F> B;
        Gemm( ADJOINT, NORMAL, F(1), G, A, B );
        Gemm( NORMAL, NORMAL, F(1), B, G, A );
    }
    else
    {
        qr::ApplyQ( LEFT, ADJOINT, G, t, d, A );
        qr::ApplyQ( RIGHT, NORMAL, G, t, d, A );
    }

    // Return || E21 ||1 / || A ||1 and the chosen rank
    auto part = ComputePartition( A );
    part.value /= oneA;
    return part;
}

template<typename F>
inline ValueInt<Base<F>>
QDWHDivide
( UpperOrLower uplo, DistMatrix<F>& A, DistMatrix<F>& G, bool returnQ=false )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::QDWHDivide"))

    // G := sgn(G)
    // G := 1/2 ( G + I )
    PolarCtrl ctrl;
    ctrl.qdwh = true;
    HermitianPolar( uplo, G, ctrl );
    ShiftDiagonal( G, F(1) );
    Scale( F(1)/F(2), G );

    // Compute the pivoted QR decomposition of the spectral projection 
    const Grid& g = A.Grid();
    DistMatrix<F,MD,STAR> t(g);
    DistMatrix<Base<F>,MD,STAR> d(g);
    DistMatrix<Int,VR,STAR> p(g);
    El::QR( G, t, d, p );

    // A := Q^H A Q
    MakeHermitian( uplo, A );
    const Base<F> oneA = OneNorm( A );
    if( returnQ )
    {
        ExpandPackedReflectors( LOWER, VERTICAL, CONJUGATED, 0, G, t );
        DiagonalScale( RIGHT, NORMAL, d, G );
        DistMatrix<F> B(g);
        Gemm( ADJOINT, NORMAL, F(1), G, A, B );
        Gemm( NORMAL, NORMAL, F(1), B, G, A );
    }
    else
    {
        qr::ApplyQ( LEFT, ADJOINT, G, t, d, A );
        qr::ApplyQ( RIGHT, NORMAL, G, t, d, A );
    }

    // Return || E21 ||1 / || A ||1 and the chosen rank
    auto part = ComputePartition( A );
    part.value /= oneA;
    return part;
}

template<typename F>
inline ValueInt<Base<F>>
RandomizedSignDivide
( UpperOrLower uplo, Matrix<F>& A, Matrix<F>& G, bool returnQ, 
  const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::RandomizedSignDivide"))

    typedef Base<F> Real;
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const Real oneA = OneNorm( A );
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*lapack::MachineEpsilon<Real>();

    // S := sgn(G)
    // S := 1/2 ( S + I )
    auto S( G );
    PolarCtrl polarCtrl;
    polarCtrl.qdwh = true;
    HermitianPolar( uplo, S, polarCtrl );
    ShiftDiagonal( S, F(1) );
    Scale( F(1)/F(2), S );

    ValueInt<Real> part;
    Matrix<F> V, B, t;
    Matrix<Base<F>> d;
    Int it=0;
    while( it < ctrl.maxInnerIts )
    {
        G = S;

        // Compute the RURV of the spectral projector
        ImplicitHaar( V, t, d, n );
        qr::ApplyQ( RIGHT, NORMAL, V, t, d, G );
        El::QR( G, t, d );

        // A := Q^H A Q [and reuse space for V for keeping original A]
        V = A;
        if( returnQ )
        {
            ExpandPackedReflectors( LOWER, VERTICAL, CONJUGATED, 0, G, t );
            DiagonalScale( RIGHT, NORMAL, d, G );
            Gemm( ADJOINT, NORMAL, F(1), G, A, B );
            Gemm( NORMAL, NORMAL, F(1), B, G, A );
        }
        else
        {
            qr::ApplyQ( LEFT, ADJOINT, G, t, d, A );
            qr::ApplyQ( RIGHT, NORMAL, G, t, d, A );
        }

        // || E21 ||1 / || A ||1 and the chosen rank
        part = ComputePartition( A );
        part.value /= oneA;

        ++it;
        if( part.value <= tol || it == ctrl.maxInnerIts )
            break;
        else
            A = V;
    }
    return part;
}

template<typename F>
inline ValueInt<Base<F>>
RandomizedSignDivide
( UpperOrLower uplo, DistMatrix<F>& A, DistMatrix<F>& G, bool returnQ, 
  const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::RandomizedSignDivide"))

    typedef Base<F> Real;
    const Grid& g = A.Grid();
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const Real oneA = OneNorm( A );
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*lapack::MachineEpsilon<Real>();

    // S := sgn(G)
    // S := 1/2 ( S + I )
    auto S( G );
    PolarCtrl polarCtrl;
    polarCtrl.qdwh = true;
    HermitianPolar( uplo, G, polarCtrl );
    ShiftDiagonal( S, F(1) );
    Scale( F(1)/F(2), S );

    ValueInt<Real> part;
    DistMatrix<F> V(g), B(g);
    DistMatrix<F,MD,STAR> t(g);
    DistMatrix<Base<F>,MD,STAR> d(g);
    Int it=0;
    while( it < ctrl.maxInnerIts )
    {
        G = S;

        // Compute the RURV of the spectral projector
        ImplicitHaar( V, t, d, n );
        qr::ApplyQ( RIGHT, NORMAL, V, t, d, G );
        El::QR( G, t, d );

        // A := Q^H A Q [and reuse space for V for keeping original A]
        V = A;
        if( returnQ )
        {
            ExpandPackedReflectors( LOWER, VERTICAL, CONJUGATED, 0, G, t );
            DiagonalScale( RIGHT, NORMAL, d, G );
            Gemm( ADJOINT, NORMAL, F(1), G, A, B );
            Gemm( NORMAL, NORMAL, F(1), B, G, A );
        }
        else
        {
            qr::ApplyQ( LEFT, ADJOINT, G, t, d, A );
            qr::ApplyQ( RIGHT, NORMAL, G, t, d, A );
        }

        // || E21 ||1 / || A ||1 and the chosen rank
        part = ComputePartition( A );
        part.value /= oneA;

        ++it;
        if( part.value <= tol || it == ctrl.maxInnerIts )
            break;
        else
            A = V;
    }
    return part;
}

template<typename F>
inline ValueInt<Base<F>>
SpectralDivide
( UpperOrLower uplo, Matrix<F>& A, const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SpectralDivide"))

    typedef Base<F> Real;
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const auto median = Median(GetRealPartOfDiagonal(A));
    const Real infNorm = InfinityNorm(A);
    const Real eps = lapack::MachineEpsilon<Real>();
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*eps;
    const Real spread = ctrl.spreadFactor*infNorm;

    Int it=0;
    ValueInt<Real> part;
    Matrix<F> G, ACopy;
    if( ctrl.maxOuterIts > 1 )
        ACopy = A;
    while( it < ctrl.maxOuterIts )
    {
        const Real shift = SampleBall<Real>(-median.value,spread);

        G = A;
        ShiftDiagonal( G, F(shift) );

        part = RandomizedSignDivide( uplo, A, G, false, ctrl );

        ++it;
        if( part.value <= tol )
            break;
        else if( it != ctrl.maxOuterIts )
            A = ACopy;
    }
    if( part.value > tol )
        RuntimeError
        ( "Unable to split spectrum to specified accuracy: part.value=",
          part.value, ", tol=", tol );

    return part;
}

template<typename F>
inline ValueInt<Base<F>>
SpectralDivide
( UpperOrLower uplo, Matrix<F>& A, Matrix<F>& Q, 
  const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SpectralDivide"))

    typedef Base<F> Real;
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const auto median = Median(GetRealPartOfDiagonal(A));
    const Real infNorm = InfinityNorm(A);
    const Real eps = lapack::MachineEpsilon<Real>();
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*eps;
    const Real spread = ctrl.spreadFactor*infNorm;

    Int it=0;
    ValueInt<Real> part;
    Matrix<F> ACopy;
    if( ctrl.maxOuterIts > 1 )
        ACopy = A;
    while( it < ctrl.maxOuterIts )
    {
        const Real shift = SampleBall<Real>(-median.value,spread);

        Q = A;
        ShiftDiagonal( Q, F(shift) );

        part = RandomizedSignDivide( uplo, A, Q, true, ctrl );

        ++it;
        if( part.value <= tol )
            break;
        else if( it != ctrl.maxOuterIts )
            A = ACopy;
    }
    if( part.value > tol )
        RuntimeError
        ( "Unable to split spectrum to specified accuracy: part.value=",
          part.value, ", tol=", tol );

    return part;
}

template<typename F>
inline ValueInt<Base<F>>
SpectralDivide
( UpperOrLower uplo, DistMatrix<F>& A, 
  const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SpectralDivide"))

    typedef Base<F> Real;
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const auto median = Median(GetRealPartOfDiagonal(A));
    const Real infNorm = InfinityNorm(A);
    const Real eps = lapack::MachineEpsilon<Real>();
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*eps;
    const Real spread = ctrl.spreadFactor*infNorm;

    Int it=0;
    ValueInt<Real> part;
    DistMatrix<F> ACopy(A.Grid()), G(A.Grid());
    if( ctrl.maxOuterIts > 1 )
        ACopy = A;
    while( it < ctrl.maxOuterIts )
    {
        Real shift = SampleBall<Real>(-median.value,spread);
        mpi::Broadcast( shift, 0, A.Grid().VCComm() );

        G = A;
        ShiftDiagonal( G, F(shift) );

        part = RandomizedSignDivide( uplo, A, G, false, ctrl );

        ++it;
        if( part.value <= tol )
            break;
        else if( it != ctrl.maxOuterIts )
            A = ACopy;
    }
    if( part.value > tol )
        RuntimeError
        ( "Unable to split spectrum to specified accuracy: part.value=",
          part.value, ", tol=", tol );

    return part;
}

template<typename F>
inline ValueInt<Base<F>>
SpectralDivide
( UpperOrLower uplo, DistMatrix<F>& A, DistMatrix<F>& Q, 
  const HermitianSDCCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SpectralDivide"))

    typedef Base<F> Real;
    const Int n = A.Height();
    MakeHermitian( uplo, A );
    const Real infNorm = InfinityNorm(A);
    const auto median = Median(GetRealPartOfDiagonal(A));
    const Real eps = lapack::MachineEpsilon<Real>();
    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = 500*n*eps;
    const Real spread = ctrl.spreadFactor*infNorm;

    Int it=0;
    ValueInt<Real> part;
    DistMatrix<F> ACopy(A.Grid());
    if( ctrl.maxOuterIts > 1 )
        ACopy = A;
    while( it < ctrl.maxOuterIts )
    {
        Real shift = SampleBall<Real>(-median.value,spread);
        mpi::Broadcast( shift, 0, A.Grid().VCComm() );

        Q = A;
        ShiftDiagonal( Q, F(shift) );

        part = RandomizedSignDivide( uplo, A, Q, true, ctrl );

        ++it;
        if( part.value <= tol )
            break;
        else if( it != ctrl.maxOuterIts )
            A = ACopy;
    }
    if( part.value > tol )
        RuntimeError
        ( "Unable to split spectrum to specified accuracy: part.value=",
          part.value, ", tol=", tol );

    return part;
}

template<typename F>
void SDC
( UpperOrLower uplo, Matrix<F>& A, Matrix<Base<F>>& w, 
  const HermitianSDCCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SDC"))

    typedef Base<F> Real;
    const Int n = A.Height();
    w.Resize( n, 1 );
    if( n <= ctrl.cutoff )
    {
        HermitianEig( uplo, A, w );
        return;
    }

    // Perform this level's split
    const auto part = SpectralDivide( uplo, A, ctrl );
    Matrix<F> ATL, ATR,
              ABL, ABR;
    PartitionDownDiagonal
    ( A, ATL, ATR,
         ABL, ABR, part.index );
    if( uplo == LOWER )
        Zero( ABL );
    else
        Zero( ATR );
    Matrix<Real> wT, wB;
    PartitionDown( w, wT, wB, part.index );

    // Recurse on the two subproblems
    SDC( uplo, ATL, wT, ctrl );
    SDC( uplo, ABR, wB, ctrl );
}

template<typename F>
void SDC
( UpperOrLower uplo, Matrix<F>& A, Matrix<Base<F>>& w, Matrix<F>& Q, 
  const HermitianSDCCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SDC"))

    typedef Base<F> Real;
    const Int n = A.Height();
    w.Resize( n, 1 );
    Q.Resize( n, n );
    if( n <= ctrl.cutoff )
    {
        HermitianEig( uplo, A, w, Q );
        return;
    }

    // Perform this level's split
    const auto part = SpectralDivide( uplo, A, Q, ctrl );
    Matrix<F> ATL, ATR,
              ABL, ABR;
    PartitionDownDiagonal
    ( A, ATL, ATR,
         ABL, ABR, part.index );
    if( uplo == LOWER )
        Zero( ABL );
    else
        Zero( ATR );
    Matrix<Real> wT, wB;
    PartitionDown( w, wT, wB, part.index );
    Matrix<F> QL, QR;
    PartitionRight( Q, QL, QR, part.index );

    // Recurse on the top-left quadrant and update eigenvectors
    Matrix<F> Z;
    SDC( uplo, ATL, wT, Z, ctrl );
    auto G( QL );
    Gemm( NORMAL, NORMAL, F(1), G, Z, QL );

    // Recurse on the bottom-right quadrant and update eigenvectors
    SDC( uplo, ABR, wB, Z, ctrl );
    G = QR;
    Gemm( NORMAL, NORMAL, F(1), G, Z, QR );
}

template<typename F>
void SDC
( UpperOrLower uplo, AbstractDistMatrix<F>& APre, 
  AbstractDistMatrix<Base<F>>& wPre, 
  const HermitianSDCCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SDC"))

    typedef Base<F> Real;
    const Grid& g = APre.Grid();
    const Int n = APre.Height();
    wPre.Resize( n, 1 );
    if( APre.Grid().Size() == 1 )
    {
        HermitianEig( uplo, APre.Matrix(), wPre.Matrix() );
        return;
    }
    if( n <= ctrl.cutoff )
    {
        HermitianEig( uplo, APre, wPre );
        return;
    }

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );  auto& A = *APtr;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre ); auto& w = *wPtr;

    // Perform this level's split
    const auto part = SpectralDivide( uplo, A, ctrl );
    DistMatrix<F> ATL(g), ATR(g),
                  ABL(g), ABR(g);
    PartitionDownDiagonal
    ( A, ATL, ATR,
         ABL, ABR, part.index );
    if( uplo == LOWER )
        Zero( ABL );
    else
        Zero( ATR );
    DistMatrix<Real,VR,STAR> wT(g), wB(g);
    PartitionDown( w, wT, wB, part.index );

    // Recurse on the two subproblems
    DistMatrix<F> ATLSub, ABRSub;
    DistMatrix<Real,VR,STAR> wTSub, wBSub;
    PushSubproblems
    ( ATL, ABR, ATLSub, ABRSub, wT, wB, wTSub, wBSub, ctrl.progress );
    if( ATL.Participating() )
        SDC( uplo, ATLSub, wTSub, ctrl );
    if( ABR.Participating() )
        SDC( uplo, ABRSub, wBSub, ctrl );
    PullSubproblems( ATL, ABR, ATLSub, ABRSub, wT, wB, wTSub, wBSub );
}

template<typename F>
void SDC
( UpperOrLower uplo, 
  AbstractDistMatrix<F>& APre, AbstractDistMatrix<Base<F>>& wPre, 
  AbstractDistMatrix<F>& QPre, 
  const HermitianSDCCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("herm_eig::SDC"))

    typedef Base<F> Real;
    const Grid& g = APre.Grid();
    const Int n = APre.Height();
    wPre.Resize( n, 1 );
    QPre.Resize( n, n );
    if( APre.Grid().Size() == 1 )
    {
        HermitianEig( uplo, APre.Matrix(), wPre.Matrix(), QPre.Matrix() );
        return;
    }
    if( n <= ctrl.cutoff )
    {
        HermitianEig( uplo, APre, wPre, QPre );
        return;
    }

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );  auto& A = *APtr;
    auto QPtr = WriteProxy<F,MC,MR>( &QPre );      auto& Q = *QPtr;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre ); auto& w = *wPtr;

    // Perform this level's split
    const auto part = SpectralDivide( uplo, A, Q, ctrl );
    DistMatrix<F> ATL(g), ATR(g),
                  ABL(g), ABR(g);
    PartitionDownDiagonal
    ( A, ATL, ATR,
         ABL, ABR, part.index );
    if( uplo == LOWER )
        Zero( ABL );
    else
        Zero( ATR );
    DistMatrix<Real,VR,STAR> wT(g), wB(g);
    PartitionDown( w, wT, wB, part.index );
    DistMatrix<F> QL(g), QR(g);
    PartitionRight( Q, QL, QR, part.index );

    // Recurse on the two subproblems
    DistMatrix<F> ATLSub, ABRSub, ZTSub, ZBSub;
    DistMatrix<Real,VR,STAR> wTSub, wBSub;
    PushSubproblems
    ( ATL, ABR, ATLSub, ABRSub, wT, wB, wTSub, wBSub, ZTSub, ZBSub, 
      ctrl.progress );
    if( ATLSub.Participating() )
        SDC( uplo, ATLSub, wTSub, ZTSub, ctrl );
    if( ABRSub.Participating() )
        SDC( uplo, ABRSub, wBSub, ZBSub, ctrl );

    // Pull the results back to this grid
    DistMatrix<F> ZT(g), ZB(g);
    PullSubproblems
    ( ATL, ABR, ATLSub, ABRSub, wT, wB, wTSub, wBSub, ZT, ZB, ZTSub, ZBSub );

    // Update the eigen vectors
    auto G( QL );
    Gemm( NORMAL, NORMAL, F(1), G, ZT, QL );
    G = QR;
    Gemm( NORMAL, NORMAL, F(1), G, ZB, QR );
}

} // namespace herm_eig
} // namespace El

#endif // ifndef EL_HERMITIANEIG_SDC_HPP
