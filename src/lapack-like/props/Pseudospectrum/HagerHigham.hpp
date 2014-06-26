/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_PSEUDOSPECTRUM_HAGERHIGHAM_HPP
#define EL_PSEUDOSPECTRUM_HAGERHIGHAM_HPP

#include "./Power.hpp"

namespace El {
namespace pspec {

template<typename Real>
inline Matrix<Int>
OneNormConvergenceTest
(       Matrix<Complex<Real>>& activeX,
  const Matrix<Complex<Real>>& activeY,
  const Matrix<Complex<Real>>& activeZ,
        Matrix<Real>& activeEsts,
        Matrix<Int >& activeItCounts,
        Int numIts )
{
    DEBUG_ONLY(CallStackEntry cse("pspec::OneNormConvergenceTest"))
    typedef Complex<Real> C;
    const Int n = activeX.Height();

    const Int numActiveShifts=activeEsts.Height();
    Matrix<Int> activeConverged;
    Zeros( activeConverged, numActiveShifts, 1 );

    // Compute the infinity norm of each column of Z and its location
    std::vector<ValueInt<Real>> valueInts(numActiveShifts);
    for( Int j=0; j<numActiveShifts; ++j )
    {
        valueInts[j].index = 0;
        valueInts[j].value = 0;
        for( Int i=0; i<n; ++i )
        {
            const Real absVal = Abs(activeZ.Get(i,j));
            if( absVal > valueInts[j].value )
            {
                valueInts[j].value = absVal;
                valueInts[j].index = i;
            }
        }
    }

    // Compute the real parts of the inner products of each column of Z and X
    // NOTE: Except in the first iteration, each column of X should only have
    //       a single nonzero entry, so this can be greatly accelerated.
    std::vector<Real> innerProds(numActiveShifts);
    for( Int j=0; j<numActiveShifts; ++j )
    {
        const C innerProd = 
          blas::Dot
           ( n, activeZ.LockedBuffer(0,j), 1, 
                activeX.LockedBuffer(0,j), 1 );
        innerProds[j] = RealPart(innerProd);
    }

    // Check for convergence
    for( Int j=0; j<numActiveShifts; ++j )
    {
        const Real gamma = blas::Nrm1( n, activeY.LockedBuffer(0,j), 1 );
        activeEsts.Set( j, 0, gamma );
        if( numIts > 0 && valueInts[j].value <= innerProds[j] )
        {
            activeConverged.Set( j, 0, 1 );
        }
        else
        {
            for( Int i=0; i<n; ++i )
                activeX.Set( i, j, 0 );
            activeX.Set( valueInts[j].index, j, 1 );
            activeItCounts.Update( j, 0, 1 );
        }
    }
    return activeConverged;
}

template<typename Real>
inline DistMatrix<Int,MR,STAR>
OneNormConvergenceTest
(       DistMatrix<Complex<Real>>& activeX,
  const DistMatrix<Complex<Real>>& activeY,
  const DistMatrix<Complex<Real>>& activeZ,
        DistMatrix<Real,MR,STAR>& activeEsts,
        DistMatrix<Int ,VR,STAR>& activeItCounts,
        Int numIts )
{
    DEBUG_ONLY(
        CallStackEntry cse("pspec::OneNormConvergenceTest");
        if( activeX.Height() != activeY.Height() || 
            activeY.Height() != activeZ.Height() )
            LogicError("active{X,Y,Z} should be the same height");
        if( activeX.Width() != activeY.Width() || 
            activeY.Width() != activeZ.Width() )
            LogicError("active{X,Y,Z} should be the same height");
        if( activeX.ColAlign() != activeY.ColAlign() || 
            activeY.ColAlign() != activeZ.ColAlign() )
            LogicError("active{X,Y,Z} should be column aligned");
        if( activeX.RowAlign() != activeY.RowAlign() || 
            activeY.RowAlign() != activeZ.RowAlign() )
            LogicError("active{X,Y,Z} should be row aligned");
        if( activeZ.RowAlign() != activeEsts.ColAlign() )
            LogicError("Invalid activeZ alignment");
        if( activeItCounts.ColAlign()%activeEsts.ColStride() !=
            activeEsts.ColAlign() )
            LogicError("Invalid column alignment");
    )
    typedef Complex<Real> C;
    const Int nLoc = activeX.LocalHeight();

    const Int numActiveShifts=activeEsts.Height();
    DistMatrix<Int,MR,STAR> activeConverged( activeEsts.Grid() );
    activeConverged.AlignWith( activeEsts );
    Zeros( activeConverged, numActiveShifts, 1 );

    // Compute the infinity norm of each local column of Z and its location
    const Int numLocShifts = activeEsts.LocalHeight();
    std::vector<ValueInt<Real>> valueInts(numLocShifts);
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        valueInts[jLoc].value = 0;
        valueInts[jLoc].index = 0;
        for( Int iLoc=0; iLoc<nLoc; ++iLoc )
        {
            const Real absVal = Abs(activeZ.GetLocal(iLoc,jLoc));
            if( absVal > valueInts[jLoc].value )
            {
                valueInts[jLoc].value = absVal;
                valueInts[jLoc].index = activeZ.GlobalRow(iLoc);
            }
        }
    }
    mpi::AllReduce
    ( valueInts.data(), numLocShifts, mpi::MaxLocOp<Real>(), 
      activeZ.ColComm() );

    // Compute the real parts of the inner products of each column of Z and X
    // NOTE: Except in the first iteration, each column of X should only have
    //       a single nonzero entry, so this can be greatly accelerated.
    std::vector<Real> innerProds(numLocShifts);
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        const C innerProd = 
          blas::Dot
           ( nLoc, activeZ.LockedBuffer(0,jLoc), 1, 
                   activeX.LockedBuffer(0,jLoc), 1 );
        innerProds[jLoc] = RealPart(innerProd);
    }
    mpi::AllReduce
    ( innerProds.data(), numLocShifts, mpi::SUM, activeZ.ColComm() );

    // Compute the one norms
    std::vector<Real> oneNorms(numLocShifts);
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
        oneNorms[jLoc] = blas::Nrm1( nLoc, activeY.LockedBuffer(0,jLoc), 1 );
    mpi::AllReduce
    ( oneNorms.data(), numLocShifts, mpi::SUM, activeY.ColComm() );

    // Check for convergence
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        activeEsts.SetLocal( jLoc, 0, oneNorms[jLoc] );
        if( numIts > 0 && valueInts[jLoc].value <= innerProds[jLoc] )
        {
            activeConverged.SetLocal( jLoc, 0, 1 );
        }
        else
        {
            for( Int iLoc=0; iLoc<nLoc; ++iLoc )
                activeX.SetLocal( iLoc, jLoc, 0 );
            const Int j = activeEsts.GlobalRow(jLoc);
            activeX.Set( valueInts[jLoc].index, j, 1 );
            activeItCounts.Update( j, 0, 1 );
        }
    }

    return activeConverged;
}

template<typename Real>
inline Matrix<Int>
HagerHigham
( const Matrix<Complex<Real>>& U,
  const Matrix<Complex<Real>>& shifts, 
  Matrix<Real>& invNorms, PseudospecCtrl<Real> psCtrl=PseudospecCtrl<Real>() )
{
    DEBUG_ONLY(CallStackEntry cse("pspec::HagerHigham"))
    using namespace pspec;
    typedef Complex<Real> C;
    const Int n = U.Height();
    const Int numShifts = shifts.Height();

    const Int maxIts = psCtrl.maxIts;
    const bool deflate = psCtrl.deflate;
    const bool progress = psCtrl.progress;

    // Keep track of the number of iterations per shift
    Matrix<Int> itCounts;
    Ones( itCounts, numShifts, 1 );

    // Keep track of the pivoting history if deflation is requested
    Matrix<Int> preimage;
    Matrix<C> pivShifts( shifts );
    if( deflate )
    {
        preimage.Resize( numShifts, 1 );
        for( Int j=0; j<numShifts; ++j )
            preimage.Set( j, 0, j );
    }

    psCtrl.snapCtrl.ResetCounts();

    // MultiShiftTrsm currently requires write access
    Matrix<C> UCopy( U );

    // The Hessenberg case currently requires explicit access to the adjoint
    Matrix<C> UAdj;
    if( !psCtrl.schur )
        Adjoint( U, UAdj );

    // Simultaneously run inverse iteration for various shifts
    Timer timer;
    Matrix<C> X;
    Ones( X, n, numShifts );
    Scale( C(1)/C(n), X );
    Int numIts=0, numDone=0;
    Matrix<Real> estimates(numShifts,1);
    Zeros( estimates, numShifts, 1 );
    Matrix<Int> activePreimage;
    while( true )
    {
        const Int numActive = ( deflate ? numShifts-numDone : numShifts );
        auto activeShifts = View( pivShifts, 0, 0, numActive, 1 );
        auto activeEsts = View( estimates, 0, 0, numActive, 1 );
        auto activeItCounts = View( itCounts, 0, 0, numActive, 1 );
        auto activeX = View( X, 0, 0, n, numActive );
        if( deflate )
            activePreimage = View( preimage, 0, 0, numActive, 1 );

        if( progress )
            timer.Start(); 

        Matrix<C> activeY, activeZ;
        if( psCtrl.schur )
        {
            // Solve against (U - zI)
            activeY = activeX;
            MultiShiftTrsm
            ( LEFT, UPPER, NORMAL, C(1), UCopy, activeShifts, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against (U - zI)^H 
            MultiShiftTrsm
            ( LEFT, UPPER, ADJOINT, C(1), UCopy, activeShifts, activeZ );
        }
        else
        {
            // Solve against (H - zI)
            activeY = activeX;
            MultiShiftHessSolve
            ( UPPER, NORMAL, C(1), U, activeShifts, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against (H - zI)^H
            Matrix<C> activeShiftsConj;
            Conjugate( activeShifts, activeShiftsConj );
            MultiShiftHessSolve
            ( LOWER, NORMAL, C(1), UAdj, activeShiftsConj, activeZ );
        }

        auto activeConverged = 
            OneNormConvergenceTest
            ( activeX, activeY, activeZ, activeEsts, activeItCounts, numIts );
        const Int numActiveDone = ZeroNorm( activeConverged );
        if( deflate )
            numDone += numActiveDone;
        else
            numDone = numActiveDone;
        if( progress )
        {
            const double iterTime = timer.Stop();
            std::cout << "iteration " << numIts << ": " << iterTime 
                      << " seconds, " << numDone << " of " << numShifts 
                      << " converged" << std::endl;
        }

        ++numIts;
        if( numIts >= maxIts )
            break;

        if( numDone == numShifts )
            break;
        else if( deflate && numActiveDone != 0 )
            Deflate
            ( activeShifts, activePreimage, activeX, activeEsts,
              activeConverged, activeItCounts, progress );

        // Save snapshots of the estimates at the requested rate
        psCtrl.snapCtrl.Iterate();
        Snapshot
        ( preimage, estimates, itCounts, numIts, deflate, psCtrl.snapCtrl );
    } 
    invNorms = estimates;
    if( deflate )
        RestoreOrdering( preimage, invNorms, itCounts );

    // Solve one final linear system to attempt to counteract possible
    // cancellation in large entries in inv(U - zI)
    for( Int j=0; j<numShifts; ++j )
        for( Int i=0; i<n; ++i )
            X.Set( i, j, (i%2==0 ?  Real(i+n-1)/Real(n-1)  
                                 : -Real(i+n-1)/Real(n-1) ) );
    if( psCtrl.schur )
        MultiShiftTrsm( LEFT, UPPER, NORMAL, C(1), UCopy, shifts, X );
    else
        MultiShiftHessSolve( UPPER, NORMAL, C(1), U, shifts, X );
    for( Int j=0; j<numShifts; ++j )
    {
        const Real oneNorm = blas::Nrm1( n, X.LockedBuffer(0,j), 1 );
        const Real heurNorm = 2*oneNorm/(3*Real(n));
        if( heurNorm > invNorms.Get(j,0) )
            invNorms.Set( j, 0, heurNorm );
    }

    FinalSnapshot( invNorms, itCounts, psCtrl.snapCtrl );
    return itCounts;
}

template<typename Real>
inline Matrix<Int>
HagerHigham
( const Matrix<Complex<Real>>& U, const Matrix<Complex<Real>>& Q, 
  const Matrix<Complex<Real>>& shifts, 
  Matrix<Real>& invNorms, PseudospecCtrl<Real> psCtrl=PseudospecCtrl<Real>() )
{
    DEBUG_ONLY(CallStackEntry cse("pspec::HagerHigham"))
    using namespace pspec;
    typedef Complex<Real> C;
    const Int n = U.Height();
    const Int numShifts = shifts.Height();

    const Int maxIts = psCtrl.maxIts;
    const bool deflate = psCtrl.deflate;
    const bool progress = psCtrl.progress;

    // Keep track of the number of iterations per shift
    Matrix<Int> itCounts;
    Ones( itCounts, numShifts, 1 );

    // Keep track of the pivoting history if deflation is requested
    Matrix<Int> preimage;
    Matrix<C> pivShifts( shifts );
    if( deflate )
    {
        preimage.Resize( numShifts, 1 );
        for( Int j=0; j<numShifts; ++j )
            preimage.Set( j, 0, j );
    }

    psCtrl.snapCtrl.ResetCounts();

    Matrix<C> UCopy( U );

    // The Hessenberg case currently requires explicit access to the adjoint
    Matrix<C> UAdj;
    if( !psCtrl.schur )
        Adjoint( U, UAdj );

    // Simultaneously run inverse iteration for various shifts
    Timer timer;
    Matrix<C> X;
    Ones( X, n, numShifts );
    Scale( C(1)/C(n), X );
    Int numIts=0, numDone=0;
    Matrix<Real> estimates(numShifts,1);
    Zeros( estimates, numShifts, 1 );
    Matrix<Int> activePreimage;
    while( true )
    {
        const Int numActive = ( deflate ? numShifts-numDone : numShifts );
        auto activeShifts = View( pivShifts, 0, 0, numActive, 1 );
        auto activeEsts = View( estimates, 0, 0, numActive, 1 );
        auto activeItCounts = View( itCounts, 0, 0, numActive, 1 );
        auto activeX = View( X, 0, 0, n, numActive );
        if( deflate )
            activePreimage = View( preimage, 0, 0, numActive, 1 );

        if( progress )
            timer.Start(); 

        Matrix<C> activeV, activeY, activeZ;
        if( psCtrl.schur )
        {
            // Solve against Q (U - zI) Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeX, activeV );
            MultiShiftTrsm
            ( LEFT, UPPER, NORMAL, C(1), UCopy, activeShifts, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against Q (U - zI)^H Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeZ, activeV );
            MultiShiftTrsm
            ( LEFT, UPPER, ADJOINT, C(1), UCopy, activeShifts, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeZ );
        }
        else
        {
            // Solve against Q (H - zI) Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeX, activeV );
            MultiShiftHessSolve
            ( UPPER, NORMAL, C(1), U, activeShifts, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against Q (H - zI)^H Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeZ, activeV );
            Matrix<C> activeShiftsConj;
            Conjugate( activeShifts, activeShiftsConj );
            MultiShiftHessSolve
            ( LOWER, NORMAL, C(1), UAdj, activeShiftsConj, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeZ );
        }

        auto activeConverged = 
            OneNormConvergenceTest
            ( activeX, activeY, activeZ, activeEsts, activeItCounts, numIts );
        const Int numActiveDone = ZeroNorm( activeConverged );
        if( deflate )
            numDone += numActiveDone;
        else
            numDone = numActiveDone;
        if( progress )
        {
            const double iterTime = timer.Stop();
            std::cout << "iteration " << numIts << ": " << iterTime 
                      << " seconds, " << numDone << " of " << numShifts 
                      << " converged" << std::endl;
        }

        ++numIts;
        if( numIts >= maxIts )
            break;

        if( numDone == numShifts )
            break;
        else if( deflate && numActiveDone != 0 )
            Deflate
            ( activeShifts, activePreimage, activeX, activeEsts,
              activeConverged, activeItCounts, progress );

        // Save snapshots of the estimates at the requested rate
        psCtrl.snapCtrl.Iterate();
        Snapshot
        ( preimage, estimates, itCounts, numIts, deflate, psCtrl.snapCtrl );
    } 
    invNorms = estimates;
    if( deflate )
        RestoreOrdering( preimage, invNorms, itCounts );

    // Solve one final linear system to attempt to counteract possible
    // cancellation in large entries in inv(U - zI)
    if( numShifts == 0 )
        return itCounts;
    auto x = View( X, 0, 0, n, 1 );
    for( Int i=0; i<n; ++i )
        x.Set( i, 0, (i%2==0 ?  Real(i+n-1)/Real(n-1) 
                             : -Real(i+n-1)/Real(n-1) ) );
    Matrix<C> yRep;
    Gemv( ADJOINT, C(1), Q, x, yRep );
    Matrix<C> Y( n, numShifts );
    for( Int j=0; j<numShifts; ++j )
    {
        auto y = View( Y, 0, j, n, 1 );    
        y = yRep;
    }
    if( psCtrl.schur )
        MultiShiftTrsm( LEFT, UPPER, NORMAL, C(1), UCopy, shifts, Y );
    else
        MultiShiftHessSolve( UPPER, NORMAL, C(1), U, shifts, Y );
    Gemm( NORMAL, NORMAL, C(1), Q, Y, X );
    for( Int j=0; j<numShifts; ++j )
    {
        const Real oneNorm = blas::Nrm1( n, X.LockedBuffer(0,j), 1 );
        const Real heurNorm = 2*oneNorm/(3*Real(n));
        if( heurNorm > invNorms.Get(j,0) )
            invNorms.Set( j, 0, heurNorm );
    }

    FinalSnapshot( invNorms, itCounts, psCtrl.snapCtrl );
    return itCounts;
}

template<typename Real>
inline DistMatrix<Int,VR,STAR>
HagerHigham
( const DistMatrix<Complex<Real>        >& U, 
  const DistMatrix<Complex<Real>,VR,STAR>& shifts, 
        DistMatrix<Real,         VR,STAR>& invNorms, 
  PseudospecCtrl<Real> psCtrl=PseudospecCtrl<Real>() )
{
    DEBUG_ONLY(CallStackEntry cse("pspec::HagerHigham"))
    using namespace pspec;
    typedef Complex<Real> C;
    const Int n = U.Height();
    const Int nLoc = U.LocalHeight();
    const Int numShifts = shifts.Height();
    const Int numLocShifts = shifts.LocalHeight();
    const Grid& g = U.Grid();

    const Int maxIts = psCtrl.maxIts;
    const bool deflate = psCtrl.deflate;
    const bool progress = psCtrl.progress;

    // Keep track of the number of iterations per shift
    DistMatrix<Int,VR,STAR> itCounts(g);
    Ones( itCounts, numShifts, 1 );

    // Keep track of the pivoting history if deflation is requested
    DistMatrix<Int,VR,STAR> preimage(g);
    DistMatrix<C,  VR,STAR> pivShifts( shifts );
    if( deflate )
    {
        preimage.AlignWith( shifts );
        preimage.Resize( numShifts, 1 );
        for( Int iLoc=0; iLoc<numLocShifts; ++iLoc )
        {
            const Int i = preimage.GlobalRow(iLoc);
            preimage.SetLocal( iLoc, 0, i );
        }
    }

    psCtrl.snapCtrl.ResetCounts();

    // The Hessenberg case currently requires explicit access to the adjoint
    DistMatrix<C,VC,STAR> U_VC_STAR(g), UAdj_VC_STAR(g);
    if( !psCtrl.schur )
    {
        U_VC_STAR = U;
        Adjoint( U, UAdj_VC_STAR );
    }

    // Simultaneously run inverse iteration for various shifts
    Timer timer;
    DistMatrix<C> X(g);
    Ones( X, n, numShifts );
    Scale( C(1)/C(n), X );
    Int numIts=0, numDone=0;
    DistMatrix<Real,MR,STAR> estimates(g);
    estimates.AlignWith( shifts );
    Zeros( estimates, numShifts, 1 );
    DistMatrix<Int,VR,STAR> activePreimage(g);
    while( true )
    {
        const Int numActive = ( deflate ? numShifts-numDone : numShifts );
        auto activeShifts = View( pivShifts, 0, 0, numActive, 1 );
        auto activeEsts = View( estimates, 0, 0, numActive, 1 );
        auto activeItCounts = View( itCounts, 0, 0, numActive, 1 );
        auto activeX = View( X, 0, 0, n, numActive );
        if( deflate )
            activePreimage = View( preimage, 0, 0, numActive, 1 );

        if( progress && g.Rank() == 0 )
            timer.Start();

        DistMatrix<C> activeY(g), activeZ(g);
        activeY.AlignWith( activeX );
        activeZ.AlignWith( activeX );
        if( psCtrl.schur )
        {
            // Solve against (U - zI)
            activeY = activeX;
            MultiShiftTrsm
            ( LEFT, UPPER, NORMAL, C(1), U, activeShifts, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha )
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against (U - zI)^H
            MultiShiftTrsm
            ( LEFT, UPPER, ADJOINT, C(1), U, activeShifts, activeZ );
        }
        else
        {
            // Solve against (H - zI)
            DistMatrix<C,STAR,VR>  activeV_STAR_VR( activeX );
            MultiShiftHessSolve
            ( UPPER, NORMAL, C(1), U_VC_STAR, activeShifts, activeV_STAR_VR );
            activeY = activeV_STAR_VR;

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against (H - zI)^H 
            activeV_STAR_VR = activeZ;
            DistMatrix<C,VR,STAR> activeShiftsConj(g);
            Conjugate( activeShifts, activeShiftsConj );
            MultiShiftHessSolve
            ( LOWER, NORMAL, C(1), UAdj_VC_STAR, activeShiftsConj, 
              activeV_STAR_VR );
            activeZ = activeV_STAR_VR;
        }

        auto activeConverged =
            OneNormConvergenceTest
            ( activeX, activeY, activeZ, activeEsts, activeItCounts, numIts );
        const Int numActiveDone = ZeroNorm( activeConverged );
        if( deflate )
            numDone += numActiveDone;
        else
            numDone = numActiveDone;
        if( progress && g.Rank() == 0 )
        {
            const double iterTime = timer.Stop();
            std::cout << "iteration " << numIts << ": " << iterTime 
                      << " seconds, " << numDone << " of " << numShifts 
                      << " converged" << std::endl;
        }

        ++numIts;
        if( numIts >= maxIts )
            break;

        if( numDone == numShifts )
            break;
        else if( deflate && numActiveDone != 0 )
            Deflate
            ( activeShifts, activePreimage, activeX, activeEsts,
              activeConverged, activeItCounts, progress );

        // Save snapshots of the estimates at the requested rate
        psCtrl.snapCtrl.Iterate();
        Snapshot
        ( preimage, estimates, itCounts, numIts, deflate, psCtrl.snapCtrl );
    } 

    invNorms = estimates;
    if( deflate )
        RestoreOrdering( preimage, invNorms, itCounts );

    // Solve one final linear system to attempt to counteract possible
    // cancellation in large entries in inv(U - zI)
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        for( Int iLoc=0; iLoc<nLoc; ++iLoc )
        {
            const Int i = X.GlobalRow(iLoc);
            X.SetLocal
            ( iLoc, jLoc, (i%2==0 ?  Real(i+n-1)/Real(n-1) : 
                                    -Real(i+n-1)/Real(n-1) ) );
        }
    }
    if( psCtrl.schur )
        MultiShiftTrsm( LEFT, UPPER, NORMAL, C(1), U, shifts, X );
    else
    {
        DistMatrix<C,STAR,VR> X_STAR_VR(X);
        MultiShiftHessSolve( UPPER, NORMAL, C(1), U, shifts, X_STAR_VR );
        X = X_STAR_VR;
    }
    std::vector<Real> oneNorms(numLocShifts);
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
        oneNorms[jLoc] = blas::Nrm1( nLoc, X.LockedBuffer(0,jLoc), 1 );
    mpi::AllReduce( oneNorms.data(), numLocShifts, mpi::SUM, g.ColComm() );
    DistMatrix<Real,MR,STAR> invNorms_MR_STAR(g);
    invNorms_MR_STAR.AlignWith( X );
    invNorms_MR_STAR = invNorms;
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        const Real heurNorm = 2*oneNorms[jLoc]/(3*Real(n));
        if( heurNorm > invNorms_MR_STAR.GetLocal(jLoc,0) )
            invNorms_MR_STAR.SetLocal( jLoc, 0, heurNorm );
    }
    invNorms = invNorms_MR_STAR;

    FinalSnapshot( invNorms, itCounts, psCtrl.snapCtrl );
    return itCounts;
}

template<typename Real>
inline DistMatrix<Int,VR,STAR>
HagerHigham
( const DistMatrix<Complex<Real>        >& U, 
  const DistMatrix<Complex<Real>        >& Q,
  const DistMatrix<Complex<Real>,VR,STAR>& shifts, 
        DistMatrix<Real,         VR,STAR>& invNorms, 
  PseudospecCtrl<Real> psCtrl=PseudospecCtrl<Real>() )
{
    DEBUG_ONLY(CallStackEntry cse("pspec::HagerHigham"))
    using namespace pspec;
    typedef Complex<Real> C;
    const Int n = U.Height();
    const Int nLoc = U.LocalHeight();
    const Int numShifts = shifts.Height();
    const Int numLocShifts = shifts.LocalHeight();
    const Grid& g = U.Grid();

    const Int maxIts = psCtrl.maxIts;
    const bool deflate = psCtrl.deflate;
    const bool progress = psCtrl.progress;

    // Keep track of the number of iterations per shift
    DistMatrix<Int,VR,STAR> itCounts(g);
    Ones( itCounts, numShifts, 1 );

    // Keep track of the pivoting history if deflation is requested
    DistMatrix<Int,VR,STAR> preimage(g);
    DistMatrix<C,  VR,STAR> pivShifts( shifts );
    if( deflate )
    {
        preimage.AlignWith( shifts );
        preimage.Resize( numShifts, 1 );
        for( Int iLoc=0; iLoc<numLocShifts; ++iLoc )
        {
            const Int i = preimage.GlobalRow(iLoc);
            preimage.SetLocal( iLoc, 0, i );
        }
    }

    psCtrl.snapCtrl.ResetCounts();

    // The Hessenberg case currently requires explicit access to the adjoint
    DistMatrix<C,VC,STAR> U_VC_STAR(g), UAdj_VC_STAR(g);
    if( !psCtrl.schur )
    {
        U_VC_STAR = U;
        Adjoint( U, UAdj_VC_STAR );
    }

    // Simultaneously run inverse iteration for various shifts
    Timer timer;
    DistMatrix<C> X(g);
    Ones( X, n, numShifts );
    Scale( C(1)/C(n), X );
    Int numIts=0, numDone=0;
    DistMatrix<Real,MR,STAR> estimates(g);
    estimates.AlignWith( shifts );
    Zeros( estimates, numShifts, 1 );
    DistMatrix<Int,VR,STAR> activePreimage(g);
    while( true )
    {
        const Int numActive = ( deflate ? numShifts-numDone : numShifts );
        auto activeShifts = View( pivShifts, 0, 0, numActive, 1 );
        auto activeEsts = View( estimates, 0, 0, numActive, 1 );
        auto activeItCounts = View( itCounts, 0, 0, numActive, 1 );
        auto activeX = View( X, 0, 0, n, numActive );
        if( deflate )
            activePreimage = View( preimage, 0, 0, numActive, 1 );

        if( progress && g.Rank() == 0 )
            timer.Start();

        DistMatrix<C> activeV(g), activeY(g), activeZ(g);
        activeV.AlignWith( activeX );
        activeY.AlignWith( activeX );
        activeZ.AlignWith( activeX );
        if( psCtrl.schur )
        {
            // Solve against Q (U - zI) Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeX, activeV );
            MultiShiftTrsm
            ( LEFT, UPPER, NORMAL, C(1), U, activeShifts, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha )
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against Q (U - zI)^H Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeZ, activeV );
            MultiShiftTrsm
            ( LEFT, UPPER, ADJOINT, C(1), U, activeShifts, activeV );
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeZ );
        }
        else
        {
            // Solve against Q (H - zI) Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeX, activeV );
            DistMatrix<C,STAR,VR> activeV_STAR_VR( activeV );
            MultiShiftHessSolve
            ( UPPER, NORMAL, C(1), U_VC_STAR, activeShifts, activeV_STAR_VR );
            activeV = activeV_STAR_VR;
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeY );

            activeZ = activeY;
            EntrywiseMap
            ( activeZ, []( C alpha ) 
                       { return alpha==C(0) ? C(1) : alpha/Abs(alpha); } );

            // Solve against Q (H - zI)^H Q^H 
            Gemm( ADJOINT, NORMAL, C(1), Q, activeZ, activeV );
            activeV_STAR_VR = activeV;
            DistMatrix<C,VR,STAR> activeShiftsConj(g);
            Conjugate( activeShifts, activeShiftsConj );
            MultiShiftHessSolve
            ( LOWER, NORMAL, C(1), UAdj_VC_STAR, activeShiftsConj, 
              activeV_STAR_VR );
            activeV = activeV_STAR_VR;
            Gemm( NORMAL, NORMAL, C(1), Q, activeV, activeY );
        }

        auto activeConverged =
            OneNormConvergenceTest
            ( activeX, activeY, activeZ, activeEsts, activeItCounts, numIts );
        const Int numActiveDone = ZeroNorm( activeConverged );
        if( deflate )
            numDone += numActiveDone;
        else
            numDone = numActiveDone;
        if( progress && g.Rank() == 0 )
        {
            const double iterTime = timer.Stop();
            std::cout << "iteration " << numIts << ": " << iterTime 
                      << " seconds, " << numDone << " of " << numShifts 
                      << " converged" << std::endl;
        }

        ++numIts;
        if( numIts >= maxIts )
            break;

        if( numDone == numShifts )
            break;
        else if( deflate && numActiveDone != 0 )
            Deflate
            ( activeShifts, activePreimage, activeX, activeEsts,
              activeConverged, activeItCounts, progress );

        // Save snapshots of the estimates at the requested rate
        psCtrl.snapCtrl.Iterate();
        Snapshot
        ( preimage, estimates, itCounts, numIts, deflate, psCtrl.snapCtrl );
    } 
    invNorms = estimates;
    if( deflate )
        RestoreOrdering( preimage, invNorms, itCounts );

    // Solve one final linear system to attempt to counteract possible
    // cancellation in large entries in inv(U - zI)
    if( numShifts == 0 )
        return itCounts;
    auto x = View( X, 0, 0, n, 1 );
    if( x.LocalWidth() == 1 )
    {
        for( Int iLoc=0; iLoc<nLoc; ++iLoc )
        {
            const Int i = x.GlobalRow(iLoc);
            x.SetLocal( i, 0, (i%2==0 ?  Real(i+n-1)/Real(n-1)
                                      : -Real(i+n-1)/Real(n-1) ) );
        }
    }
    DistMatrix<C> yRep(g);
    Gemv( ADJOINT, C(1), Q, x, yRep );
    DistMatrix<C> Y(n,numShifts,g);
    Y.AlignWith( X );
    DistMatrix<C,MC,STAR> yRep_MC_STAR(g);
    yRep_MC_STAR.AlignWith( X );
    yRep_MC_STAR = yRep;
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        const Int j = Y.GlobalCol(jLoc);
        auto y = View( Y, 0, j, n, 1 );
        y = yRep_MC_STAR;
    }
    if( psCtrl.schur )
    {
        MultiShiftTrsm( LEFT, UPPER, NORMAL, C(1), U, shifts, Y );
    }
    else
    {
        DistMatrix<C,STAR,VR> Y_STAR_VR( Y );
        MultiShiftHessSolve( UPPER, NORMAL, C(1), U, shifts, Y_STAR_VR );
        Y = Y_STAR_VR; 
    }
    Gemm( NORMAL, NORMAL, C(1), Q, Y, X );
    std::vector<Real> oneNorms(numLocShifts);
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
        oneNorms[jLoc] = blas::Nrm1( nLoc, X.LockedBuffer(0,jLoc), 1 );
    mpi::AllReduce( oneNorms.data(), numLocShifts, mpi::SUM, g.ColComm() );
    DistMatrix<Real,MR,STAR> invNorms_MR_STAR(g);
    invNorms_MR_STAR.AlignWith( X );
    invNorms_MR_STAR = invNorms;
    for( Int jLoc=0; jLoc<numLocShifts; ++jLoc )
    {
        const Real heurNorm = 2*oneNorms[jLoc]/(3*Real(n));
        if( heurNorm > invNorms_MR_STAR.GetLocal(jLoc,0) )
            invNorms_MR_STAR.SetLocal( jLoc, 0, heurNorm );
    }
    invNorms = invNorms_MR_STAR;

    FinalSnapshot( invNorms, itCounts, psCtrl.snapCtrl );
    return itCounts;
}

} // namespace pspec
} // namespace El

#endif // ifndef EL_PSEUDOSPECTRUM_HAGERHIGHAM_HPP
