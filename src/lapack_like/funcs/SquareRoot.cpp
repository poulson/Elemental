/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

// See Eq. 6.3 of Nicholas J. Higham and Awad H. Al-Mohy's "Computing Matrix
// Functions", which is currently available at:
// http://eprints.ma.man.ac.uk/1451/01/covered/MIMS_ep2010_18.pdf
//
// TODO: Determine whether stopping criterion should be different than that of
//       Sign

namespace El {

namespace square_root {

template<typename F>
inline void
NewtonStep
( const Matrix<F>& A, const Matrix<F>& X, Matrix<F>& XNew, Matrix<F>& XTmp )
{
    DEBUG_ONLY(CallStackEntry cse("square_root::NewtonStep"))
    // XNew := inv(X) A
    XTmp = X;
    Matrix<Int> p;
    LU( XTmp, p );
    XNew = A;
    lu::SolveAfter( NORMAL, XTmp, p, XNew );

    // XNew := 1/2 ( X + XNew )
    typedef Base<F> Real;
    Axpy( Real(1)/Real(2), X, XNew );
}

template<typename F>
inline void
NewtonStep
( const DistMatrix<F>& A, const DistMatrix<F>& X, 
  DistMatrix<F>& XNew, DistMatrix<F>& XTmp )
{
    DEBUG_ONLY(CallStackEntry cse("square_root::NewtonStep"))
    // XNew := inv(X) A
    XTmp = X;
    DistMatrix<Int,VC,STAR> p(X.Grid());
    LU( XTmp, p );
    XNew = A;
    lu::SolveAfter( NORMAL, XTmp, p, XNew );

    // XNew := 1/2 ( X + XNew )
    typedef Base<F> Real;
    Axpy( Real(1)/Real(2), X, XNew );
}

template<typename F>
inline int
Newton( Matrix<F>& A, const SquareRootCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("square_root::Newton"))
    typedef Base<F> Real;
    Matrix<F> B(A), C, XTmp;
    Matrix<F> *X=&B, *XNew=&C;

    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = A.Height()*lapack::MachineEpsilon<Real>();

    Int numIts=0;
    while( numIts < ctrl.maxIts )
    {
        // Overwrite XNew with the new iterate
        NewtonStep( A, *X, *XNew, XTmp );

        // Use the difference in the iterates to test for convergence
        Axpy( Real(-1), *XNew, *X );
        const Real oneDiff = OneNorm( *X );
        const Real oneNew = OneNorm( *XNew );

        // Ensure that X holds the current iterate and break if possible
        ++numIts;
        std::swap( X, XNew );
        if( ctrl.progress )
            cout << "after " << numIts << " Newton iter's: "
                 << "oneDiff=" << oneDiff << ", oneNew=" << oneNew
                 << ", oneDiff/oneNew=" << oneDiff/oneNew << ", tol="
                 << tol << endl;
        if( oneDiff/oneNew <= Pow(oneNew,ctrl.power)*tol )
            break;
    }
    if( X != &A )
        A = *X;
    return numIts;
}

template<typename F>
inline int
Newton( AbstractDistMatrix<F>& APre, const SquareRootCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("square_root::Newton"))

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    typedef Base<F> Real;
    const Grid& g = A.Grid();
    DistMatrix<F> B(A), C(g), XTmp(g);
    DistMatrix<F> *X=&B, *XNew=&C;

    Real tol = ctrl.tol;
    if( tol == Real(0) )
        tol = A.Height()*lapack::MachineEpsilon<Real>();

    Int numIts=0;
    while( numIts < ctrl.maxIts )
    {
        // Overwrite XNew with the new iterate
        NewtonStep( A, *X, *XNew, XTmp );

        // Use the difference in the iterates to test for convergence
        Axpy( Real(-1), *XNew, *X );
        const Real oneDiff = OneNorm( *X );
        const Real oneNew = OneNorm( *XNew );

        // Ensure that X holds the current iterate and break if possible
        ++numIts;
        std::swap( X, XNew );
        if( ctrl.progress && g.Rank() == 0 )
            cout << "after " << numIts << " Newton iter's: "
                 << "oneDiff=" << oneDiff << ", oneNew=" << oneNew
                 << ", oneDiff/oneNew=" << oneDiff/oneNew << ", tol="
                 << tol << endl;
        if( oneDiff/oneNew <= Pow(oneNew,ctrl.power)*tol )
            break;
    }
    if( X != &A )
        A = *X;
    return numIts;
}

} // namespace square_root

template<typename F>
void SquareRoot( Matrix<F>& A, const SquareRootCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("SquareRoot"))
    square_root::Newton( A, ctrl );
}

template<typename F>
void SquareRoot( AbstractDistMatrix<F>& A, const SquareRootCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("SquareRoot"))
    square_root::Newton( A, ctrl );
}

// Square-root the eigenvalues of A
// --------------------------------

template<typename F>
void HPSDSquareRoot
( UpperOrLower uplo, Matrix<F>& A, const HermitianEigCtrl<F>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("HPSDSquareRoot"))
    typedef Base<F> Real;

    // Get the EVD of A
    Matrix<Real> w;
    Matrix<F> Z;
    HermitianEigSubset<Real> subset;
    HermitianEig( uplo, A, w, Z, UNSORTED, subset, ctrl );

    // Compute the two-norm of A as the maximum absolute value of the eigvals
    const Real twoNorm = MaxNorm( w );

    // Compute the smallest eigenvalue of A
    Real minEig = twoNorm;
    const Int n = w.Height();
    for( Int i=0; i<n; ++i )
    {
        const Real omega = w.Get(i,0);
        minEig = Min(minEig,omega);
    }

    // Set the tolerance equal to n ||A||_2 eps
    const Real eps = lapack::MachineEpsilon<Real>();
    const Real tolerance = n*twoNorm*eps;

    // Ensure that the minimum eigenvalue is not less than - n ||A||_2 eps
    if( minEig < -tolerance )
        throw NonHPSDMatrixException();

    // Overwrite the eigenvalues with f(w)
    for( Int i=0; i<n; ++i )
    {
        const Real omega = w.Get(i,0);
        if( omega > Real(0) )
            w.Set(i,0,Sqrt(omega));
        else
            w.Set(i,0,0);
    }

    // Form the pseudoinverse
    HermitianFromEVD( uplo, A, w, Z );
}

template<typename F>
void HPSDSquareRoot
( UpperOrLower uplo, AbstractDistMatrix<F>& APre, 
  const HermitianEigCtrl<F>& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("HPSDSquareRoot"))

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    // Get the EVD of A
    typedef Base<F> Real;
    const Grid& g = A.Grid();
    DistMatrix<Real,VR,STAR> w(g);
    DistMatrix<F> Z(g);
    HermitianEigSubset<Real> subset;
    HermitianEig( uplo, A, w, Z, UNSORTED, subset, ctrl );

    // Compute the two-norm of A as the maximum absolute value of the eigvals
    const Real twoNorm = MaxNorm( w );

    // Compute the smallest eigenvalue of A
    Real minLocalEig = twoNorm;
    const Int numLocalEigs = w.LocalHeight();
    for( Int iLoc=0; iLoc<numLocalEigs; ++iLoc )
    {
        const Real omega = w.GetLocal(iLoc,0);
        minLocalEig = Min(minLocalEig,omega);
    }
    const Real minEig = mpi::AllReduce( minLocalEig, mpi::MIN, g.VCComm() );

    // Set the tolerance equal to n ||A||_2 eps
    const Int n = A.Height();
    const Real eps = lapack::MachineEpsilon<Real>();
    const Real tolerance = n*twoNorm*eps;

    // Ensure that the minimum eigenvalue is not less than - n ||A||_2 eps
    if( minEig < -tolerance )
        throw NonHPSDMatrixException();

    // Overwrite the eigenvalues with f(w)
    for( Int iLoc=0; iLoc<numLocalEigs; ++iLoc )
    {
        const Real omega = w.GetLocal(iLoc,0);
        if( omega > Real(0) )
            w.SetLocal(iLoc,0,Sqrt(omega));
        else
            w.SetLocal(iLoc,0,0);
    }

    // Form the pseudoinverse
    HermitianFromEVD( uplo, A, w, Z );
}

#define PROTO(F) \
  template void SquareRoot \
  ( Matrix<F>& A, const SquareRootCtrl<Base<F>> ctrl ); \
  template void SquareRoot \
  ( AbstractDistMatrix<F>& A, const SquareRootCtrl<Base<F>> ctrl ); \
  template void HPSDSquareRoot \
  ( UpperOrLower uplo, Matrix<F>& A, const HermitianEigCtrl<F>& ctrl ); \
  template void HPSDSquareRoot \
  ( UpperOrLower uplo, AbstractDistMatrix<F>& A, \
    const HermitianEigCtrl<F>& ctrl );

#define EL_NO_INT_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
