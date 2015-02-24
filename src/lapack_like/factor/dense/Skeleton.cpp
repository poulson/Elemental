/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

// NOTE: There are *many* algorithms for (pseudo-)skeleton/CUR decompositions,
//       and, for now, we will simply implement one.

// TODO: More algorithms and more options (e.g., default tolerances).

// TODO: Implement randomized algorithms from Jiawei Chiu and Laurent Demanet's 
//       "Sublinear randomized algorithms for skeleton decompositions"?

namespace El {

template<typename F> 
void Skeleton
( const Matrix<F>& A, 
  Matrix<Int>& permR, Matrix<Int>& permC, 
  Matrix<F>& Z, const QRCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("Skeleton"))
    // Find the row permutation
    Matrix<F> B;
    Adjoint( A, B );
    Matrix<F> t;
    Matrix<Base<F>> d;
    QR( B, t, d, permR, ctrl );
    const Int numSteps = t.Height();

    // Form pinv(AR')=pinv(AR)'
    Adjoint( A, B );
    InversePermuteCols( B, permR );
    B.Resize( B.Height(), numSteps );
    Pseudoinverse( B );

    // Form K := A pinv(AR)
    Matrix<F> K;
    Gemm( NORMAL, ADJOINT, F(1), A, B, K );

    // Find the column permutation (force the same number of steps)
    B = A;
    auto secondCtrl = ctrl; 
    secondCtrl.adaptive = false;
    secondCtrl.boundRank = true;
    secondCtrl.maxRank = numSteps;
    QR( B, t, d, permC, secondCtrl );

    // Form pinv(AC)
    B = A;
    InversePermuteCols( B, permC );
    B.Resize( B.Height(), numSteps );
    Pseudoinverse( B );

    // Form Z := pinv(AC) K = pinv(AC) (A pinv(AR))
    Gemm( NORMAL, NORMAL, F(1), B, K, Z );
}

template<typename F> 
void Skeleton
( const AbstractDistMatrix<F>& APre, 
  AbstractDistMatrix<Int>& permR, AbstractDistMatrix<Int>& permC, 
  AbstractDistMatrix<F>& Z, const QRCtrl<Base<F>> ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("Skeleton"))

    auto APtr = ReadProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    const Grid& g = A.Grid();

    // Find the row permutation
    DistMatrix<F> B(g);
    Adjoint( A, B );
    DistMatrix<F,MD,STAR> t(g);
    DistMatrix<Base<F>,MD,STAR> d(g);
    QR( B, t, d, permR, ctrl );
    const Int numSteps = t.Height();

    // Form pinv(AR')=pinv(AR)'
    Adjoint( A, B );
    InversePermuteCols( B, permR );
    B.Resize( B.Height(), numSteps );
    Pseudoinverse( B );

    // Form K := A pinv(AR)
    DistMatrix<F> K(g);
    Gemm( NORMAL, ADJOINT, F(1), A, B, K );

    // Find the column permutation (force the same number of steps)
    B = A;
    auto secondCtrl = ctrl; 
    secondCtrl.adaptive = false;
    secondCtrl.boundRank = true;
    secondCtrl.maxRank = numSteps;
    QR( B, t, d, permC, secondCtrl );

    // Form pinv(AC)
    B = A;
    InversePermuteCols( B, permC );
    B.Resize( B.Height(), numSteps );
    Pseudoinverse( B );

    // Form Z := pinv(AC) K = pinv(AC) (A pinv(AR))
    Gemm( NORMAL, NORMAL, F(1), B, K, Z );
}

#define PROTO(F) \
  template void Skeleton \
  ( const Matrix<F>& A, \
    Matrix<Int>& permR, Matrix<Int>& permC, \
    Matrix<F>& Z, const QRCtrl<Base<F>> ctrl ); \
  template void Skeleton \
  ( const AbstractDistMatrix<F>& A, \
    AbstractDistMatrix<Int>& permR, AbstractDistMatrix<Int>& permC, \
    AbstractDistMatrix<F>& Z, const QRCtrl<Base<F>> ctrl );

#define EL_NO_INT_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
