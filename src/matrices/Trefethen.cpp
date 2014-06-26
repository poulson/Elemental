/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

// The symbol of the matrix described at the beginning of Chapter II of 
// Lloyd N. Trefethen and Mark Embree's "Spectra and Pseudospectra" is
//     f(z) = 2 z^{-3} - z^{-2} + 2i z^{-1} - 4 z^2 - 2i z^3.
// For lack of a better name, we will refer to such a Toeplitz matrix as a 
// Trefethen matrix.

template<typename Real> 
void Trefethen( Matrix<Complex<Real>>& A, Int n )
{
    DEBUG_ONLY(CallStackEntry cse("Trefethen"))
    if( n < 4 )
        LogicError("Must be at least 4x4 to have a third-order symbol");
    typedef Complex<Real> C;
    Zeros( A, n, n );
    SetDiagonal( A,  2,       3 );
    SetDiagonal( A, -1,       2 );
    SetDiagonal( A, C(0,2),   1 );
    SetDiagonal( A, -4,      -2 );
    SetDiagonal( A, C(0,-2), -3 );
}

template<typename Real>
void Trefethen( AbstractDistMatrix<Complex<Real>>& A, Int n )
{
    DEBUG_ONLY(CallStackEntry cse("Trefethen"))
    if( n < 4 )
        LogicError("Must be at least 4x4 to have a third-order symbol");
    typedef Complex<Real> C;
    Zeros( A, n, n );
    SetDiagonal( A,  2,       3 );
    SetDiagonal( A, -1,       2 );
    SetDiagonal( A, C(0,2),   1 );
    SetDiagonal( A, -4,      -2 );
    SetDiagonal( A, C(0,-2), -3 );
}

template<typename Real>
void Trefethen( AbstractBlockDistMatrix<Complex<Real>>& A, Int n )
{
    DEBUG_ONLY(CallStackEntry cse("Trefethen"))
    if( n < 4 )
        LogicError("Must be at least 4x4 to have a third-order symbol");
    typedef Complex<Real> C;
    Zeros( A, n, n );
    SetDiagonal( A,  2,       3 );
    SetDiagonal( A, -1,       2 );
    SetDiagonal( A, C(0,2),   1 );
    SetDiagonal( A, -4,      -2 );
    SetDiagonal( A, C(0,-2), -3 );
}

#define PROTO(Real) \
  template void Trefethen( Matrix<Complex<Real>>& A, Int n ); \
  template void Trefethen( AbstractDistMatrix<Complex<Real>>& A, Int n ); \
  template void Trefethen( AbstractBlockDistMatrix<Complex<Real>>& A, Int n );

PROTO(float)
PROTO(double)

} // namespace El
