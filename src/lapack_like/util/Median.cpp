/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

template<typename Real>
ValueInt<Real> Median( const Matrix<Real>& x )
{
    DEBUG_ONLY(CallStackEntry cse("Median"))
    if( IsComplex<Real>::val )
        LogicError("Complex numbers do not have a natural ordering");
    const Int m = x.Height();
    const Int n = x.Width();
    if( m != 1 && n != 1 )
        LogicError("Median is meant for a single vector");

    const Int k = ( n==1 ? m : n );
    const Int stride = ( n==1 ? 1 : x.LDim() );
    const Real* xBuffer = x.LockedBuffer();

    vector<ValueInt<Real>> pairs( k );
    for( Int i=0; i<k; ++i )
    {
        pairs[i].value = xBuffer[i*stride];
        pairs[i].index = i;
    }
    std::sort( pairs.begin(), pairs.end(), ValueInt<Real>::Lesser );

    return pairs[k/2];
}

template<typename Real>
ValueInt<Real> Median( const AbstractDistMatrix<Real>& x )
{
    DEBUG_ONLY(CallStackEntry cse("Median"))
    if( x.ColDist() == STAR && x.RowDist() == STAR )
    {
        return Median( x.LockedMatrix() );
    }
    else
    {
        DistMatrix<Real,STAR,STAR> x_STAR_STAR( x );
        return Median( x_STAR_STAR.LockedMatrix() );
    }
}

#define PROTO(Real) \
  template ValueInt<Real> Median( const Matrix<Real>& x ); \
  template ValueInt<Real> Median( const AbstractDistMatrix<Real>& x );

#define EL_NO_COMPLEX_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
