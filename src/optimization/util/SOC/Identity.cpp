/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {
namespace soc {

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void Identity
(       Matrix<Real>& x,
  const Matrix<Int>& orders,
  const Matrix<Int>& firstInds )
{
    EL_DEBUG_CSE
    const Int height = orders.Height();
    EL_DEBUG_ONLY(
      if( firstInds.Height() != height ||
          firstInds.Width() != 1 || orders.Width() != 1 )
          LogicError("orders and firstInds should vectors of the same height");
    )

    Zeros( x, height, 1 );
    for( Int i=0; i<height; ++i )
        if( i == firstInds(i) )
            x(i) = 1;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void Identity
(       AbstractDistMatrix<Real>& xPre,
  const AbstractDistMatrix<Int>& ordersPre,
  const AbstractDistMatrix<Int>& firstIndsPre )
{
    EL_DEBUG_CSE
    AssertSameGrids( xPre, ordersPre, firstIndsPre );

    ElementalProxyCtrl ctrl;
    ctrl.colConstrain = true;
    ctrl.colAlign = 0;

    DistMatrixWriteProxy<Real,Real,VC,STAR>
      xProx( xPre, ctrl );
    DistMatrixReadProxy<Int,Int,VC,STAR>
      ordersProx( ordersPre, ctrl ),
      firstIndsProx( firstIndsPre, ctrl );
    auto& x = xProx.Get();
    auto& orders = ordersProx.GetLocked();
    auto& firstInds = firstIndsProx.GetLocked();

    const Int height = orders.Height();
    EL_DEBUG_ONLY(
      if( firstInds.Height() != height ||
          firstInds.Width() != 1 || orders.Width() != 1 )
          LogicError("orders and firstInds should vectors of the same height");
    )

    const Int* firstIndBuf = firstInds.LockedBuffer();

    Zeros( x, height, 1 );
    Real* xBuf = x.Buffer();
    const Int localHeight = x.LocalHeight();
    for( Int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const Int i = x.GlobalRow(iLoc);
        if( i == firstIndBuf[iLoc] )
            xBuf[iLoc] = 1;
    }
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void Identity
(       DistMultiVec<Real>& x,
  const DistMultiVec<Int>& orders,
  const DistMultiVec<Int>& firstInds )
{
    EL_DEBUG_CSE

    const Int height = orders.Height();
    EL_DEBUG_ONLY(
      if( firstInds.Height() != height ||
          firstInds.Width() != 1 || orders.Width() != 1 )
          LogicError("orders and firstInds should vectors of the same height");
    )

    const Int* firstIndBuf = firstInds.LockedMatrix().LockedBuffer();

    x.SetComm( orders.Comm() );
    Zeros( x, height, 1 );
    Real* xBuf = x.Matrix().Buffer();
    const Int localHeight = x.LocalHeight();
    for( Int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const Int i = x.GlobalRow(iLoc);
        if( i == firstIndBuf[iLoc] )
            xBuf[iLoc] = 1;
    }
}

#define PROTO(Real) \
  template void Identity \
  (       Matrix<Real>& x, \
    const Matrix<Int>& orders, \
    const Matrix<Int>& firstInds ); \
  template void Identity \
  (       AbstractDistMatrix<Real>& x, \
    const AbstractDistMatrix<Int>& orders, \
    const AbstractDistMatrix<Int>& firstInds ); \
  template void Identity \
  (       DistMultiVec<Real>& x, \
    const DistMultiVec<Int>& orders, \
    const DistMultiVec<Int>& firstInds );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace soc
} // namespace El
