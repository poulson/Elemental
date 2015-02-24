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
void LowerClip( Matrix<Real>& X, Real lowerBound )
{
    DEBUG_ONLY(
        CallStackEntry cse("LowerClip");
        if( IsComplex<Real>::val )
            LogicError("Lower clip does not apply to complex data");
    )
    auto lowerClip = [&]( Real alpha ) { return Max(lowerBound,alpha); };
    EntrywiseMap( X, function<Real(Real)>(lowerClip) );
}

template<typename Real>
void UpperClip( Matrix<Real>& X, Real upperBound )
{
    DEBUG_ONLY(
        CallStackEntry cse("UpperClip");
        if( IsComplex<Real>::val )
            LogicError("Upper clip does not apply to complex data");
    )
    auto upperClip = [&]( Real alpha ) { return Min(upperBound,alpha); };
    EntrywiseMap( X, function<Real(Real)>(upperClip) );
}

template<typename Real>
void Clip( Matrix<Real>& X, Real lowerBound, Real upperBound )
{
    DEBUG_ONLY(
        CallStackEntry cse("Clip");
        if( IsComplex<Real>::val )
            LogicError("Clip does not apply to complex data");
    )
    auto clip = [&]( Real alpha ) 
                { return Max(lowerBound,Min(upperBound,alpha)); };
    EntrywiseMap( X, function<Real(Real)>(clip) );
}

template<typename Real>
void LowerClip( AbstractDistMatrix<Real>& X, Real lowerBound )
{ LowerClip( X.Matrix(), lowerBound ); }

template<typename Real>
void UpperClip( AbstractDistMatrix<Real>& X, Real upperBound )
{ UpperClip( X.Matrix(), upperBound ); }

template<typename Real>
void Clip( AbstractDistMatrix<Real>& X, Real lowerBound, Real upperBound )
{ Clip( X.Matrix(), lowerBound, upperBound ); }

template<typename Real>
void LowerClip( AbstractBlockDistMatrix<Real>& X, Real lowerBound )
{ LowerClip( X.Matrix(), lowerBound ); }

template<typename Real>
void UpperClip( AbstractBlockDistMatrix<Real>& X, Real upperBound )
{ UpperClip( X.Matrix(), upperBound ); }

template<typename Real>
void Clip( AbstractBlockDistMatrix<Real>& X, Real lowerBound, Real upperBound )
{ Clip( X.Matrix(), lowerBound, upperBound ); }

template<typename Real>
void LowerClip( DistMultiVec<Real>& X, Real lowerBound )
{ LowerClip( X.Matrix(), lowerBound ); }

template<typename Real>
void UpperClip( DistMultiVec<Real>& X, Real upperBound )
{ UpperClip( X.Matrix(), upperBound ); }

template<typename Real>
void Clip( DistMultiVec<Real>& X, Real lowerBound, Real upperBound )
{ Clip( X.Matrix(), lowerBound, upperBound ); }

#define PROTO(Real) \
  template void LowerClip \
  ( Matrix<Real>& X, Real lowerBound ); \
  template void LowerClip \
  ( AbstractDistMatrix<Real>& X, Real lowerBound ); \
  template void LowerClip \
  ( AbstractBlockDistMatrix<Real>& X, Real lowerBound ); \
  template void LowerClip \
  ( DistMultiVec<Real>& X, Real lowerBound ); \
  template void UpperClip \
  ( Matrix<Real>& X, Real upperBound ); \
  template void UpperClip \
  ( AbstractDistMatrix<Real>& X, Real upperBound ); \
  template void UpperClip \
  ( AbstractBlockDistMatrix<Real>& X, Real upperBound ); \
  template void UpperClip \
  ( DistMultiVec<Real>& X, Real upperBound ); \
  template void Clip \
  ( Matrix<Real>& X, Real lowerBound, Real upperBound ); \
  template void Clip \
  ( AbstractDistMatrix<Real>& X, Real lowerBound, Real upperBound ); \
  template void Clip \
  ( AbstractBlockDistMatrix<Real>& X, Real lowerBound, Real upperBound ); \
  template void Clip \
  ( DistMultiVec<Real>& X, Real lowerBound, Real upperBound );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
