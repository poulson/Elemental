/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_AXPY_HPP
#define EL_BLAS_AXPY_HPP

#include <iosfwd>
#include <memory>

#include "./Axpy/util.hpp"
#include "El/core.hpp"
#include "El/core/./DistMatrix/Abstract.hpp"
#include "El/core/./DistMatrix/Block.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/Matrix.hpp"
#include "El/core/environment/decl.hpp"
#include "El/core/imports/blas.hpp"
#include "El/core/imports/mpi.hpp"
#include "El/core/types.hpp"

namespace El {

template <typename T> class DistMultiVec;
template <typename T> class DistSparseMatrix;
template <typename T> class SparseMatrix;

template<typename T,typename S>
void Axpy( S alphaS, const Matrix<T>& X, Matrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("Axpy"))
    const T alpha = T(alphaS);
    const Int mX = X.Height();
    const Int nX = X.Width();
    const Int nY = Y.Width();
    const Int ldX = X.LDim();
    const Int ldY = Y.LDim();
    const T* XBuf = X.LockedBuffer();
          T* YBuf = Y.Buffer();
    // If X and Y are vectors, we can allow one to be a column and the other
    // to be a row. Otherwise we force X and Y to be the same dimension.
    if( mX == 1 || nX == 1 )
    {
        const Int XLength = ( nX==1 ? mX : nX );
        const Int XStride = ( nX==1 ? 1  : ldX );
        const Int YStride = ( nY==1 ? 1  : ldY );
        DEBUG_ONLY(
          const Int mY = Y.Height();
          const Int YLength = ( nY==1 ? mY : nY );
          if( XLength != YLength )
              LogicError("Nonconformal Axpy");
        )
        blas::Axpy( XLength, alpha, XBuf, XStride, YBuf, YStride );
    }
    else
    {
        DEBUG_ONLY(
          const Int mY = Y.Height();
          if( mX != mY || nX != nY )
              LogicError("Nonconformal Axpy");
        )
        if( nX <= mX )
            for( Int j=0; j<nX; ++j )
                blas::Axpy( mX, alpha, &XBuf[j*ldX], 1, &YBuf[j*ldY], 1 );
        else
            for( Int i=0; i<mX; ++i )
                blas::Axpy( nX, alpha, &XBuf[i], ldX, &YBuf[i], ldY );
    }
}

template<typename T,typename S>
void Axpy( S alphaS, const SparseMatrix<T>& X, SparseMatrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("Axpy"))
    if( X.Height() != Y.Height() || X.Width() != Y.Width() )
        LogicError("X and Y must have the same dimensions");
    const T alpha = T(alphaS);
    const Int numEntries = X.NumEntries();
    const T* XValBuf = X.LockedValueBuffer();
    const Int* XRowBuf = X.LockedSourceBuffer();
    const Int* XColBuf = X.LockedTargetBuffer();
    if( !Y.FrozenSparsity() )
        Y.Reserve( numEntries );
    for( Int k=0; k<numEntries; ++k ) 
        Y.QueueUpdate( XRowBuf[k], XColBuf[k], alpha*XValBuf[k] );
    Y.ProcessQueues();
}

template<typename T,typename S>
void Axpy( S alphaS, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y )
{
    DEBUG_ONLY(
      CSE cse("Axpy");
      AssertSameGrids( X, Y );
    )
    const T alpha = T(alphaS);

    const ElementalData XDistData = X.DistData();
    const ElementalData YDistData = Y.DistData();

    if( XDistData == YDistData )
    {
        Axpy( alpha, X.LockedMatrix(), Y.Matrix() );
    }
    else
    {
        // TODO: Consider what happens if one is a row vector and the other
        //       is a column vector...
        unique_ptr<ElementalMatrix<T>> XCopy( Y.Construct(Y.Grid(),Y.Root()) );
        XCopy->AlignWith( YDistData );
        Copy( X, *XCopy );
        Axpy( alpha, XCopy->LockedMatrix(), Y.Matrix() );
    }
}

template<typename T,typename S>
void Axpy( S alphaS, const BlockMatrix<T>& X, BlockMatrix<T>& Y )
{
    DEBUG_ONLY(
      CSE cse("Axpy");
      AssertSameGrids( X, Y );
    )
    const T alpha = T(alphaS);

    const DistData XDistData = X.DistData();
    const DistData YDistData = Y.DistData();

    if( XDistData == YDistData )
    {
        Axpy( alpha, X.LockedMatrix(), Y.Matrix() );
    }
    else
    {
        unique_ptr<BlockMatrix<T>>
          XCopy( Y.Construct(Y.Grid(),Y.Root()) );
        XCopy->AlignWith( YDistData );
        Copy( X, *XCopy );
        Axpy( alpha, XCopy->LockedMatrix(), Y.Matrix() );
    }
}

template<typename T,typename S>
void Axpy( S alphaS, const AbstractDistMatrix<T>& X, AbstractDistMatrix<T>& Y )
{
    DEBUG_ONLY(
      CSE cse("Axpy");
      AssertSameGrids( X, Y );
    )
    const T alpha = T(alphaS);

    if( X.Wrap() == ELEMENT && Y.Wrap() == ELEMENT )
    {
        const auto& XCast = static_cast<const ElementalMatrix<T>&>(X);
              auto& YCast = static_cast<      ElementalMatrix<T>&>(Y);
        Axpy( alpha, XCast, YCast );
    }
    else if( X.Wrap() == BLOCK && Y.Wrap() == BLOCK )
    {
        const auto& XCast = static_cast<const BlockMatrix<T>&>(X);
              auto& YCast = static_cast<      BlockMatrix<T>&>(Y);
        Axpy( alpha, XCast, YCast );
    }
    else if( X.Wrap() == ELEMENT )
    {
        const auto& XCast = static_cast<const ElementalMatrix<T>&>(X);
              auto& YCast = static_cast<      BlockMatrix<T>&>(Y);
        unique_ptr<BlockMatrix<T>>
          XCopy( YCast.Construct(Y.Grid(),Y.Root()) );
        XCopy->AlignWith( YCast.DistData() );
        Copy( XCast, *XCopy );
        Axpy( alpha, XCopy->LockedMatrix(), Y.Matrix() );
    }
    else
    {
        const auto& XCast = static_cast<const BlockMatrix<T>&>(X);
              auto& YCast = static_cast<      ElementalMatrix<T>&>(Y);
        unique_ptr<ElementalMatrix<T>>
          XCopy( YCast.Construct(Y.Grid(),Y.Root()) );
        XCopy->AlignWith( YCast.DistData() );
        Copy( XCast, *XCopy );
        Axpy( alpha, XCopy->LockedMatrix(), Y.Matrix() );
    }
}

template<typename T,typename S>
void Axpy( S alphaS, const DistSparseMatrix<T>& X, DistSparseMatrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("Axpy"))
    if( X.Height() != Y.Height() || X.Width() != Y.Width() )
        LogicError("X and Y must have the same dimensions");
    if( X.Comm() != Y.Comm() )
        LogicError("X and Y must have the same communicator");
    const T alpha = T(alphaS);
    const Int numLocalEntries = X.NumLocalEntries();
    const Int firstLocalRow = X.FirstLocalRow();
    const T* XValBuf = X.LockedValueBuffer();
    const Int* XRowBuf = X.LockedSourceBuffer();
    const Int* XColBuf = X.LockedTargetBuffer();
    if( !Y.FrozenSparsity() )
        Y.Reserve( numLocalEntries );
    for( Int k=0; k<numLocalEntries; ++k ) 
        Y.QueueLocalUpdate
        ( XRowBuf[k]-firstLocalRow, XColBuf[k], alpha*XValBuf[k] );
    Y.ProcessLocalQueues();
}

template<typename T,typename S>
void Axpy( S alpha, const DistMultiVec<T>& X, DistMultiVec<T>& Y )
{
    DEBUG_ONLY(
      CSE cse("Axpy");
      if( !mpi::Congruent( X.Comm(), Y.Comm() ) )
          LogicError("X and Y must have congruent communicators");
      if( X.Height() != Y.Height() )
          LogicError("X and Y must be the same height");
      if( X.Width() != Y.Width() )
          LogicError("X and Y must be the same width");
    )
    Axpy( alpha, X.LockedMatrix(), Y.Matrix() );
}

} // namespace El

#endif // ifndef EL_BLAS_AXPY_HPP
