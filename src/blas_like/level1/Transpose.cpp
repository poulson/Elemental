/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

namespace transpose {

template<typename T>
void ColFilter
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void ColFilter
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

template<typename T>
void RowFilter
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void RowFilter
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

template<typename T>
void PartialColFilter
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void PartialColFilter
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

template<typename T>
void PartialRowFilter
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void PartialRowFilter
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

template<typename T>
void ColAllGather
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void ColAllGather
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

template<typename T>
void PartialColAllGather
( const AbstractDistMatrix<T>& A,
        AbstractDistMatrix<T>& B, bool conjugate );
template<typename T>
void PartialColAllGather
( const AbstractBlockDistMatrix<T>& A,
        AbstractBlockDistMatrix<T>& B, bool conjugate );

} // namespace transpose

template<typename T>
void Transpose( const Matrix<T>& A, Matrix<T>& B, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("Transpose"))
    const Int m = A.Height();
    const Int n = A.Width();
    B.Resize( n, m );
    // TODO: Optimize this routine
    const T* ABuf = A.LockedBuffer();
          T* BBuf = B.Buffer();
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    if( conjugate )
    {
        for( Int j=0; j<n; ++j )
            for( Int i=0; i<m; ++i )
                BBuf[j+i*ldB] = Conj(ABuf[i+j*ldA]);
    }
    else
    {
        copy::util::InterleaveMatrix
        ( m, n, 
          ABuf, 1,   ldA,
          BBuf, ldB, 1 );
    }
}

template<typename T>
void Transpose
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("Transpose"))
    const DistData AData = A.DistData();
    const DistData BData = B.DistData();

    // TODO: Add shortcuts for square process grids and ensure that they have
    //       an effect on Cholesky and LDL factorizations

    // NOTE: The following are ordered in terms of increasing cost
    if( AData.colDist == BData.rowDist &&
        AData.rowDist == BData.colDist &&
        ((AData.colAlign==BData.rowAlign) || !B.RowConstrained()) &&
        ((AData.rowAlign==BData.colAlign) || !B.ColConstrained()) )
    {
        B.Align( A.RowAlign(), A.ColAlign() );
        B.Resize( A.Width(), A.Height() );
        Transpose( A.LockedMatrix(), B.Matrix(), conjugate );
    }
    else if( AData.colDist == BData.rowDist &&
             AData.rowDist == Collect(BData.colDist) )
    {
        transpose::ColFilter( A, B, conjugate );
    }
    else if( AData.colDist == Collect(BData.rowDist) &&
             AData.rowDist == BData.colDist )
    {
        transpose::RowFilter( A, B, conjugate );
    }
    else if( AData.colDist == BData.rowDist &&
             AData.rowDist == Partial(BData.colDist) )
    {
        transpose::PartialColFilter( A, B, conjugate );
    }
    else if( AData.colDist == Partial(BData.rowDist) &&
             AData.rowDist == BData.colDist )
    {
        transpose::PartialRowFilter( A, B, conjugate );
    }
    else if( Partial(AData.colDist) == BData.rowDist &&
             AData.rowDist          == BData.colDist )
    {
        transpose::PartialColAllGather( A, B, conjugate );
    }
    else if( Collect(AData.colDist) == BData.rowDist &&
             AData.rowDist          == BData.colDist )
    {
        transpose::ColAllGather( A, B, conjugate );
    }
    else
    {
        std::unique_ptr<AbstractDistMatrix<T>> 
            C( B.ConstructTranspose(A.Grid(),A.Root()) );
        C->AlignRowsWith( BData );
        C->AlignColsWith( BData );
        Copy( A, *C );
        B.Resize( A.Width(), A.Height() );
        Transpose( C->LockedMatrix(), B.Matrix(), conjugate );
    }
}

template<typename T>
void Transpose
( const AbstractBlockDistMatrix<T>& A, AbstractBlockDistMatrix<T>& B, 
  bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("Transpose"))
    const BlockDistData AData = A.DistData();
    const BlockDistData BData = B.DistData();
    if( AData.colDist == BData.rowDist &&
        AData.rowDist == BData.colDist &&
        ((AData.colAlign    == BData.rowAlign && 
          AData.blockHeight == BData.blockWidth &&
          AData.colCut      == BData.rowCut) || !B.RowConstrained()) &&
        ((AData.rowAlign   == BData.colAlign && 
          AData.blockWidth == BData.blockHeight &&
          AData.rowCut     == BData.colCut) || !B.ColConstrained()))
    {
        B.Align
        ( A.BlockWidth(), A.BlockHeight(), 
          A.RowAlign(), A.ColAlign(), A.RowCut(), A.ColCut() );
        B.Resize( A.Width(), A.Height() );
        Transpose( A.LockedMatrix(), B.Matrix(), conjugate );
    }
    else if( AData.colDist == BData.rowDist &&
             AData.rowDist == Collect(BData.colDist) )
    {
        transpose::ColFilter( A, B, conjugate );
    }
    else if( AData.colDist == Collect(BData.rowDist) &&
             AData.rowDist == BData.colDist )
    {
        transpose::RowFilter( A, B, conjugate );
    }
    else if( AData.colDist == BData.rowDist &&
             AData.rowDist == Partial(BData.colDist) )
    {
        transpose::PartialColFilter( A, B, conjugate );
    }
    else if( AData.colDist == Partial(BData.rowDist) &&
             AData.rowDist == BData.colDist )
    {
        transpose::PartialRowFilter( A, B, conjugate );
    }
    else if( Partial(AData.colDist) == BData.rowDist &&
             AData.rowDist          == BData.colDist )
    {
        transpose::PartialColAllGather( A, B, conjugate );
    }
    else if( Collect(AData.colDist) == BData.rowDist &&
             AData.rowDist          == BData.colDist )
    {
        transpose::ColAllGather( A, B, conjugate );
    }
    else
    {
        std::unique_ptr<AbstractBlockDistMatrix<T>> 
            C( B.ConstructTranspose(A.Grid(),A.Root()) );
        C->AlignRowsWith( BData );
        C->AlignColsWith( BData );
        Copy( A, *C );
        B.Resize( A.Width(), A.Height() );
        Transpose( C->LockedMatrix(), B.Matrix(), conjugate );
    }
}

template<typename T>
void Transpose
( const SparseMatrix<T>& A, SparseMatrix<T>& B, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("Transpose"))
    Zeros( B, A.Width(), A.Height() );
    TransposeAxpy( T(1), A, B, conjugate );
}

template<typename T>
void Transpose
( const DistSparseMatrix<T>& A, DistSparseMatrix<T>& B, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("Transpose"))
    B.SetComm( A.Comm() );
    Zeros( B, A.Width(), A.Height() );
    TransposeAxpy( T(1), A, B, conjugate );
}

#define PROTO(T) \
  template void Transpose( const Matrix<T>& A, Matrix<T>& B, bool conjugate ); \
  template void Transpose \
  ( const AbstractDistMatrix<T>& A, \
          AbstractDistMatrix<T>& B, bool conjugate ); \
  template void Transpose \
  ( const AbstractBlockDistMatrix<T>& A, \
          AbstractBlockDistMatrix<T>& B, bool conjugate ); \
  template void Transpose \
  ( const SparseMatrix<T>& A, \
          SparseMatrix<T>& B, bool conjugate ); \
  template void Transpose \
  ( const DistSparseMatrix<T>& A, \
          DistSparseMatrix<T>& B, bool conjugate );

#include "El/macros/Instantiate.h"

} // namespace El
