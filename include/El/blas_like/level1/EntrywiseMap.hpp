/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ENTRYWISEMAP_HPP
#define EL_BLAS_ENTRYWISEMAP_HPP

#include <functional>
#include <iosfwd>

#include "El/core.hpp"
#include "El/core/./DistMatrix/Abstract.hpp"
#include "El/core/./DistMatrix/Block.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/Matrix.hpp"
#include "El/core/environment/decl.hpp"
#include "El/core/imports/mpi.hpp"
#include "El/core/types.hpp"

namespace El {

template <typename T> class DistMultiVec;
template <typename T> class DistSparseMatrix;
template <typename T> class SparseMatrix;

template<typename T>
void EntrywiseMap( Matrix<T>& A, function<T(T)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    const Int m = A.Height();
    const Int n = A.Width();
    T* ABuf = A.Buffer();
    const Int ALDim = A.LDim();
    for( Int j=0; j<n; ++j )
        for( Int i=0; i<m; ++i )
            ABuf[i+j*ALDim] = func(ABuf[i+j*ALDim]);
}

template<typename T>
void EntrywiseMap( SparseMatrix<T>& A, function<T(T)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    T* vBuf = A.ValueBuffer();
    const Int numEntries = A.NumEntries();
    for( Int k=0; k<numEntries; ++k )
        vBuf[k] = func(vBuf[k]);
}

template<typename T>
void EntrywiseMap( AbstractDistMatrix<T>& A, function<T(T)> func )
{ EntrywiseMap( A.Matrix(), func ); }

template<typename T>
void EntrywiseMap( DistSparseMatrix<T>& A, function<T(T)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    T* vBuf = A.ValueBuffer();
    const Int numLocalEntries = A.NumLocalEntries();
    for( Int k=0; k<numLocalEntries; ++k )
        vBuf[k] = func(vBuf[k]);
}

template<typename T>
void EntrywiseMap( DistMultiVec<T>& A, function<T(T)> func )
{ EntrywiseMap( A.Matrix(), func ); }

template<typename S,typename T>
void EntrywiseMap( const Matrix<S>& A, Matrix<T>& B, function<T(S)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    const Int m = A.Height();
    const Int n = A.Width();
    const S* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    B.Resize( m, n );
    T* BBuf = B.Buffer();
    const Int BLDim = B.LDim();
    for( Int j=0; j<n; ++j )
        for( Int i=0; i<m; ++i )
            BBuf[i+j*BLDim] = func(ABuf[i+j*ALDim]);
}

template<typename S,typename T>
void EntrywiseMap
( const SparseMatrix<S>& A, SparseMatrix<T>& B, function<T(S)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    const Int numEntries = A.NumEntries();

    B.graph_ = A.graph_;
    B.vals_.resize( numEntries );
    for( Int k=0; k<numEntries; ++k )
        B.vals_[k] = func(A.vals_[k]);
    B.ProcessQueues();
}

template<typename S,typename T>
void EntrywiseMap
( const ElementalMatrix<S>& A, ElementalMatrix<T>& B, 
  function<T(S)> func )
{ 
    if( A.DistData().colDist == B.DistData().colDist &&
        A.DistData().rowDist == B.DistData().rowDist )
    {
        B.AlignWith( A.DistData() );
        B.Resize( A.Height(), A.Width() );
        EntrywiseMap( A.LockedMatrix(), B.Matrix(), func );
    }
    else
    {
        B.Resize( A.Height(), A.Width() );
        #define GUARD(CDIST,RDIST) \
          B.DistData().colDist == CDIST && B.DistData().rowDist == RDIST
        #define PAYLOAD(CDIST,RDIST) \
          DistMatrix<S,CDIST,RDIST> AProx(B.Grid()); \
          AProx.AlignWith( B.DistData() ); \
          Copy( A, AProx ); \
          EntrywiseMap( AProx.Matrix(), B.Matrix(), func );
        #include "El/macros/GuardAndPayload.h"

        #undef GUARD
        #undef PAYLOAD
    }
}

template<typename S,typename T>
void EntrywiseMap
( const BlockMatrix<S>& A, BlockMatrix<T>& B, 
  function<T(S)> func )
{ 
    if( A.DistData().colDist == B.DistData().colDist &&
        A.DistData().rowDist == B.DistData().rowDist )
    {
        B.AlignWith( A.DistData() );
        B.Resize( A.Height(), A.Width() );
        EntrywiseMap( A.LockedMatrix(), B.Matrix(), func );
    }
    else
    {
        B.Resize( A.Height(), A.Width() );
        #define GUARD(CDIST,RDIST) \
          B.DistData().colDist == CDIST && B.DistData().rowDist == RDIST
        #define PAYLOAD(CDIST,RDIST) \
          DistMatrix<S,CDIST,RDIST,BLOCK> AProx(B.Grid()); \
          AProx.AlignWith( B.DistData() ); \
          Copy( A, AProx ); \
          EntrywiseMap( AProx.Matrix(), B.Matrix(), func );
        #include "El/macros/GuardAndPayload.h"

        #undef GUARD
        #undef PAYLOAD
    }
}

template<typename S,typename T>
void EntrywiseMap
( const DistSparseMatrix<S>& A, DistSparseMatrix<T>& B, 
  function<T(S)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    const Int numEntries = A.vals_.size();
    const Int numRemoteEntries = A.remoteVals_.size();

    B.distGraph_ = A.distGraph_;
    B.vals_.resize( numEntries );
    for( Int k=0; k<numEntries; ++k )
        B.vals_[k] = func(A.vals_[k]);
    B.remoteVals_.resize( numRemoteEntries );
    for( Int k=0; k<numRemoteEntries; ++k )
        B.remoteVals_[k] = func(A.remoteVals_[k]);
    B.multMeta = A.multMeta;
    B.ProcessQueues();
}

template<typename S,typename T>
void EntrywiseMap
( const DistMultiVec<S>& A, DistMultiVec<T>& B, function<T(S)> func )
{
    DEBUG_ONLY(CSE cse("EntrywiseMap"))
    B.SetComm( A.Comm() );
    B.Resize( A.Height(), A.Width() );
    EntrywiseMap( A.LockedMatrix(), B.Matrix(), func );
}

} // namespace El
#endif // ifndef EL_BLAS_ENTRYWISEMAP_HPP
