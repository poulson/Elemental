#ifndef EL_BLAS_LIKE_LEVEL3_SYRK_HPP
#define EL_BLAS_LIKE_LEVEL3_SYRK_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#include "./Syrk/LN.hpp"
#include "./Syrk/LT.hpp"
#include "./Syrk/UN.hpp"
#include "./Syrk/UT.hpp"

namespace El {

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const Matrix<T>& A, T beta, Matrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(
      CSE cse("Syrk");
      if( orientation == NORMAL )
      {
          if( A.Height() != C.Height() || A.Height() != C.Width() )
              LogicError("Nonconformal Syrk");
      }
      else
      {
          if( A.Width() != C.Height() || A.Width() != C.Width() )
              LogicError("Nonconformal Syrk");
      }
    )
    const char uploChar = UpperOrLowerToChar( uplo );
    const char transChar = OrientationToChar( orientation );
    const Int k = ( orientation == NORMAL ? A.Width() : A.Height() );
    if( conjugate )
    {
        blas::Herk
        ( uploChar, transChar, C.Height(), k,
          RealPart(alpha), A.LockedBuffer(), A.LDim(),
          RealPart(beta),  C.Buffer(),       C.LDim() );
    }
    else
    {
        blas::Syrk
        ( uploChar, transChar, C.Height(), k,
          alpha, A.LockedBuffer(), A.LDim(),
          beta,  C.Buffer(),       C.LDim() );
    }
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const Matrix<T>& A, Matrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const ElementalMatrix<T>& A, 
  T beta,        ElementalMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))
    ScaleTrapezoid( beta, uplo, C );
    if( uplo == LOWER && orientation == NORMAL )
        syrk::LN( alpha, A, C, conjugate );
    else if( uplo == LOWER )
        syrk::LT( alpha, A, C, conjugate );
    else if( orientation == NORMAL )
        syrk::UN( alpha, A, C, conjugate );
    else
        syrk::UT( alpha, A, C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const ElementalMatrix<T>& A, 
                 ElementalMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const SparseMatrix<T>& A, 
  T beta,        SparseMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))

    if( orientation == NORMAL )
    {
        SparseMatrix<T> B;
        Transpose( A, B, conjugate );
        const Orientation newOrient = ( conjugate ? ADJOINT : TRANSPOSE );
        Syrk( uplo, newOrient, alpha, B, beta, C, conjugate );
        return;
    }

    const Int m = A.Height();
    const Int n = A.Width();
    if( C.Height() != n || C.Width() != n )
        LogicError("C was of the incorrect size");

    ScaleTrapezoid( beta, uplo, C );

    // Compute an upper bound on the required capacity
    // ===============================================
    Int newCapacity = C.NumEntries();
    for( Int k=0; k<m; ++k )
    {
        const Int numConn = A.NumConnections(k);
        newCapacity += numConn*numConn;
    }
    C.Reserve( newCapacity );

    // Queue the updates
    // =================
    for( Int k=0; k<m; ++k )
    {
        const Int offset = A.RowOffset(k);
        const Int numConn = A.NumConnections(k);
        for( Int iConn=0; iConn<numConn; ++iConn )
        {
            const Int i = A.Col(offset+iConn);
            const T A_ki = A.Value(offset+iConn);
            for( Int jConn=0; jConn<numConn; ++jConn )
            {
                const Int j = A.Col(offset+jConn);
                if( (uplo == LOWER && i >= j) || (uplo == UPPER && i <= j) )
                {
                    const T A_kj = A.Value(offset+jConn);
                    if( conjugate )
                        C.QueueUpdate( i, j, T(alpha)*Conj(A_ki)*A_kj ); 
                    else
                        C.QueueUpdate( i, j, T(alpha)*A_ki*A_kj );
                }
            }
        }
    }

    // Force the result to be consistent
    // =================================
    C.ProcessQueues();
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const SparseMatrix<T>& A, 
                 SparseMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))
    const Int m = A.Height();
    const Int n = A.Width();
    if( orientation == NORMAL )
        Zeros( C, m, m );
    else
        Zeros( C, n, n );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const DistSparseMatrix<T>& A, 
  T beta,        DistSparseMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))

    if( orientation == NORMAL )
    {
        DistSparseMatrix<T> B(A.Comm());
        Transpose( A, B, conjugate );
        const Orientation newOrient = ( conjugate ? ADJOINT : TRANSPOSE );
        Syrk( uplo, newOrient, alpha, B, beta, C, conjugate );
        return;
    }

    const Int n = A.Width();
    if( C.Height() != n || C.Width() != n )
        LogicError("C was of the incorrect size");
    if( C.Comm() != A.Comm() )
        LogicError("Communicators of A and C must match");

    ScaleTrapezoid( beta, uplo, C );

    // Count the number of entries that we will send
    // =============================================
    Int sendCount = 0;
    const Int localHeightA = A.LocalHeight();
    for( Int kLoc=0; kLoc<localHeightA; ++kLoc )
    {
        const Int offset = A.RowOffset(kLoc);
        const Int numConn = A.NumConnections(kLoc);
        for( Int iConn=0; iConn<numConn; ++iConn )
        {
            const Int i = A.Col(offset+iConn);
            for( Int jConn=0; jConn<numConn; ++jConn )
            {
                const Int j = A.Col(offset+jConn);
                if( (uplo==LOWER && i>=j) || (uplo==UPPER && i<=j) )
                    ++sendCount;
            }
        }
    }

    // Apply the updates
    // ================= 
    C.Reserve( sendCount, sendCount );
    for( Int kLoc=0; kLoc<localHeightA; ++kLoc )
    {
        const Int offset = A.RowOffset(kLoc);
        const Int numConn = A.NumConnections(kLoc);
        for( Int iConn=0; iConn<numConn; ++iConn ) 
        {
            const Int i = A.Col(offset+iConn);
            const T A_ki = A.Value(offset+iConn);
            for( Int jConn=0; jConn<numConn; ++jConn )
            {
                const Int j = A.Col(offset+jConn);
                if( (uplo==LOWER && i>=j) || (uplo==UPPER && i<=j) )
                {
                    const T A_kj = A.Value(offset+jConn);
                    if( conjugate )
                        C.QueueUpdate( i, j, T(alpha)*Conj(A_ki)*A_kj );
                    else
                        C.QueueUpdate( i, j, T(alpha)*A_ki*A_kj );
                }
            }
        }
    }
    C.ProcessQueues();
}

template<typename T>
void Syrk
( UpperOrLower uplo, Orientation orientation,
  T alpha, const DistSparseMatrix<T>& A, 
                 DistSparseMatrix<T>& C, bool conjugate )
{
    DEBUG_ONLY(CSE cse("Syrk"))
    const Int m = A.Height();
    const Int n = A.Width();
    if( orientation == NORMAL )
        Zeros( C, m, m );
    else
        Zeros( C, n, n );
    Syrk( uplo, orientation, alpha, A, T(0), C, conjugate );
}
#ifdef EL_INSTANTIATE_BLAS_LEVEL3
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif


#define PROTO(T) \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const Matrix<T>& A, T beta, Matrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const Matrix<T>& A, Matrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const ElementalMatrix<T>& A, \
    T beta, ElementalMatrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const ElementalMatrix<T>& A, \
                   ElementalMatrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const SparseMatrix<T>& A, \
    T beta,        SparseMatrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const SparseMatrix<T>& A, \
                   SparseMatrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const DistSparseMatrix<T>& A, \
    T beta,        DistSparseMatrix<T>& C, bool conjugate ); \
  EL_EXTERN template void Syrk \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const DistSparseMatrix<T>& A, \
                   DistSparseMatrix<T>& C, bool conjugate );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

#undef EL_EXTERN
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_SYRK_HPP */