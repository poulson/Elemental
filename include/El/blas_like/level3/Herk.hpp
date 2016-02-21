#ifndef EL_BLAS_LIKE_LEVEL3_HERK_HPP
#define EL_BLAS_LIKE_LEVEL3_HERK_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const Matrix<T>& A, Base<T> beta, Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, T(beta), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const Matrix<T>& A, Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syrk( uplo, orientation, T(alpha), A, T(0), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const ElementalMatrix<T>& A, 
  Base<T> beta,        ElementalMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, T(beta), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const ElementalMatrix<T>& A, ElementalMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syrk( uplo, orientation, T(alpha), A, T(0), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const SparseMatrix<T>& A,
  Base<T> beta,        SparseMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, T(beta), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const SparseMatrix<T>& A,
                       SparseMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const DistSparseMatrix<T>& A,
  Base<T> beta,        DistSparseMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, T(beta), C, true );
}

template<typename T>
void Herk
( UpperOrLower uplo, Orientation orientation,
  Base<T> alpha, const DistSparseMatrix<T>& A,
                       DistSparseMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Herk"))
    Syrk( uplo, orientation, T(alpha), A, C, true );
}
#ifdef EL_INSTANTIATE_BLAS_LEVEL3
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif


#define PROTO(T) \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const Matrix<T>& A, \
    Base<T> beta,        Matrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const Matrix<T>& A, Matrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const ElementalMatrix<T>& A, ElementalMatrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const ElementalMatrix<T>& A, \
    Base<T> beta,        ElementalMatrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const SparseMatrix<T>& A, \
    Base<T> beta,        SparseMatrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const SparseMatrix<T>& A, \
                         SparseMatrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const DistSparseMatrix<T>& A, \
    Base<T> beta,        DistSparseMatrix<T>& C ); \
  EL_EXTERN template void Herk \
  ( UpperOrLower uplo, Orientation orientation, \
    Base<T> alpha, const DistSparseMatrix<T>& A, \
                         DistSparseMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

#undef EL_EXTERN
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_HERK_HPP */
