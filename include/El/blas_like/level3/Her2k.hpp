#ifndef EL_BLAS_LIKE_LEVEL3_HER2K_HPP
#define EL_BLAS_LIKE_LEVEL3_HER2K_HPP

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
void Her2k
( UpperOrLower uplo, Orientation orientation,
  T alpha,      const Matrix<T>& A, const Matrix<T>& B, 
  Base<T> beta,       Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Her2k"))
    Syr2k( uplo, orientation, alpha, A, B, T(beta), C, true );
}

template<typename T>
void Her2k
( UpperOrLower uplo, Orientation orientation,
  T alpha, const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Her2k"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syr2k( uplo, orientation, alpha, A, B, T(0), C, true );
}

template<typename T>
void Her2k
( UpperOrLower uplo, Orientation orientation,
  T alpha,      const ElementalMatrix<T>& A, const ElementalMatrix<T>& B,
  Base<T> beta,       ElementalMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Her2k"))
    Syr2k( uplo, orientation, alpha, A, B, T(beta), C, true );
}

template<typename T>
void Her2k
( UpperOrLower uplo, Orientation orientation,
  T alpha, const ElementalMatrix<T>& A, const ElementalMatrix<T>& B,
                 ElementalMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Her2k"))
    const Int n = ( orientation==NORMAL ? A.Height() : A.Width() );
    Zeros( C, n, n );
    Syr2k( uplo, orientation, alpha, A, B, T(0), C, true );
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL3
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif


#define PROTO(T) \
  EL_EXTERN template void Her2k \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha,      const Matrix<T>& A, const Matrix<T>& B, \
    Base<T> beta,       Matrix<T>& C ); \
  EL_EXTERN template void Her2k \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const Matrix<T>& A, const Matrix<T>& B, \
                   Matrix<T>& C ); \
  EL_EXTERN template void Her2k \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha, const ElementalMatrix<T>& A, \
             const ElementalMatrix<T>& B, \
                   ElementalMatrix<T>& C ); \
  EL_EXTERN template void Her2k \
  ( UpperOrLower uplo, Orientation orientation, \
    T alpha,      const ElementalMatrix<T>& A, \
                  const ElementalMatrix<T>& B, \
    Base<T> beta,       ElementalMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

#undef EL_EXTERN
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_HER2K_HPP */