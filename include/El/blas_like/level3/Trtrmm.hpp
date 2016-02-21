#ifndef EL_BLAS_LIKE_LEVEL3_TRTRMM_HPP
#define EL_BLAS_LIKE_LEVEL3_TRTRMM_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#include "./Trtrmm/Unblocked.hpp"
#include "./Trtrmm/LVar1.hpp"
#include "./Trtrmm/UVar1.hpp"

namespace El {

template<typename T>
void Trtrmm( UpperOrLower uplo, Matrix<T>& A, bool conjugate )
{
    DEBUG_ONLY(
      CSE cse("Trtrmm");
      if( A.Height() != A.Width() )
          LogicError("A must be square");
    )
    if( uplo == LOWER )
        trtrmm::LVar1( A, conjugate );
    else
        trtrmm::UVar1( A, conjugate );
}

template<typename T>
void Trtrmm( UpperOrLower uplo, ElementalMatrix<T>& A, bool conjugate )
{
    DEBUG_ONLY(
        CSE cse("Trtrmm");
        if( A.Height() != A.Width() )
            LogicError("A must be square");
    )
    if( uplo == LOWER )
        trtrmm::LVar1( A, conjugate );
    else
        trtrmm::UVar1( A, conjugate );
}

template<typename T>
void Trtrmm( UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A, bool conjugate )
{ Trtrmm( uplo, A.Matrix(), conjugate ); }
#ifdef EL_INSTANTIATE_BLAS_LEVEL3
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif


#define PROTO(T) \
  EL_EXTERN template void Trtrmm( UpperOrLower uplo, Matrix<T>& A, bool conjugate ); \
  EL_EXTERN template void Trtrmm \
  ( UpperOrLower uplo, ElementalMatrix<T>& A, bool conjugate ); \
  EL_EXTERN template void Trtrmm \
  ( UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A, bool conjugate );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

#undef EL_EXTERN
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_TRTRMM_HPP */