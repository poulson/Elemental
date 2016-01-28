/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El/blas_like/level2/Syr2.hpp>

namespace El {

#define PROTO(T) \
  template void Syr2 \
  ( UpperOrLower uplo, T alpha, \
    const Matrix<T>& x, const Matrix<T>& y, Matrix<T>& A, bool conjugate ); \
  template void Syr2 \
  ( UpperOrLower uplo, T alpha, \
    const ElementalMatrix<T>& x, const ElementalMatrix<T>& y, \
    ElementalMatrix<T>& A, bool conjugate );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace El
