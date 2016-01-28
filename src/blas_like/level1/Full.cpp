/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El/core/Matrix.hpp>
#include <El/core/SparseMatrix.hpp>
#include <El/blas_like/level1/Copy.hpp>
#include <El/blas_like/level1.hpp>

namespace El {

template<typename T>
Matrix<T> Full( const SparseMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("Full"))
    Matrix<T> B;
    Copy( A, B );
    return B;
}

// NOTE: A DistSparseMatrix version does not exist since it is not yet clear
//       whether Elemental can currently handle creating a grid in such a 
//       routine without a memory leak


#define PROTO(T) \
  template Matrix<T> Full( const SparseMatrix<T>& A );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace El
