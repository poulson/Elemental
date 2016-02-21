#ifndef EL_BLAS_LIKE_LEVEL3_MULTISHIFTTRSM_HPP
#define EL_BLAS_LIKE_LEVEL3_MULTISHIFTTRSM_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#include "./MultiShiftTrsm/LUN.hpp"
#include "./MultiShiftTrsm/LUT.hpp"

namespace El {

template<typename F>
void MultiShiftTrsm
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  F alpha, Matrix<F>& U, const Matrix<F>& shifts, Matrix<F>& X )
{
    DEBUG_ONLY(CSE cse("MultiShiftTrsm"))
    X *= alpha;
    if( side == LEFT && uplo == UPPER )
    {
        if( orientation == NORMAL )
            mstrsm::LUN( U, shifts, X );
        else
            mstrsm::LUT( orientation, U, shifts, X );
    }
    else
        LogicError("This option is not yet supported");
}

template<typename F>
void MultiShiftTrsm
( LeftOrRight side, UpperOrLower uplo, Orientation orientation,
  F alpha, const ElementalMatrix<F>& U, const ElementalMatrix<F>& shifts, 
  ElementalMatrix<F>& X )
{
    DEBUG_ONLY(CSE cse("MultiShiftTrsm"))
    X *= alpha;
    if( side == LEFT && uplo == UPPER )
    {
        if( orientation == NORMAL )
            mstrsm::LUN( U, shifts, X );
        else
            mstrsm::LUT( orientation, U, shifts, X );
    }
    else
        LogicError("This option is not yet supported");
}
#ifdef EL_INSTANTIATE_BLAS_LEVEL3
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif


#define PROTO(F) \
  EL_EXTERN template void MultiShiftTrsm \
  ( LeftOrRight side, UpperOrLower uplo, Orientation orientation, \
    F alpha, Matrix<F>& U, const Matrix<F>& shifts, Matrix<F>& X ); \
  EL_EXTERN template void MultiShiftTrsm \
  ( LeftOrRight side, UpperOrLower uplo, Orientation orientation, \
    F alpha, const ElementalMatrix<F>& U, \
    const ElementalMatrix<F>& shifts, ElementalMatrix<F>& X );

#define EL_NO_INT_PROTO
#define EL_ENABLE_QUAD
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

#undef EL_EXTERN
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_MULTISHIFTTRSM_HPP */