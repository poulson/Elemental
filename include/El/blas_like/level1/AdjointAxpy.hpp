/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ADJOINTAXPY_HPP
#define EL_BLAS_ADJOINTAXPY_HPP

#include <iosfwd>

#include "El/core.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/Matrix.hpp"
#include "El/core/environment/decl.hpp"

namespace El {

template <typename T> class DistSparseMatrix;
template <typename T> class SparseMatrix;

template<typename T,typename S>
void AdjointAxpy( S alphaS, const Matrix<T>& X, Matrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("AdjointAxpy"))
    TransposeAxpy( alphaS, X, Y, true );
}

template<typename T,typename S>
void AdjointAxpy( S alphaS, const SparseMatrix<T>& X, SparseMatrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("AdjointAxpy"))
    TransposeAxpy( alphaS, X, Y, true );
}

template<typename T,typename S>
void AdjointAxpy
( S alphaS, const ElementalMatrix<T>& X, ElementalMatrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("AdjointAxpy"))
    TransposeAxpy( alphaS, X, Y, true );
}

template<typename T,typename S>
void AdjointAxpy
( S alphaS, const DistSparseMatrix<T>& X, DistSparseMatrix<T>& Y )
{
    DEBUG_ONLY(CSE cse("AdjointAxpy"))
    TransposeAxpy( alphaS, X, Y, true );
}

} // namespace El

#endif // ifndef EL_BLAS_ADJOINTAXPY_HPP
