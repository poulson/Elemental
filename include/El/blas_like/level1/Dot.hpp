/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_DOT_HPP
#define EL_BLAS_DOT_HPP

#include <iosfwd>

#include "El/core.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/Matrix.hpp"
#include "El/core/environment/decl.hpp"

namespace El {

template <typename T> class DistMultiVec;

template<typename T> 
T Dot( const Matrix<T>& A, const Matrix<T>& B )
{
    DEBUG_ONLY(CSE cse("Dot"))
    return HilbertSchmidt( A, B );
}

template<typename T>
T Dot( const ElementalMatrix<T>& A, const ElementalMatrix<T>& B )
{
    DEBUG_ONLY(CSE cse("Dot"))
    return HilbertSchmidt( A, B );
}

template<typename T>
T Dot( const DistMultiVec<T>& A, const DistMultiVec<T>& B )
{
    DEBUG_ONLY(CSE cse("Dot"))
    return HilbertSchmidt( A, B );
}

} // namespace El

#endif // ifndef EL_BLAS_DOT_HPP
