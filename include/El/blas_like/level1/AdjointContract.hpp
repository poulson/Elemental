/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_ADJOINTCONTRACT_HPP
#define EL_BLAS_ADJOINTCONTRACT_HPP

#include <iosfwd>

#include "El/core.hpp"
#include "El/core/./DistMatrix/Block.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/environment/decl.hpp"

namespace El {

template<typename T>
void AdjointContract( const ElementalMatrix<T>& A, ElementalMatrix<T>& B )
{
    DEBUG_ONLY(CSE cse("AdjointContract"))
    TransposeContract( A, B, true );
}

template<typename T>
void AdjointContract
( const BlockMatrix<T>& A, 
        BlockMatrix<T>& B )
{
    DEBUG_ONLY(CSE cse("AdjointContract"))
    TransposeContract( A, B, true );
}

} // namespace El

#endif // ifndef EL_BLAS_ADJOINTCONTRACT_HPP
