/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_TRR2K_TTNN_HPP
#define EL_TRR2K_TTNN_HPP

#include "./NNTT.hpp"

namespace El {
namespace trr2k {

// E := alpha A' B' + beta C D + gamma E
template<typename T>
void Trr2kTTNN
( UpperOrLower uplo,
  Orientation orientA, Orientation orientB,
  T alpha, const AbstractDistMatrix<T>& A, const AbstractDistMatrix<T>& B,
  T beta,  const AbstractDistMatrix<T>& C, const AbstractDistMatrix<T>& D,
  T gamma,       AbstractDistMatrix<T>& E )
{
    DEBUG_ONLY(CallStackEntry cse("trr2k::Trr2kTTNN"))
    Trr2kNNTT
    ( uplo, orientA, orientB, beta, C, D, alpha, A, B, gamma, E );
}

} // namespace trr2k
} // namespace El

#endif // ifndef EL_TRR2K_TTNN_HPP
