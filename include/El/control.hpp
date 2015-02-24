/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_CONTROL_HPP
#define EL_CONTROL_HPP

namespace El {

// Lyapunov
// ========
template<typename F>
void Lyapunov
( const Matrix<F>& A, const Matrix<F>& C, Matrix<F>& X,
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );
template<typename F>
void Lyapunov
( const AbstractDistMatrix<F>& A, const AbstractDistMatrix<F>& C, 
        AbstractDistMatrix<F>& X, SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );

// Ricatti
// =======
template<typename F>
void Ricatti
( Matrix<F>& W, Matrix<F>& X, 
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );
template<typename F>
void Ricatti
( AbstractDistMatrix<F>& W, AbstractDistMatrix<F>& X,
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );

template<typename F>
void Ricatti
( UpperOrLower uplo,
  const Matrix<F>& A, const Matrix<F>& K, const Matrix<F>& L, 
  Matrix<F>& X, SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );
template<typename F>
void Ricatti
( UpperOrLower uplo,
  const AbstractDistMatrix<F>& A, const AbstractDistMatrix<F>& K, 
  const AbstractDistMatrix<F>& L,       AbstractDistMatrix<F>& X, 
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );

// Sylvester
// =========
template<typename F>
void Sylvester
( Int m, Matrix<F>& W, Matrix<F>& X,
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );
template<typename F>
void Sylvester
( Int m, AbstractDistMatrix<F>& W, AbstractDistMatrix<F>& X,
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );

template<typename F>
void Sylvester
( const Matrix<F>& A, const Matrix<F>& B, const Matrix<F>& C, Matrix<F>& X,
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );
template<typename F>
void Sylvester
( const AbstractDistMatrix<F>& A, const AbstractDistMatrix<F>& B, 
  const AbstractDistMatrix<F>& C,       AbstractDistMatrix<F>& X, 
  SignCtrl<Base<F>> ctrl=SignCtrl<Base<F>>() );

} // namespace El

#endif // ifndef EL_CONTROL_HPP
