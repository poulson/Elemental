/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_GETDIAGONAL_HPP
#define EL_BLAS_GETDIAGONAL_HPP

#include <functional>
#include <iosfwd>

#include "El/core.hpp"
#include "El/core/./DistMatrix/Element.hpp"
#include "El/core/Element/decl.hpp"
#include "El/core/Element/impl.hpp"
#include "El/core/Matrix.hpp"
#include "El/core/environment/decl.hpp"
#include "El/core/imports/mpi.hpp"
#include "El/core/types.hpp"

namespace El {

template <typename T = double, El::DistNS::Dist U = MC, El::DistNS::Dist V = MR, El::DistWrapNS::DistWrap wrap = ELEMENT> class DistMatrix;

template<typename T>
void GetDiagonal( const Matrix<T>& A, Matrix<T>& d, Int offset )
{
    DEBUG_ONLY(CSE cse("GetDiagonal"))
    function<T(T)> identity( []( T alpha ) { return alpha; } ); 
    GetMappedDiagonal( A, d, identity, offset );
}

template<typename T>
void GetRealPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CSE cse("GetRealPartOfDiagonal"))
    function<Base<T>(T)> realPart
    ( []( T alpha ) { return RealPart(alpha); } ); 
    GetMappedDiagonal( A, d, realPart, offset );
}

template<typename T>
void GetImagPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CSE cse("GetImagPartOfDiagonal"))
    function<Base<T>(T)> imagPart
    ( []( T alpha ) { return ImagPart(alpha); } ); 
    GetMappedDiagonal( A, d, imagPart, offset );
}

template<typename T>
Matrix<T> GetDiagonal( const Matrix<T>& A, Int offset )
{
    Matrix<T> d;
    GetDiagonal( A, d, offset );
    return d;
}

template<typename T>
Matrix<Base<T>> GetRealPartOfDiagonal( const Matrix<T>& A, Int offset )
{
    Matrix<Base<T>> d;
    GetRealPartOfDiagonal( A, d, offset );
    return d;
}

template<typename T>
Matrix<Base<T>> GetImagPartOfDiagonal( const Matrix<T>& A, Int offset )
{
    Matrix<Base<T>> d;
    GetImagPartOfDiagonal( A, d, offset );
    return d;
}

// TODO: SparseMatrix implementation

template<typename T,Dist U,Dist V>
void GetDiagonal
( const DistMatrix<T,U,V>& A, ElementalMatrix<T>& d, Int offset )
{ 
    DEBUG_ONLY(CSE cse("GetDiagonal"))
    function<T(T)> identity( []( T alpha ) { return alpha; } );
    GetMappedDiagonal( A, d, identity, offset );
}

template<typename T,Dist U,Dist V>
void GetRealPartOfDiagonal
( const DistMatrix<T,U,V>& A, ElementalMatrix<Base<T>>& d, Int offset )
{ 
    DEBUG_ONLY(CSE cse("GetRealPartOfDiagonal"))
    function<Base<T>(T)> realPart
    ( []( T alpha ) { return RealPart(alpha); } );
    GetMappedDiagonal( A, d, realPart, offset );
}

template<typename T,Dist U,Dist V>
void GetImagPartOfDiagonal
( const DistMatrix<T,U,V>& A, ElementalMatrix<Base<T>>& d, Int offset )
{ 
    DEBUG_ONLY(CSE cse("GetImagPartOfDiagonal"))
    function<Base<T>(T)> imagPart
    ( []( T alpha ) { return ImagPart(alpha); } );
    GetMappedDiagonal( A, d, imagPart, offset );
}

template<typename T>
void GetDiagonal
( const ElementalMatrix<T>& A, ElementalMatrix<T>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      GetDiagonal( ACast, d, offset );
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
void GetRealPartOfDiagonal
( const ElementalMatrix<T>& A, ElementalMatrix<Base<T>>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      GetRealPartOfDiagonal( ACast, d, offset );
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
void GetImagPartOfDiagonal
( const ElementalMatrix<T>& A, ElementalMatrix<Base<T>>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = static_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      GetImagPartOfDiagonal( ACast, d, offset );
    #include "El/macros/GuardAndPayload.h"
}

template<typename T,Dist U,Dist V>
DistMatrix<T,DiagCol<U,V>(),DiagRow<U,V>()>
GetDiagonal( const DistMatrix<T,U,V>& A, Int offset )
{
    DistMatrix<T,DiagCol<U,V>(),DiagRow<U,V>()> d(A.Grid());
    GetDiagonal( A, d, offset );
    return d;
}

template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()>
GetRealPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset )
{
    DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()> d(A.Grid());
    GetRealPartOfDiagonal( A, d, offset );
    return d;
}

template<typename T,Dist U,Dist V>
DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()>
GetImagPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset )
{
    DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()> d(A.Grid());
    GetImagPartOfDiagonal( A, d, offset );
    return d;
}

// TODO: DistSparseMatrix implementation

} // namespace El

#endif // ifndef EL_BLAS_GETDIAGONAL_HPP
