/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

template<typename T>
void GetDiagonal( const Matrix<T>& A, Matrix<T>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GetDiagonal"))
    std::function<T(T)> identity( []( T alpha ) { return alpha; } ); 
    GetMappedDiagonal( A, d, identity, offset );
}

template<typename T>
void GetRealPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GetRealPartOfDiagonal"))
    std::function<Base<T>(T)> realPart
    ( []( T alpha ) { return RealPart(alpha); } ); 
    GetMappedDiagonal( A, d, realPart, offset );
}

template<typename T>
void GetImagPartOfDiagonal
( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset )
{
    DEBUG_ONLY(CallStackEntry cse("GetImagPartOfDiagonal"))
    std::function<Base<T>(T)> imagPart
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
( const DistMatrix<T,U,V>& A, AbstractDistMatrix<T>& d, Int offset )
{ 
    DEBUG_ONLY(CallStackEntry cse("GetDiagonal"))
    std::function<T(T)> identity( []( T alpha ) { return alpha; } );
    GetMappedDiagonal( A, d, identity, offset );
}

template<typename T,Dist U,Dist V>
void GetRealPartOfDiagonal
( const DistMatrix<T,U,V>& A, AbstractDistMatrix<Base<T>>& d, Int offset )
{ 
    DEBUG_ONLY(CallStackEntry cse("GetRealPartOfDiagonal"))
    std::function<Base<T>(T)> realPart
    ( []( T alpha ) { return RealPart(alpha); } );
    GetMappedDiagonal( A, d, realPart, offset );
}

template<typename T,Dist U,Dist V>
void GetImagPartOfDiagonal
( const DistMatrix<T,U,V>& A, AbstractDistMatrix<Base<T>>& d, Int offset )
{ 
    DEBUG_ONLY(CallStackEntry cse("GetImagPartOfDiagonal"))
    std::function<Base<T>(T)> imagPart
    ( []( T alpha ) { return ImagPart(alpha); } );
    GetMappedDiagonal( A, d, imagPart, offset );
}

template<typename T>
void GetDiagonal
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = dynamic_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      GetDiagonal( ACast, d, offset );
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
void GetRealPartOfDiagonal
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = dynamic_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      GetRealPartOfDiagonal( ACast, d, offset );
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
void GetImagPartOfDiagonal
( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& d, Int offset )
{
    // Manual dynamic dispatch
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = dynamic_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
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

#define PROTO_DIST(T,U,V) \
  template void GetDiagonal \
  ( const DistMatrix<T,U,V>& A, AbstractDistMatrix<T>& d, Int offset ); \
  template void GetRealPartOfDiagonal \
  ( const DistMatrix<T,U,V>& A, AbstractDistMatrix<Base<T>>& d, Int offset ); \
  template void GetImagPartOfDiagonal \
  ( const DistMatrix<T,U,V>& A, AbstractDistMatrix<Base<T>>& d, Int offset ); \
  template DistMatrix<T,DiagCol<U,V>(),DiagRow<U,V>()> \
  GetDiagonal( const DistMatrix<T,U,V>& A, Int offset ); \
  template DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()> \
  GetRealPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset ); \
  template DistMatrix<Base<T>,DiagCol<U,V>(),DiagRow<U,V>()> \
  GetImagPartOfDiagonal( const DistMatrix<T,U,V>& A, Int offset );

#define PROTO(T) \
  template void GetDiagonal \
  ( const Matrix<T>& A, Matrix<T>& d, Int offset ); \
  template void GetRealPartOfDiagonal \
  ( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset ); \
  template void GetImagPartOfDiagonal \
  ( const Matrix<T>& A, Matrix<Base<T>>& d, Int offset ); \
  template Matrix<T> GetDiagonal \
  ( const Matrix<T>& A, Int offset ); \
  template Matrix<Base<T>> GetRealPartOfDiagonal \
  ( const Matrix<T>& A, Int offset ); \
  template Matrix<Base<T>> GetImagPartOfDiagonal \
  ( const Matrix<T>& A, Int offset ); \
  template void GetDiagonal \
  ( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& d, Int offset ); \
  template void GetRealPartOfDiagonal \
  ( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& d, \
    Int offset ); \
  template void GetImagPartOfDiagonal \
  ( const AbstractDistMatrix<T>& A, AbstractDistMatrix<Base<T>>& d, \
    Int offset ); \
  PROTO_DIST(T,CIRC,CIRC) \
  PROTO_DIST(T,MC,  MR  ) \
  PROTO_DIST(T,MC,  STAR) \
  PROTO_DIST(T,MD,  STAR) \
  PROTO_DIST(T,MR,  MC  ) \
  PROTO_DIST(T,MR,  STAR) \
  PROTO_DIST(T,STAR,MC  ) \
  PROTO_DIST(T,STAR,MD  ) \
  PROTO_DIST(T,STAR,MR  ) \
  PROTO_DIST(T,STAR,STAR) \
  PROTO_DIST(T,STAR,VC  ) \
  PROTO_DIST(T,STAR,VR  ) \
  PROTO_DIST(T,VC,  STAR) \
  PROTO_DIST(T,VR,  STAR)

#include "El/macros/Instantiate.h"

} // namespace El
