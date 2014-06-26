/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#include "./Write/Ascii.hpp"
#include "./Write/AsciiMatlab.hpp"
#include "./Write/Binary.hpp"
#include "./Write/BinaryFlat.hpp"
#include "./Write/Image.hpp"
#include "./Write/MatrixMarket.hpp"

namespace El {

template<typename T>
void Write
( const Matrix<T>& A, 
  std::string basename, FileFormat format, std::string title )
{
    DEBUG_ONLY(CallStackEntry cse("Write"))
    switch( format )
    {
    case ASCII:         write::Ascii( A, basename, title );       break;
    case ASCII_MATLAB:  write::AsciiMatlab( A, basename, title ); break;
    case BINARY:        write::Binary( A, basename );             break;
    case BINARY_FLAT:   write::BinaryFlat( A, basename );         break;
    case MATRIX_MARKET: write::MatrixMarket( A, basename );       break;
    case BMP:
    case JPG:
    case JPEG:
    case PNG:
    case PPM:
    case XBM:
    case XPM:
        write::Image( A, basename, format ); break;
    default:
        LogicError("Invalid file format");
    }
}

template<typename T,Dist U,Dist V>
void Write
( const DistMatrix<T,U,V>& A, 
  std::string basename, FileFormat format, std::string title )
{
    DEBUG_ONLY(CallStackEntry cse("Write"))
    if( U == A.UGath && V == A.VGath )
    {
        if( A.CrossRank() == A.Root() && A.RedundantRank() == 0 )
            Write( A.LockedMatrix(), basename, format, title );
    }
    else
    {
        DistMatrix<T,CIRC,CIRC> A_CIRC_CIRC( A );
        if( A_CIRC_CIRC.CrossRank() == A_CIRC_CIRC.Root() )
            Write( A_CIRC_CIRC.LockedMatrix(), basename, format, title );
    }
}

template<typename T,Dist U,Dist V>
void Write
( const BlockDistMatrix<T,U,V>& A, 
  std::string basename, FileFormat format, std::string title )
{
    DEBUG_ONLY(CallStackEntry cse("Write"))
    if( U == A.UGath && V == A.VGath )
    {
        if( A.CrossRank() == A.Root() && A.RedundantRank() == 0 )
            Write( A.LockedMatrix(), basename, format, title );
    }
    else
    {
        BlockDistMatrix<T,CIRC,CIRC> A_CIRC_CIRC( A );
        if( A_CIRC_CIRC.CrossRank() == A_CIRC_CIRC.Root() )
            Write( A_CIRC_CIRC.LockedMatrix(), basename, format, title );
    }
}

#define PROTO_DIST(T,U,V) \
  template void Write \
  ( const DistMatrix<T,U,V>& A, \
    std::string basename, FileFormat format, std::string title ); \
  template void Write \
  ( const BlockDistMatrix<T,U,V>& A, \
    std::string basename, FileFormat format, std::string title );

#define PROTO(T) \
  template void Write \
  ( const Matrix<T>& A, \
    std::string basename, FileFormat format, std::string title ); \
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

PROTO(Int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} // namespace El
