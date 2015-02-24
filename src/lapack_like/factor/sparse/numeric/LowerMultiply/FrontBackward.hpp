/*
   Copyright (c) 2009-2012, Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin.
   All rights reserved.

   Copyright (c) 2013, Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright (c) 2013-2014, Jack Poulson and 
   The Georgia Institute of Technology.
   All rights reserved.

   Copyright (c) 2014-2015, Jack Poulson and Stanford University.
   All rights reserved.
   
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_FACTOR_NUMERIC_LOWERMULTIPLY_FRONTBACKWARD_HPP
#define EL_FACTOR_NUMERIC_LOWERMULTIPLY_FRONTBACKWARD_HPP

namespace El {

template<typename F>
inline void FrontVanillaLowerBackwardMultiply
( const Matrix<F>& L, Matrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardMultiply");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal multiply:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    Matrix<F> LT, LB, XT, XB;
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Trmm( LEFT, LOWER, orientation, UNIT, F(1), LT, XT );
    Gemm( orientation, NORMAL, F(1), LB, XB, F(1), XT );
}

template<typename F>
inline void FrontLowerBackwardMultiply
( const SymmFront<F>& front, Matrix<F>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardMultiply"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot multiply against an unfactored matrix");

    if( type == LDL_2D )
        FrontVanillaLowerBackwardMultiply( front.L, W, conjugate );
    else
        LogicError("Unsupported front type");
}

template<typename F>
inline void FrontVanillaLowerBackwardMultiply
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardMultiply");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal multiply:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
      if( L.ColAlign() != X.ColAlign() )
          LogicError("L and X are assumed to be aligned");
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardMultiply
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Trmm( LEFT, LOWER, orientation, UNIT, F(1), LT, XT );

    if( XB.Height() != 0 )
    {
        // Subtract off the parent updates
        DistMatrix<F,STAR,STAR> Z(g);
        LocalGemm( orientation, NORMAL, F(1), LB, XB, Z );
        AxpyContract( F(1), Z, XT );
    }
}

template<typename F>
inline void FrontVanillaLowerBackwardMultiply
( const DistMatrix<F>& L, DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardMultiply");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal multiply:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardMultiply
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Trmm( LEFT, LOWER, orientation, UNIT, F(1), LT, XT );
    Gemm( orientation, NORMAL, F(1), LB, XB, F(1), XT );
}

template<typename F>
inline void FrontLowerBackwardMultiply
( const DistSymmFront<F>& front, DistMatrix<F>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardMultiply"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot multiply against an unfactored matrix");

    if( type == LDL_2D )
        FrontVanillaLowerBackwardMultiply( front.L2D, W, conjugate );
    else
        LogicError("Unsupported front type");
}

template<typename F>
inline void FrontLowerBackwardMultiply
( const DistSymmFront<F>& front, DistMatrix<F,VC,STAR>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardMultiply"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot multiply against an unfactored matrix");

    if( type == LDL_1D )
        FrontVanillaLowerBackwardMultiply( front.L1D, W, conjugate );
    else
        LogicError("Unsupported front type");
}

} // namespace El

#endif // ifndef EL_FACTOR_NUMERIC_LOWERMULTIPLY_FRONTBACKWARD_HPP
