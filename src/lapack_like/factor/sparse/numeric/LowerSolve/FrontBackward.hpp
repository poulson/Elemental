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
#ifndef EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTBACKWARD_HPP
#define EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTBACKWARD_HPP

#include "./FrontUtil.hpp"

namespace El {

namespace internal {

template<typename F>
void BackwardMany
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool conjugate=false )
{
    // TODO: Replace this with modified inline code?
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    trsm::LLTSmall( orientation, UNIT, L, X );
}

template<typename F>
void BackwardSingle
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool conjugate=false )
{
    const Grid& g = L.Grid();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    DistMatrix<F,STAR,STAR> D(g), L11_STAR_STAR(g), Z1_STAR_STAR(g);
    FormDiagonalBlocks( L, D, conjugate );

    const Int m = L.Height();
    const Int n = L.Width();
    const Int numRHS = X.Width();
    const Int bsize = Blocksize();

    const Int kLast = LastOffset( n, bsize );
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,n-k);
        const Range<Int> ind1(k,k+nb), ind2(k+nb,m);
   
        auto L11Trans_STAR_STAR = D( IR(0,nb), ind1 );
        auto L21 = L( ind2, ind1 ); 
        auto X1 = X( ind1, IR(0,numRHS) );
        auto X2 = X( ind2, IR(0,numRHS) );

        // X1 -= L21' X2
        LocalGemm( orientation, NORMAL, F(-1), L21, X2, Z1_STAR_STAR );
        AddInLocalData( X1, Z1_STAR_STAR );
        El::AllReduce( Z1_STAR_STAR, X1.DistComm() );

        // X1 := L11^-1 X1
        LocalTrsm
        ( LEFT, UPPER, NORMAL, UNIT, F(1), L11Trans_STAR_STAR, Z1_STAR_STAR );
        X1 = Z1_STAR_STAR;
    }
}

} // namespace internal

template<typename F>
inline void FrontVanillaLowerBackwardSolve
( const Matrix<F>& L, Matrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardSolve");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    Matrix<F> LT, LB, XT, XB;
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Gemm( orientation, NORMAL, F(-1), LB, XB, F(1), XT );
    Trsm( LEFT, LOWER, orientation, UNIT, F(1), LT, XT, true );
}

template<typename F>
inline void FrontIntraPivLowerBackwardSolve
( const Matrix<F>& L, const Matrix<Int>& p, Matrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerBackwardSolve"))
    FrontVanillaLowerBackwardSolve( L, X, conjugate );
    Matrix<F> XT, XB;
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontLowerBackwardSolve
( const SymmFront<F>& front, Matrix<F>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardSolve"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot solve against an unfactored matrix");

    if( BlockFactorization(type) )
        FrontBlockLowerBackwardSolve( front.L, W, conjugate );
    else if( PivotedFactorization(type) )
        FrontIntraPivLowerBackwardSolve( front.L, front.piv, W, conjugate );
    else
        FrontVanillaLowerBackwardSolve( front.L, W, conjugate );
}

template<typename F>
inline void FrontVanillaLowerBackwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool conjugate, bool singleL11AllGather=true )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
      if( L.ColAlign() != X.ColAlign() )
          LogicError("L and X are assumed to be aligned");
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardSolve
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    if( XB.Height() != 0 )
    {
        // Subtract off the parent updates
        DistMatrix<F,STAR,STAR> Z(g);
        const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
        LocalGemm( orientation, NORMAL, F(-1), LB, XB, Z );
        AxpyContract( F(1), Z, XT );
    }

    if( singleL11AllGather )
        internal::BackwardSingle( LT, XT, conjugate );
    else
        internal::BackwardMany( LT, XT, conjugate );
}

template<typename F>
inline void FrontIntraPivLowerBackwardSolve
( const DistMatrix<F,VC,STAR>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F,VC,STAR>& X, bool conjugate, bool singleL11AllGather=true )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerBackwardSolve"))

    FrontVanillaLowerBackwardSolve( L, X, conjugate, singleL11AllGather );

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontVanillaLowerBackwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardSolve
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Gemm( orientation, NORMAL, F(-1), LB, XB, F(1), XT );
    Trsm( LEFT, LOWER, orientation, UNIT, F(1), LT, XT );
}

template<typename F>
inline void FrontIntraPivLowerBackwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerBackwardSolve"))

    FrontVanillaLowerBackwardSolve( L, X, conjugate );

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontFastLowerBackwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
      if( L.ColAlign() != X.ColAlign() )
          LogicError("L and X are assumed to be aligned");
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardSolve
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := XT - LB^{T/H} XB
    DistMatrix<F,STAR,STAR> Z(g);
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    if( XB.Height() != 0 )
    {
        LocalGemm( orientation, NORMAL, F(-1), LB, XB, Z );
        AxpyContract( F(1), Z, XT );
    }

    // XT := LT^{T/H} XT
    LocalGemm( orientation, NORMAL, F(1), LT, XT, Z );
    Contract( Z, XT );
}

template<typename F>
inline void FrontFastIntraPivLowerBackwardSolve
( const DistMatrix<F,VC,STAR>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F,VC,STAR>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerBackwardSolve"))

    FrontFastLowerBackwardSolve( L, X, conjugate );

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontFastLowerBackwardSolve
( const DistMatrix<F>& L, DistMatrix<F,VC,STAR>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardSolve
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, snSize );

    DistMatrix<F,MR,STAR> ZT_MR_STAR( g );
    DistMatrix<F,VR,STAR> ZT_VR_STAR( g );
    ZT_MR_STAR.AlignWith( LB );
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    if( XB.Height() != 0 )
    {
        // ZT[MR,* ] := -(LB[MC,MR])^{T/H} XB[MC,* ]
        DistMatrix<F,MC,STAR> XB_MC_STAR( g );
        XB_MC_STAR.AlignWith( LB );
        XB_MC_STAR = XB;
        LocalGemm( orientation, NORMAL, F(-1), LB, XB_MC_STAR, ZT_MR_STAR );

        Contract( ZT_MR_STAR, ZT_VR_STAR );

        // ZT[VC,* ] := ZT[VR,* ]
        DistMatrix<F,VC,STAR> ZT_VC_STAR( g );
        ZT_VC_STAR.AlignWith( XT );
        ZT_VC_STAR = ZT_VR_STAR;

        // XT[VC,* ] += ZT[VC,* ]
        Axpy( F(1), ZT_VC_STAR, XT );
    }

    {
        // ZT[MR,* ] := (LT[MC,MR])^{T/H} XT[MC,* ]
        DistMatrix<F,MC,STAR> XT_MC_STAR( g );
        XT_MC_STAR.AlignWith( LT );
        XT_MC_STAR = XT;
        LocalGemm( orientation, NORMAL, F(1), LT, XT_MC_STAR, ZT_MR_STAR );

        Contract( ZT_MR_STAR, ZT_VR_STAR );

        // XT[VC,* ] := ZT[VR,* ]
        XT = ZT_VR_STAR;
    }
}

template<typename F>
inline void FrontFastIntraPivLowerBackwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F,VC,STAR>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerBackwardSolve"))

    FrontFastLowerBackwardSolve( L, X, conjugate );

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontFastLowerBackwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerBackwardSolve
        ( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := XT - LB^{T/H} XB
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Gemm( orientation, NORMAL, F(-1), LB, XB, F(1), XT );

    // XT := LT^{T/H} XT
    DistMatrix<F> Z(XT.Grid());
    Gemm( orientation, NORMAL, F(1), LT, XT, Z );
    XT = Z;
}

template<typename F>
inline void FrontFastIntraPivLowerBackwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerBackwardSolve"))

    FrontFastLowerBackwardSolve( L, X, conjugate );

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    InversePermuteRows( XT, p );
}

template<typename F>
inline void FrontBlockLowerBackwardSolve
( const Matrix<F>& L, Matrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerBackwardSolve");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    Matrix<F> LT, LB, XT, XB;
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    // YT := LB^[T/H] XB
    Matrix<F> YT;
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    Gemm( orientation, NORMAL, F(1), LB, XB, YT );

    // XT := XT - inv(ATL) YT
    Gemm( NORMAL, NORMAL, F(-1), LT, YT, F(1), XT );
}

template<typename F>
inline void FrontBlockLowerBackwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
      if( L.ColAlign() != X.ColAlign() )
          LogicError("L and X are assumed to be aligned");
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontBlockLowerBackwardSolve( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    if( XB.Height() == 0 )
        return;

    // YT := LB^{T/H} XB
    DistMatrix<F,STAR,STAR> Z( g );
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    LocalGemm( orientation, NORMAL, F(1), LB, XB, Z );
    DistMatrix<F,VC,STAR> YT(g);
    YT.AlignWith( XT );
    Contract( Z, YT );

    // XT := XT - inv(ATL) YT
    LocalGemm( NORMAL, NORMAL, F(1), LT, YT, Z );
    AxpyContract( F(-1), Z, XT );
}

template<typename F>
inline void FrontBlockLowerBackwardSolve
( const DistMatrix<F>& L, DistMatrix<F,VC,STAR>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontBlockLowerBackwardSolve( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, snSize );

    if( XB.Height() == 0 )
        return;

    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    DistMatrix<F,MR,STAR> ZT_MR_STAR( g );
    DistMatrix<F,VR,STAR> ZT_VR_STAR( g );
    ZT_MR_STAR.AlignWith( LB );
    DistMatrix<F,VC,STAR> YT(g);
    YT.AlignWith( XT );
    {
        // ZT[MR,* ] := (LB[MC,MR])^{T/H} XB[MC,* ]
        DistMatrix<F,MC,STAR> XB_MC_STAR( g );
        XB_MC_STAR.AlignWith( LB );
        XB_MC_STAR = XB;
        LocalGemm( orientation, NORMAL, F(1), LB, XB_MC_STAR, ZT_MR_STAR );

        Contract( ZT_MR_STAR, ZT_VR_STAR );

        // YT[VC,* ] := ZT[VR,* ]
        YT = ZT_VR_STAR;
    }

    {
        // ZT[MR,* ] := inv(ATL)[MC,MR] YT[MC,* ]
        DistMatrix<F,MC,STAR> YT_MC_STAR( g );
        YT_MC_STAR.AlignWith( LT );
        YT_MC_STAR = YT;
        LocalGemm( orientation, NORMAL, F(1), LT, YT_MC_STAR, ZT_MR_STAR );

        Contract( ZT_MR_STAR, ZT_VR_STAR );

        // ZT[VC,* ] := ZT[VR,* ]
        DistMatrix<F,VC,STAR> ZT_VC_STAR( g );
        ZT_VC_STAR.AlignWith( XT );
        ZT_VC_STAR = ZT_VR_STAR;

        // XT[VC,* ] -= ZT[VC,* ]
        Axpy( F(-1), ZT_VC_STAR, XT );
    }
}

template<typename F>
inline void FrontBlockLowerBackwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerBackwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontBlockLowerBackwardSolve( L.LockedMatrix(), X.Matrix(), conjugate );
        return;
    }

    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );
    if( XB.Height() == 0 )
        return;

    // YT := LB^{T/H} XB
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    DistMatrix<F> YT( XT.Grid() );
    Gemm( orientation, NORMAL, F(1), LB, XB, YT );

    // XT := XT - inv(ATL) YT
    Gemm( NORMAL, NORMAL, F(-1), LT, YT, F(1), XT );
}

template<typename F>
inline void FrontLowerBackwardSolve
( const DistSymmFront<F>& front, DistMatrix<F>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardSolve"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot solve against an unfactored matrix");
    const bool blocked = BlockFactorization(type);

    if( type == LDL_2D )
        FrontVanillaLowerBackwardSolve( front.L2D, W, conjugate );
    else if( type == LDL_SELINV_2D )
        FrontFastLowerBackwardSolve( front.L2D, W, conjugate );
    else if( type == LDL_INTRAPIV_2D )
        FrontIntraPivLowerBackwardSolve
        ( front.L2D, front.piv, W, conjugate );
    else if( type == LDL_INTRAPIV_SELINV_2D )
        FrontFastIntraPivLowerBackwardSolve
        ( front.L2D, front.piv, W, conjugate );
    else if( blocked )
        FrontBlockLowerBackwardSolve( front.L2D, W, conjugate );
    else
        LogicError("Unsupported front type");
}

template<typename F>
inline void FrontLowerBackwardSolve
( const DistSymmFront<F>& front, DistMatrix<F,VC,STAR>& W, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerBackwardSolve"))
    SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot solve against an unfactored matrix");
    const bool blocked = BlockFactorization(type);

    if( type == LDL_1D )
        FrontVanillaLowerBackwardSolve( front.L1D, W, conjugate );
    else if( type == LDL_SELINV_1D )
        FrontFastLowerBackwardSolve( front.L1D, W, conjugate );
    else if( type == LDL_SELINV_2D )
        FrontFastLowerBackwardSolve( front.L2D, W, conjugate );
    else if( type == LDL_INTRAPIV_1D )
        FrontIntraPivLowerBackwardSolve
        ( front.L1D, front.piv, W, conjugate );
    else if( type == LDL_INTRAPIV_SELINV_1D )
        FrontFastIntraPivLowerBackwardSolve
        ( front.L1D, front.piv, W, conjugate );
    else if( type == LDL_INTRAPIV_SELINV_2D )
        FrontFastIntraPivLowerBackwardSolve
        ( front.L2D, front.piv, W, conjugate );
    else if( blocked )
        FrontBlockLowerBackwardSolve( front.L2D, W, conjugate );
    else
        LogicError("Unsupported front type");
}

} // namespace El

#endif // ifndef EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTBACKWARD_HPP
