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
#ifndef EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTFORWARD_HPP
#define EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTFORWARD_HPP

#include "./FrontUtil.hpp"

namespace El {

template<typename F>
inline void FrontVanillaLowerForwardSolve( const Matrix<F>& L, Matrix<F>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerForwardSolve");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    Matrix<F> LT, LB, XT, XB;
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    Trsm( LEFT, LOWER, NORMAL, UNIT, F(1), LT, XT );
    Gemm( NORMAL, NORMAL, F(-1), LB, XT, F(1), XB );
}

template<typename F>
inline void FrontIntraPivLowerForwardSolve
( const Matrix<F>& L, const Matrix<Int>& p, Matrix<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerForwardSolve"))
    Matrix<F> XT, XB;
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );
    FrontVanillaLowerForwardSolve( L, X );
}

template<typename F>
inline void FrontBlockLowerForwardSolve( const Matrix<F>& L, Matrix<F>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerForwardSolve");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
    )
    Matrix<F> LT, LB, XT, XB;
    LockedPartitionDown( L, LT, LB, L.Width() );
    PartitionDown( X, XT, XB, L.Width() );

    // XT := inv(ATL) XT
    Matrix<F> YT( XT );
    Gemm( NORMAL, NORMAL, F(1), LT, YT, F(0), XT );

    // XB := XB - LB XT
    Gemm( NORMAL, NORMAL, F(-1), LB, XT, F(1), XB );
}

template<typename F>
inline void 
FrontLowerForwardSolve( const SymmFront<F>& front, Matrix<F>& W )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerForwardSolve"))
    const SymmFrontType type = front.type;
    if( Unfactored(type) )
        LogicError("Cannot solve against an unfactored front");

    if( BlockFactorization(type) )
        FrontBlockLowerForwardSolve( front.L, W );
    else if( PivotedFactorization(type) )
        FrontIntraPivLowerForwardSolve( front.L, front.piv, W );
    else
        FrontVanillaLowerForwardSolve( front.L, W );
}

namespace internal {

template<typename F>
inline void ForwardMany
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X )
{
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    const Int m = L.Height();
    const Int n = L.Width();
    const Int numRHS = X.Width();
    const Int bsize = Blocksize();

    DistMatrix<F,STAR,STAR> L11_STAR_STAR(g), X1_STAR_STAR(g);

    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(bsize,n-k);
        const Range<Int> ind1(k,k+nb), ind2(k+nb,m);
         
        auto L11 = L( ind1, ind1 );
        auto L21 = L( ind2, ind1 );
        auto X1 = X( ind1, IR(0,numRHS) ); 
        auto X2 = X( ind2, IR(0,numRHS) );

        L11_STAR_STAR = L11;
        X1_STAR_STAR = X1;

        // X1[* ,* ] := (L11[* ,* ])^-1 X1[* ,* ]
        LocalTrsm
        ( LEFT, LOWER, NORMAL, UNIT, F(1), L11_STAR_STAR, X1_STAR_STAR );
        X1 = X1_STAR_STAR;

        // X2[VC,* ] -= L21[VC,* ] X1[* ,* ]
        LocalGemm( NORMAL, NORMAL, F(-1), L21, X1_STAR_STAR, F(1), X2 );
    }
}

template<typename F>
void ForwardSingle( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X )
{
    const Grid& g = L.Grid();
    if( g.Size() == 1 )
    {
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    const Int m = L.Height();
    const Int n = L.Width();
    const Int numRHS = X.Width();
    const Int bsize = Blocksize();

    DistMatrix<F,STAR,STAR> X1_STAR_STAR(g), D(g);
    FormDiagonalBlocks( L, D, false );

    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(bsize,n-k);
        const Range<Int> ind1(k,k+nb), ind2(k+nb,m);

        auto L11Trans_STAR_STAR = D( IR(0,nb), ind1 );
        auto L21 = L( ind2, ind1 );
        auto X1 = X( ind1, IR(0,numRHS) );
        auto X2 = X( ind2, IR(0,numRHS) );

        AccumulateRHS( X1, X1_STAR_STAR ); 

        // X1[* ,* ] := (L11[* ,* ])^-1 X1[* ,* ]
        LocalTrsm
        ( LEFT, UPPER, TRANSPOSE, UNIT, 
          F(1), L11Trans_STAR_STAR, X1_STAR_STAR );
        X1 = X1_STAR_STAR;

        // X2[VC,* ] -= L21[VC,* ] X1[* ,* ]
        LocalGemm( NORMAL, NORMAL, F(-1), L21, X1_STAR_STAR, F(1), X2 );
    }
}

} // namespace internal

template<typename F>
inline void FrontVanillaLowerForwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool singleL11AllGather=true )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerForwardSolve");
      if( L.Grid() != X.Grid() )
          LogicError("L and X must be distributed over the same grid");
      if( L.Height() < L.Width() || L.Height() != X.Height() )
          LogicError
          ("Nonconformal solve:\n",
           DimsString(L,"L"),"\n",DimsString(X,"X"));
      if( L.ColAlign() != X.ColAlign() )
          LogicError("L and X are assumed to be aligned");
    )
    if( singleL11AllGather )
        internal::ForwardSingle( L, X );
    else
        internal::ForwardMany( L, X );
}

template<typename F>
inline void FrontIntraPivLowerForwardSolve
( const DistMatrix<F,VC,STAR>& L, const DistMatrix<Int,VC,STAR>& p, 
  DistMatrix<F,VC,STAR>& X, bool singleL11AllGather=true )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerForwardSolve"))

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );

    FrontVanillaLowerForwardSolve( L, X, singleL11AllGather );
}

template<typename F>
inline void FrontVanillaLowerForwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontVanillaLowerForwardSolve");
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
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const Int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := inv(LT) XT
    // TODO: Replace with TrsmLLNMedium?
    Trsm( LEFT, LOWER, NORMAL, UNIT, F(1), LT, XT );

    // XB := XB - LB XT
    Gemm( NORMAL, NORMAL, F(-1), LB, XT, F(1), XB );
}

template<typename F>
inline void FrontIntraPivLowerForwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p, DistMatrix<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("FrontIntraPivLowerForwardSolve"))

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );

    FrontVanillaLowerForwardSolve( L, X );
}

template<typename F>
inline void FrontFastLowerForwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerForwardSolve");
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
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := LT XT
    DistMatrix<F,STAR,STAR> XT_STAR_STAR( XT );
    LocalGemm( NORMAL, NORMAL, F(1), LT, XT_STAR_STAR, F(0), XT );

    // XB := XB - LB XT
    if( LB.Height() != 0 )
    {
        XT_STAR_STAR = XT;
        LocalGemm( NORMAL, NORMAL, F(-1), LB, XT_STAR_STAR, F(1), XB );
    }
}

template<typename F>
inline void FrontFastIntraPivLowerForwardSolve
( const DistMatrix<F,VC,STAR>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerForwardSolve"))

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    if( mpi::WorldRank() == 0 )
        cerr << "Performance warning: pivots not yet cached" << endl;
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );

    FrontFastLowerForwardSolve( L, X );
}

template<typename F>
inline void FrontFastLowerForwardSolve
( const DistMatrix<F>& L, DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerForwardSolve");
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
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, snSize );

    // Get ready for the local multiply
    DistMatrix<F,MR,STAR> XT_MR_STAR(g);
    XT_MR_STAR.AlignWith( LT );

    {
        // ZT[MC,* ] := LT[MC,MR] XT[MR,* ], 
        DistMatrix<F,MC,STAR> ZT_MC_STAR(g);
        ZT_MC_STAR.AlignWith( LT );
        XT_MR_STAR = XT;
        LocalGemm( NORMAL, NORMAL, F(1), LT, XT_MR_STAR, ZT_MC_STAR );

        Contract( ZT_MC_STAR, XT );
    }

    if( LB.Height() != 0 )
    {
        // Set up for the local multiply
        XT_MR_STAR = XT;

        // ZB[MC,* ] := LB[MC,MR] XT[MR,* ]
        DistMatrix<F,MC,STAR> ZB_MC_STAR(g);
        ZB_MC_STAR.AlignWith( LB );
        LocalGemm( NORMAL, NORMAL, F(-1), LB, XT_MR_STAR, ZB_MC_STAR );

        // XB[VC,* ] += ZB[MC,* ]
        AxpyContract( F(1), ZB_MC_STAR, XB );
    }
}

template<typename F>
inline void FrontFastIntraPivLowerForwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p,
  DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerForwardSolve"))

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );

    FrontFastLowerForwardSolve( L, X );
}

template<typename F>
inline void FrontFastLowerForwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontFastLowerForwardSolve");
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
        FrontVanillaLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := LT XT
    DistMatrix<F> YT( XT );
    Gemm( NORMAL, NORMAL, F(1), LT, YT, F(0), XT );

    // XB := XB - LB XT
    Gemm( NORMAL, NORMAL, F(-1), LB, XT, F(1), XB );
}

template<typename F>
inline void FrontFastIntraPivLowerForwardSolve
( const DistMatrix<F>& L, const DistMatrix<Int,VC,STAR>& p, DistMatrix<F>& X )
{
    DEBUG_ONLY(CallStackEntry cse("FrontFastIntraPivLowerForwardSolve"))

    // TODO: Cache the send and recv data for the pivots to avoid p[*,*]
    const Grid& g = L.Grid();
    DistMatrix<F> XT(g), XB(g);
    PartitionDown( X, XT, XB, L.Width() );
    PermuteRows( XT, p );

    FrontFastLowerForwardSolve( L, X );
}

template<typename F>
inline void FrontBlockLowerForwardSolve
( const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerForwardSolve");
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
        FrontBlockLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F,VC,STAR> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := inv(ATL) XT
    DistMatrix<F,STAR,STAR> XT_STAR_STAR( XT );
    LocalGemm( NORMAL, NORMAL, F(1), LT, XT_STAR_STAR, F(0), XT );

    // XB := XB - LB XT
    if( LB.Height() != 0 )
    {
        XT_STAR_STAR = XT;
        LocalGemm( NORMAL, NORMAL, F(-1), LB, XT_STAR_STAR, F(1), XB );
    }
}

template<typename F>
inline void FrontBlockLowerForwardSolve
( const DistMatrix<F>& L, DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerForwardSolve");
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
        FrontBlockLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    DistMatrix<F,VC,STAR> XT(g), XB(g);
    PartitionDown( X, XT, XB, snSize );

    // Get ready for the local multiply
    DistMatrix<F,MR,STAR> XT_MR_STAR(g);
    XT_MR_STAR.AlignWith( LT );

    {
        // ZT[MC,* ] := inv(ATL)[MC,MR] XT[MR,* ], 
        XT_MR_STAR = XT;
        DistMatrix<F,MC,STAR> ZT_MC_STAR(g);
        ZT_MC_STAR.AlignWith( LT );
        LocalGemm( NORMAL, NORMAL, F(1), LT, XT_MR_STAR, ZT_MC_STAR );

        Contract( ZT_MC_STAR, XT );
    }

    if( LB.Height() != 0 )
    {
        // ZB[MC,* ] := LB[MC,MR] XT[MR,* ]
        XT_MR_STAR = XT;
        DistMatrix<F,MC,STAR> ZB_MC_STAR(g);
        ZB_MC_STAR.AlignWith( LB );
        LocalGemm( NORMAL, NORMAL, F(1), LB, XT_MR_STAR, ZB_MC_STAR );

        // XB[VC,* ] -= ZB[MC,* ] = LB[MC,MR] XT[MR,* ]
        AxpyContract( F(-1), ZB_MC_STAR, XB );
    }
}

template<typename F>
inline void FrontBlockLowerForwardSolve
( const DistMatrix<F>& L, DistMatrix<F>& X )
{
    DEBUG_ONLY(
      CallStackEntry cse("FrontBlockLowerForwardSolve");
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
        FrontBlockLowerForwardSolve( L.LockedMatrix(), X.Matrix() );
        return;
    }

    // Separate the top and bottom portions of X and L
    const int snSize = L.Width();
    DistMatrix<F> LT(g), LB(g), XT(g), XB(g);
    LockedPartitionDown( L, LT, LB, snSize );
    PartitionDown( X, XT, XB, snSize );

    // XT := inv(ATL) XT
    DistMatrix<F> Z( XT );
    Gemm( NORMAL, NORMAL, F(1), LT, Z, XT );

    // XB := XB - LB XT
    Gemm( NORMAL, NORMAL, F(-1), LB, XT, F(1), XB );
}

template<typename F>
inline void 
FrontLowerForwardSolve
( const DistSymmFront<F>& front, DistMatrix<F,VC,STAR>& W )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerForwardSolve"))
    const SymmFrontType type = front.type;

    if( type == LDL_1D )
        FrontVanillaLowerForwardSolve( front.L1D, W );
    else if( type == LDL_SELINV_1D )
        FrontFastLowerForwardSolve( front.L1D, W );
    else if( type == LDL_SELINV_2D )
        FrontFastLowerForwardSolve( front.L2D, W );
    else if( type == LDL_INTRAPIV_1D )
        FrontIntraPivLowerForwardSolve( front.L1D, front.piv, W );
    else if( type == LDL_INTRAPIV_SELINV_1D )
        FrontFastIntraPivLowerForwardSolve( front.L1D, front.piv, W );
    else if( type == LDL_INTRAPIV_SELINV_2D )
        FrontFastIntraPivLowerForwardSolve( front.L2D, front.piv, W );
    else if( BlockFactorization(type) )
        FrontBlockLowerForwardSolve( front.L2D, W );
    else
        LogicError("Unsupported front type");
}

template<typename F>
inline void 
FrontLowerForwardSolve
( const DistSymmFront<F>& front, DistMatrix<F>& W )
{
    DEBUG_ONLY(CallStackEntry cse("FrontLowerForwardSolve"))
    const SymmFrontType type = front.type;

    if( type == LDL_2D )
        FrontVanillaLowerForwardSolve( front.L2D, W );
    else if( type == LDL_SELINV_2D )
        FrontFastLowerForwardSolve( front.L2D, W );
    else if( type == LDL_INTRAPIV_2D )
        FrontIntraPivLowerForwardSolve( front.L2D, front.piv, W );
    else if( type == LDL_INTRAPIV_SELINV_2D )
        FrontFastIntraPivLowerForwardSolve( front.L2D, front.piv, W );
    else if( BlockFactorization(type) )
        FrontBlockLowerForwardSolve( front.L2D, W );
    else
        LogicError("Unsupported front type");
}

} // namespace El

#endif // ifndef EL_FACTOR_NUMERIC_LOWERSOLVE_FRONTFORWARD_HPP
