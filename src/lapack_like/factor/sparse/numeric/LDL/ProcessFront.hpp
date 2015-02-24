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
#ifndef EL_LDL_PROCESSFRONT_HPP
#define EL_LDL_PROCESSFRONT_HPP

namespace El {
namespace ldl {

template<typename F>
inline void ProcessFrontVanilla( Matrix<F>& AL, Matrix<F>& ABR, bool conjugate )
{
    DEBUG_ONLY(
      CallStackEntry cse("ldl::ProcessFrontVanilla");
      if( ABR.Height() != ABR.Width() )
          LogicError("ABR must be square");
      if( AL.Height() != AL.Width() + ABR.Width() )
          LogicError("AL and ABR don't have conformal dimensions");
    )
    const Int m = AL.Height();
    const Int n = AL.Width();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    Matrix<F> d1;
    Matrix<F> S21;

    Matrix<F> S21T, S21B;
    Matrix<F> AL21T, AL21B;

    const Int bsize = Blocksize();
    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(bsize,n-k);
        const Range<Int> ind1( k, k+nb ), 
                         ind2Vert( k+nb, m ), ind2Horz( k+nb, n );
        auto AL11 = AL( ind1,     ind1     );
        auto AL21 = AL( ind2Vert, ind1     );
        auto AL22 = AL( ind2Vert, ind2Horz );

        LDL( AL11, conjugate );
        GetDiagonal( AL11, d1 );

        Trsm( RIGHT, LOWER, orientation, UNIT, F(1), AL11, AL21 );

        S21 = AL21;
        DiagonalSolve( RIGHT, NORMAL, d1, AL21 );

        PartitionDown( S21, S21T, S21B, AL22.Width() );
        PartitionDown( AL21, AL21T, AL21B, AL22.Width() );
        Gemm( NORMAL, orientation, F(-1), S21, AL21T, F(1), AL22 );
        MakeTrapezoidal( LOWER, AL22 );
        Trrk( LOWER, NORMAL, orientation, F(-1), S21B, AL21B, F(1), ABR );
    }
}

template<typename F>
void ProcessFrontIntraPiv
( Matrix<F>& AL, Matrix<F>& subdiag, Matrix<Int>& piv, Matrix<F>& ABR,
  bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::ProcessFrontIntraPiv"))
    const Int n = AL.Width();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    Matrix<F> ATL, ABL;
    PartitionDown( AL, ATL, ABL, n );

    LDL( ATL, subdiag, piv, conjugate );
    auto diag = GetDiagonal(ATL);

    PermuteCols( ABL, piv );
    Trsm( RIGHT, LOWER, orientation, UNIT, F(1), ATL, ABL );
    Matrix<F> SBL( ABL );

    QuasiDiagonalSolve( RIGHT, LOWER, diag, subdiag, ABL, conjugate );
    Trrk( LOWER, NORMAL, orientation, F(-1), SBL, ABL, F(1), ABR );
}

template<typename F>
inline void ProcessFrontBlock
( Matrix<F>& AL, Matrix<F>& ABR, bool conjugate, bool intraPiv )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::ProcessFrontBlock"))
    Matrix<F> ATL, ABL;
    PartitionDown( AL, ATL, ABL, AL.Width() );

    // Make a copy of the original contents of ABL
    Matrix<F> BBL( ABL );

    if( intraPiv )
    {
        Matrix<Int> p;
        Matrix<F> dSub;
        // TODO: Expose the pivot type as an option?
        LDL( ATL, dSub, p, conjugate );

        // Solve against ABL and update ABR
        // NOTE: This does not exploit symmetry
        SolveAfter( ATL, dSub, p, ABL, conjugate );
        const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
        Gemm( NORMAL, orientation, F(-1), ABL, BBL, F(1), ABR );

        // Copy the original contents of ABL back
        ABL = BBL;

        // Finish inverting ATL
        TriangularInverse( LOWER, UNIT, ATL );
        Trdtrmm( LOWER, ATL, dSub, conjugate );
        // TODO: SymmetricPermutation
        MakeSymmetric( LOWER, ATL, conjugate );
        PermuteRows( ATL, p );
        PermuteCols( ATL, p );
    }
    else
    {
        // Call the standard routine
        ProcessFrontVanilla( AL, ABR, conjugate );

        // Copy the original contents of ABL back
        ABL = BBL;

        // Finish inverting ATL
        TriangularInverse( LOWER, UNIT, ATL );
        Trdtrmm( LOWER, ATL, conjugate );
        MakeSymmetric( LOWER, ATL, conjugate );
    }
}

template<typename F>
inline void ProcessFront( SymmFront<F>& front, SymmFrontType factorType )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::ProcessFront"))
    front.type = factorType;
    const bool pivoted = PivotedFactorization( factorType );
    if( BlockFactorization(factorType) )
        ProcessFrontBlock( front.L, front.work, front.isHermitian, pivoted );
    else if( pivoted )
    {
        ProcessFrontIntraPiv
        ( front.L, front.subdiag, front.piv, front.work, front.isHermitian );
        GetDiagonal( front.L, front.diag );
    }
    else
    {
        ProcessFrontVanilla( front.L, front.work, front.isHermitian );
        GetDiagonal( front.L, front.diag );
    }
}

template<typename F> 
inline void ProcessFrontVanilla
( DistMatrix<F>& AL, DistMatrix<F>& ABR, bool conjugate=false )
{
    DEBUG_ONLY(
      CallStackEntry cse("ldl::ProcessFrontVanilla");
      if( ABR.Height() != ABR.Width() )
          LogicError("ABR must be square");
      if( AL.Height() != AL.Width()+ABR.Height() )
          LogicError("AL and ABR must have compatible dimensions");
      if( AL.Grid() != ABR.Grid() )
          LogicError("AL and ABR must use the same grid");
      if( ABR.ColAlign() !=
          (AL.ColAlign()+AL.Width()) % AL.Grid().Height() )
          LogicError("AL and ABR must have compatible col alignments");
      if( ABR.RowAlign() != 
          (AL.RowAlign()+AL.Width()) % AL.Grid().Width() )
          LogicError("AL and ABR must have compatible row alignments");
    )
    const Grid& g = AL.Grid();
    const Int m = AL.Height();
    const Int n = AL.Width();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    DistMatrix<F,STAR,STAR> AL11_STAR_STAR(g);
    DistMatrix<F,STAR,STAR> d1_STAR_STAR(g);
    DistMatrix<F,VC,  STAR> AL21_VC_STAR(g);
    DistMatrix<F,VR,  STAR> AL21_VR_STAR(g);
    DistMatrix<F,STAR,MC  > S21Trans_STAR_MC(g);
    DistMatrix<F,STAR,MR  > AL21Trans_STAR_MR(g);

    DistMatrix<F,STAR,MC> leftL(g), leftR(g);
    DistMatrix<F,STAR,MR> rightL(g), rightR(g);
    DistMatrix<F> AL22T(g), AL22B(g);

    const Int bsize = Blocksize();
    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(bsize,n-k);
        const Range<Int> ind1( k, k+nb ),
                         ind2Vert( k+nb, m ), ind2Horz( k+nb, n );
        auto AL11 = AL( ind1,     ind1     );
        auto AL21 = AL( ind2Vert, ind1     );
        auto AL22 = AL( ind2Vert, ind2Horz );

        AL11_STAR_STAR = AL11; 
        LocalLDL( AL11_STAR_STAR, conjugate );
        GetDiagonal( AL11_STAR_STAR, d1_STAR_STAR );
        AL11 = AL11_STAR_STAR;

        AL21_VC_STAR.AlignWith( AL22 );
        AL21_VC_STAR = AL21;
        LocalTrsm
        ( RIGHT, LOWER, orientation, UNIT, F(1), AL11_STAR_STAR, AL21_VC_STAR );

        S21Trans_STAR_MC.AlignWith( AL22 );
        Transpose( AL21_VC_STAR, S21Trans_STAR_MC );
        DiagonalSolve( RIGHT, NORMAL, d1_STAR_STAR, AL21_VC_STAR );
        AL21Trans_STAR_MR.AlignWith( AL22 );
        Transpose( AL21_VC_STAR, AL21Trans_STAR_MR, conjugate );

        // Partition the update of the bottom-right corner into three pieces
        PartitionRight( S21Trans_STAR_MC, leftL, leftR, AL22.Width() );
        PartitionRight( AL21Trans_STAR_MR, rightL, rightR, AL22.Width() );
        PartitionDown( AL22, AL22T, AL22B, AL22.Width() );
        LocalTrrk( LOWER, orientation,  F(-1), leftL, rightL, F(1), AL22T );
        LocalGemm( orientation, NORMAL, F(-1), leftR, rightL, F(1), AL22B );
        LocalTrrk( LOWER, orientation,  F(-1), leftR, rightR, F(1), ABR );

        DiagonalSolve( LEFT, NORMAL, d1_STAR_STAR, S21Trans_STAR_MC );
        Transpose( S21Trans_STAR_MC, AL21 );
    }
}

template<typename F>
void ProcessFrontIntraPiv
( DistMatrix<F>& AL, DistMatrix<F,MD,STAR>& subdiag, 
  DistMatrix<Int,VC,STAR>& p, DistMatrix<F>& ABR, bool conjugate )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::ProcessFrontIntraPiv"))
    const Grid& g = AL.Grid();
    const Int n = AL.Width();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    
    DistMatrix<F> ATL(g), ABL(g);
    PartitionDown( AL, ATL, ABL, n );

    LDL( ATL, subdiag, p, conjugate );
    auto diag = GetDiagonal(ATL);

    PermuteCols( ABL, p );
    Trsm( RIGHT, LOWER, orientation, UNIT, F(1), ATL, ABL );
    DistMatrix<F,MC,STAR> SBL_MC_STAR(g);
    SBL_MC_STAR.AlignWith( ABR );
    SBL_MC_STAR = ABL;

    QuasiDiagonalSolve( RIGHT, LOWER, diag, subdiag, ABL, conjugate );
    DistMatrix<F,VR,STAR> ABL_VR_STAR(g);
    DistMatrix<F,STAR,MR> ABLTrans_STAR_MR(g);
    ABL_VR_STAR.AlignWith( ABR );
    ABLTrans_STAR_MR.AlignWith( ABR );
    ABL_VR_STAR = ABL;
    Transpose( ABL_VR_STAR, ABLTrans_STAR_MR, conjugate );
    LocalTrrk( LOWER, F(-1), SBL_MC_STAR, ABLTrans_STAR_MR, F(1), ABR );
}

template<typename F>
inline void ProcessFrontBlock
( DistMatrix<F>& AL, DistMatrix<F>& ABR, bool conjugate, bool intraPiv )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::ProcessFrontBlock"))
    const Grid& g = AL.Grid();
    DistMatrix<F> ATL(g), ABL(g);
    PartitionDown( AL, ATL, ABL, AL.Width() );

    // Make a copy of the original contents of ABL
    DistMatrix<F> BBL( ABL );

    if( intraPiv )
    {
        DistMatrix<Int,VC,STAR> p( ATL.Grid() );
        DistMatrix<F,MD,STAR> dSub( ATL.Grid() );
        // TODO: Expose the pivot type as an option?
        LDL( ATL, dSub, p, conjugate );

        // Solve against ABL and update ABR
        // NOTE: This update does not exploit symmetry
        SolveAfter( ATL, dSub, p, ABL, conjugate );
        const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
        Gemm( NORMAL, orientation, F(-1), ABL, BBL, F(1), ABR );

        // Copy the original contents of ABL back
        ABL = BBL;

        // Finish inverting ATL
        TriangularInverse( LOWER, UNIT, ATL );
        Trdtrmm( LOWER, ATL, dSub, conjugate );
        ApplyInverseSymmetricPivots( LOWER, ATL, p, conjugate );
    }
    else
    {
        // Call the standard routine
        ProcessFrontVanilla( AL, ABR, conjugate );

        // Copy the original contents of ABL back
        ABL = BBL;

        // Finish inverting ATL
        TriangularInverse( LOWER, UNIT, ATL );
        Trdtrmm( LOWER, ATL, conjugate );
    }
    MakeSymmetric( LOWER, ATL, conjugate );
}

template<typename F>
inline void ProcessFront( DistSymmFront<F>& front, SymmFrontType factorType )
{
    DEBUG_ONLY(
      CallStackEntry cse("ldl::ProcessFront");
      if( FrontIs1D(front.type) )
          LogicError("Expected front to be in a 2D distribution");
    )
    front.type = factorType;
    const bool pivoted = PivotedFactorization( factorType );
    const Grid& grid = front.L2D.Grid();

    if( BlockFactorization(factorType) )
    {
        ProcessFrontBlock( front.L2D, front.work, front.isHermitian, pivoted );
    }
    else if( pivoted )
    {
        DistMatrix<F,MD,STAR> subdiag(grid);
        front.piv.SetGrid( grid );
        ProcessFrontIntraPiv
        ( front.L2D, subdiag, front.piv, front.work, front.isHermitian );

        auto diag = GetDiagonal( front.L2D );
        front.diag.SetGrid( grid );
        front.subdiag.SetGrid( grid ); 
        front.diag = diag;
        front.subdiag = subdiag;
    }
    else
    {
        ProcessFrontVanilla( front.L2D, front.work, front.isHermitian );

        auto diag = GetDiagonal( front.L2D );
        front.diag.SetGrid( grid );
        front.diag = diag;
    }
}

} // namespace ldl
} // namespace El

#endif // ifndef EL_LDL_PROCESSFRONT_HPP
