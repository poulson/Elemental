/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_LDL_PIVOTED_UNBLOCKED_HPP
#define EL_LDL_PIVOTED_UNBLOCKED_HPP

namespace El {
namespace ldl {
namespace pivot {

template<typename F>
inline LDLPivot
Select( const Matrix<F>& A, LDLPivotType pivotType, Base<F> gamma )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::pivot::Select"))
    LDLPivot pivot;
    switch( pivotType )
    {
    case BUNCH_KAUFMAN_A: 
    case BUNCH_KAUFMAN_C: pivot = BunchKaufmanA( A, gamma ); break;
    case BUNCH_KAUFMAN_D: pivot = BunchKaufmanD( A, gamma ); break;
    case BUNCH_PARLETT:   pivot = BunchParlett( A, gamma ); break;
    default: LogicError("This pivot type not yet supported");
    }
    return pivot;
}

template<typename F>
inline LDLPivot
Select( const DistMatrix<F>& A, LDLPivotType pivotType, Base<F> gamma )
{
    DEBUG_ONLY(CallStackEntry cse("ldl::pivot::Select"))
    LDLPivot pivot;
    switch( pivotType )
    {
    case BUNCH_KAUFMAN_A: 
    case BUNCH_KAUFMAN_C: pivot = BunchKaufmanA( A, gamma ); break;
    case BUNCH_KAUFMAN_D: pivot = BunchKaufmanD( A, gamma ); break;
    case BUNCH_PARLETT:   pivot = BunchParlett( A, gamma ); break;
    default: LogicError("This pivot type not yet supported");
    }
    return pivot;
}

// Unblocked sequential pivoted LDL
template<typename F>
inline void
Unblocked
( Matrix<F>& A, Matrix<F>& dSub, Matrix<Int>& p, bool conjugate=false,
  LDLPivotType pivotType=BUNCH_KAUFMAN_A, Base<F> gamma=0 )
{
    DEBUG_ONLY(
        CallStackEntry cse("ldl::pivot::Unblocked");
        if( A.Height() != A.Width() )
            LogicError("A must be square");
    )
    const Int n = A.Height();
    if( n == 0 )
    {
        dSub.Resize( 0, 1 );
        p.Resize( 0, 1 );
        return;
    }
    Zeros( dSub, n-1, 1 );

    // Initialize the permutation to the identity
    p.Resize( n, 1 );
    for( Int j=0; j<n; ++j )
        p.Set( j, 0, j );
     
    Matrix<F> Y21;

    Int k=0;
    while( k < n )
    {
        const Range<Int> indB( k, n ), indR( k, n );

        // Determine the pivot (block)
        auto ABR = A( indB, indR );
        if( pivotType == BUNCH_KAUFMAN_C )
        {
            LogicError("Have not yet generalized pivot storage");
            const auto diagMax = VectorMaxAbs( GetDiagonal(ABR) );
            SymmetricSwap( LOWER, A, k, k+diagMax.index, conjugate );
        }
        const LDLPivot pivot = Select( ABR, pivotType, gamma );

        for( Int l=0; l<pivot.nb; ++l )
        {
            const Int from = k + pivot.from[l];
            SymmetricSwap( LOWER, A, k+l, from, conjugate );
            RowSwap( p, k+l, from );
        }

        // Update trailing submatrix and store pivots
        const Range<Int> ind1( k,          k+pivot.nb ),
                         ind2( k+pivot.nb, n          );
        if( pivot.nb == 1 )
        {
            // Rank-one update: A22 -= a21 inv(delta11) a21'
            const F delta11Inv = F(1)/ABR.Get(0,0);
            auto a21 = A( ind2, ind1 );
            auto A22 = A( ind2, ind2 );
            Syr( LOWER, -delta11Inv, a21, A22, conjugate );
            Scale( delta11Inv, a21 );
        }
        else
        {
            // Rank-two update: A22 -= A21 inv(D11) A21'
            auto D11 = A( ind1, ind1 );
            auto A21 = A( ind2, ind1 );
            auto A22 = A( ind2, ind2 );
            Y21 = A21;
            Symmetric2x2Solve( RIGHT, LOWER, D11, A21, conjugate );
            Trr2( LOWER, F(-1), A21, Y21, A22, conjugate );

            // Only leave the main diagonal of D in A, so that routines like
            // Trsm can still be used. Thus, return the subdiagonal.
            dSub.Set( k, 0, D11.Get(1,0) );
            D11.Set( 1, 0, 0 );
        }
        k += pivot.nb;
    }
}

template<typename F>
inline void
Unblocked
( AbstractDistMatrix<F>& APre, AbstractDistMatrix<F>& dSub, 
  AbstractDistMatrix<Int>& p, bool conjugate=false, 
  LDLPivotType pivotType=BUNCH_KAUFMAN_A, Base<F> gamma=0 )
{
    DEBUG_ONLY(
        CallStackEntry cse("ldl::pivot::Unblocked");
        if( APre.Height() != APre.Width() )
            LogicError("A must be square");
        AssertSameGrids( APre, dSub, p );
    )
    const Int n = APre.Height();
    const Grid& g = APre.Grid();

    Zeros( dSub, n-1, 1 );
    p.Resize( n, 1 );

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    // Initialize the permutation to the identity
    for( Int iLoc=0; iLoc<p.LocalHeight(); ++iLoc )
        p.SetLocal( iLoc, 0, p.GlobalRow(iLoc) );

    DistMatrix<F> Y21(g);
    DistMatrix<F,STAR,STAR> D11_STAR_STAR(g);

    Int k=0;
    while( k < n )
    {
        const Range<Int> indB( k, n ), indR( k, n );

        // Determine the pivot (block)
        auto ABR = A( indB, indR );
        if( pivotType == BUNCH_KAUFMAN_C )
        {
            LogicError("Have not yet generalized pivot storage");
            const auto diagMax = VectorMaxAbs( GetDiagonal(ABR) );
            SymmetricSwap( LOWER, A, k, k+diagMax.index, conjugate );
        }
        const LDLPivot pivot = Select( ABR, pivotType, gamma );

        for( Int l=0; l<pivot.nb; ++l )
        {
            const Int from = k + pivot.from[l];
            SymmetricSwap( LOWER, A, k+l, from, conjugate );
            RowSwap( p, k+l, from );
        }


        // Update trailing submatrix and store pivots
        const Range<Int> ind1( k,          k+pivot.nb ),
                         ind2( k+pivot.nb, n          );
        if( pivot.nb == 1 )
        {
            // Rank-one update: A22 -= a21 inv(delta11) a21'
            const F delta11Inv = F(1)/ABR.Get(0,0);
            auto a21 = A( ind2, ind1 );
            auto A22 = A( ind2, ind2 );
            Syr( LOWER, -delta11Inv, a21, A22, conjugate );
            Scale( delta11Inv, a21 );
        }
        else
        {
            // Rank-two update: A22 -= A21 inv(D11) A21'
            auto D11 = A( ind1, ind1 );
            auto A21 = A( ind2, ind1 );
            auto A22 = A( ind2, ind2 );
            Y21 = A21;
            D11_STAR_STAR = D11;
            Symmetric2x2Solve( RIGHT, LOWER, D11_STAR_STAR, A21, conjugate );
            Trr2( LOWER, F(-1), A21, Y21, A22, conjugate );

            // Only leave the main diagonal of D in A, so that routines like
            // Trsm can still be used. Thus, return the subdiagonal.
            dSub.Set( k, 0, D11_STAR_STAR.GetLocal(1,0) );
            D11.Set( 1, 0, 0 );
        }
        k += pivot.nb;
    }
}

} // namespace pivot
} // namespace ldl
} // namespace El

#endif // ifndef EL_LDL_PIVOTED_UNBLOCKED_HPP
