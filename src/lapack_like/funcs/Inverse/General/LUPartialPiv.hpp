/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_INVERSE_LUPARTIALPIV_HPP
#define EL_INVERSE_LUPARTIALPIV_HPP

namespace El {
namespace inverse {

// Start by forming the partially pivoted LU decomposition of A,
//     P A = L U,
// then inverting the system of equations,
//     inv(A) inv(P) = inv(U) inv(L),
// then,
//     inv(A) = inv(U) inv(L) P.

template<typename F> 
void AfterLUPartialPiv( Matrix<F>& A, const Matrix<Int>& p )
{
    DEBUG_ONLY(CallStackEntry cse("inverse::AfterLUPartialPiv"))
    if( A.Height() != A.Width() )
        LogicError("Cannot invert non-square matrices");
    if( A.Height() != p.Height() )
        LogicError("Pivot vector is incorrect length");

    TriangularInverse( UPPER, NON_UNIT, A );

    const Int n = A.Height();
    const Range<Int> outerInd( 0, n );

    // Solve inv(A) L = inv(U) for inv(A)
    const Int bsize = Blocksize();
    const Int kLast = LastOffset( n, bsize );
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,n-k);

        const Range<Int> ind1( k,    k+nb ),
                         ind2( k+nb, n    );

        auto A1 = A( outerInd, ind1 );
        auto A2 = A( outerInd, ind2 );

        auto A11 = A( ind1, ind1 );
        auto A21 = A( ind2, ind1 );

        // Copy out L1
        auto L11( A11 );
        auto L21( A21 );

        // Zero the strictly lower triangular portion of A1
        MakeTrapezoidal( UPPER, A11 );
        Zero( A21 );

        // Perform the lazy update of A1
        Gemm( NORMAL, NORMAL, F(-1), A2, L21, F(1), A1 );

        // Solve against this diagonal block of L11
        Trsm( RIGHT, LOWER, NORMAL, UNIT, F(1), L11, A1 );
    }

    // inv(A) := inv(A) P
    InversePermuteCols( A, p );
}

template<typename F> 
inline void
LUPartialPiv( Matrix<F>& A )
{
    DEBUG_ONLY(CallStackEntry cse("inverse::LUPartialPiv"))
    if( A.Height() != A.Width() )
        LogicError("Cannot invert non-square matrices");
    Matrix<Int> p;
    El::LU( A, p );
    inverse::AfterLUPartialPiv( A, p );
}

template<typename F> 
void AfterLUPartialPiv
( AbstractDistMatrix<F>& APre, const AbstractDistMatrix<Int>& p )
{
    DEBUG_ONLY(CallStackEntry cse("inverse::AfterLUPartialPiv"))

    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;

    if( A.Height() != A.Width() )
        LogicError("Cannot invert non-square matrices");
    if( A.Height() != p.Height() )
        LogicError("Pivot vector is incorrect length");
    AssertSameGrids( A, p );

    TriangularInverse( UPPER, NON_UNIT, A );

    const Grid& g = A.Grid();
    DistMatrix<F,VC,  STAR> A1_VC_STAR(g);
    DistMatrix<F,STAR,STAR> L11_STAR_STAR(g);
    DistMatrix<F,VR,  STAR> L21_VR_STAR(g);
    DistMatrix<F,STAR,MR  > L21Trans_STAR_MR(g);
    DistMatrix<F,MC,  STAR> Z1(g);

    const Int n = A.Height();
    const Range<Int> outerInd( 0, n );

    // Solve inv(A) L = inv(U) for inv(A)
    const Int bsize = Blocksize();
    const Int kLast = LastOffset( n, bsize );
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,n-k);

        const Range<Int> ind1( k,    k+nb ),
                         ind2( k+nb, n    );

        auto A1 = A( outerInd, ind1 );
        auto A2 = A( outerInd, ind2 );

        auto A11 = A( ind1, ind1 );
        auto A21 = A( ind2, ind1 );

        // Copy out L1
        L11_STAR_STAR = A11;
        L21_VR_STAR.AlignWith( A2 );
        L21_VR_STAR = A21;
        L21Trans_STAR_MR.AlignWith( A2 );
        Transpose( L21_VR_STAR, L21Trans_STAR_MR );

        // Zero the strictly lower triangular portion of A1
        MakeTrapezoidal( UPPER, A11 );
        Zero( A21 );

        // Perform the lazy update of A1
        Z1.AlignWith( A1 );
        Zeros( Z1, n, nb );
        LocalGemm( NORMAL, TRANSPOSE, F(-1), A2, L21Trans_STAR_MR, F(0), Z1 );
        AxpyContract( F(1), Z1, A1 );

        // Solve against this diagonal block of L11
        A1_VC_STAR = A1;
        LocalTrsm
        ( RIGHT, LOWER, NORMAL, UNIT, F(1), L11_STAR_STAR, A1_VC_STAR );
        A1 = A1_VC_STAR;
    }

    // inv(A) := inv(A) P
    InversePermuteCols( A, p );
}

template<typename F> 
inline void
LUPartialPiv( AbstractDistMatrix<F>& A )
{
    DEBUG_ONLY(CallStackEntry cse("inverse::LUPartialPiv"))
    if( A.Height() != A.Width() )
        LogicError("Cannot invert non-square matrices");
    const Grid& g = A.Grid();
    DistMatrix<Int,VC,STAR> p( g );
    El::LU( A, p );
    inverse::AfterLUPartialPiv( A, p );
}

} // namespace inverse
} // namespace El

#endif // ifndef EL_INVERSE_LUPARTIALPIV_HPP
