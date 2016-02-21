#ifndef EL_BLAS_LIKE_LEVEL3_SYRK_LN_HPP
#define EL_BLAS_LIKE_LEVEL3_SYRK_LN_HPP

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace El {
namespace syrk {

template<typename T>
inline void
LN
( T alpha,
  const ElementalMatrix<T>& APre, 
        ElementalMatrix<T>& CPre,
  bool conjugate=false )
{
    DEBUG_ONLY(
      CSE cse("syrk::LN");
      AssertSameGrids( APre, CPre );
      if( APre.Height() != CPre.Height() || APre.Height() != CPre.Width() )
          LogicError
          ("Nonconformal:\n",DimsString(APre,"A"),"\n",DimsString(CPre,"C"))
    )
    const Int r = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR> AProx( APre );
    DistMatrixReadWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& A = AProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MC,  STAR> A1_MC_STAR(g);
    DistMatrix<T,VR,  STAR> A1_VR_STAR(g);
    DistMatrix<T,STAR,MR  > A1Trans_STAR_MR(g);

    A1_MC_STAR.AlignWith( C );
    A1_VR_STAR.AlignWith( C );
    A1Trans_STAR_MR.AlignWith( C );

    for( Int k=0; k<r; k+=bsize )
    {
        const Int nb = Min(bsize,r-k);
        auto A1 = A( ALL, IR(k,k+nb) );

        A1_VR_STAR = A1_MC_STAR = A1;
        Transpose( A1_VR_STAR, A1Trans_STAR_MR, conjugate );
        LocalTrrk( LOWER, alpha, A1_MC_STAR, A1Trans_STAR_MR, T(1), C );
    }
}

} // namespace syrk 
} // namespace El

#endif /* EL_BLAS_LIKE_LEVEL3_SYRK_LN_HPP */
