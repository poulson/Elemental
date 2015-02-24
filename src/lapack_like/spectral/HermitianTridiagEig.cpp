/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#include "./HermitianTridiagEig/Sort.hpp"

// NOTE: dSubReal and ZReal could be packed into their complex counterparts

namespace El {

// Return eigenvalues
// ==================

namespace herm_tridiag_eig {

template<typename Real>
inline void Helper
( Matrix<Real>& d, Matrix<Real>& dSub, Matrix<Real>& w, SortType sort,
  const HermitianEigSubset<Real>& subset )
{
    const Int n = d.Height();
    w.Resize( n, 1 );
    if( subset.rangeSubset )
    {
         const Int k = lapack::SymmetricTridiagEig
          ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer(), 
            subset.lowerBound, subset.upperBound );
         w.Resize( k, 1 );
    }
    else if( subset.indexSubset )
    {
        const Int numEig = subset.upperIndex-subset.lowerIndex+1;
        lapack::SymmetricTridiagEig
        ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer(), 
          int(subset.lowerIndex), int(subset.upperIndex) );
        w.Resize( numEig, 1 );
    }
    else
        lapack::SymmetricTridiagEig
        ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer() );
    Sort( w, sort );
}

template<typename Real>
inline void Helper
( Matrix<Real>& d, Matrix<Complex<Real>>& dSub, Matrix<Real>& w, SortType sort,
  const HermitianEigSubset<Real>& subset )
{
    typedef Complex<Real> C;
    const Int n = d.Height();
    Matrix<Real> dSubReal( n-1, 1 );
    C yLast = 1;
    for( Int j=0; j<n-1; ++j )
    {
        const C psi = dSub.Get(j,0);
        const Real psiAbs = Abs(psi);
        if( psiAbs == Real(0) )
            yLast = 1;
        else
            yLast = ComplexFromPolar(Real(1),Arg(psi*yLast));
        dSubReal.Set( j, 0, psiAbs );
    }
    HermitianTridiagEig( d, dSubReal, w, sort, subset );
}

} // namespace herm_tridiag_eig

template<typename F>
void HermitianTridiagEig
( Matrix<Base<F>>& d, Matrix<F>& dSub, Matrix<Base<F>>& w, SortType sort, 
  const HermitianEigSubset<Base<F>>& subset )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEig"))
    herm_tridiag_eig::Helper( d, dSub, w, sort, subset );
}

namespace herm_tridiag_eig {

template<typename Real>
inline void Helper
( const AbstractDistMatrix<Real>& d,
  const AbstractDistMatrix<Real>& dSub,
        AbstractDistMatrix<Real>& wPre, SortType sort,
  const HermitianEigSubset<Real>& subset )
{
    ProxyCtrl wCtrl;
    wCtrl.colConstrain = true;
    wCtrl.colAlign = 0;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre, wCtrl ); 
    auto& w = *wPtr;

    // Force the computation to take place with double-precision since PMRRR
    // currently only supports this case
    const Int n = d.Height();
    const Grid& g = d.Grid();
    DistMatrix<double,STAR,STAR> d_STAR_STAR(g), dSub_STAR_STAR(g);
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );

    vector<double> wVector(n);
    herm_tridiag_eig::Info info;
    if( subset.rangeSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), w.ColComm(), 
            subset.lowerBound, subset.upperBound );
    else if( subset.indexSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), w.ColComm(), 
            int(subset.lowerIndex), int(subset.upperIndex) );
    else
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), w.ColComm() );
    w.Resize( info.numGlobalEigenvalues, 1 );
    for( Int iLoc=0; iLoc<w.LocalHeight(); ++iLoc )
        w.SetLocal( iLoc, 0, Real(wVector[iLoc]) );
    Sort( w, sort );
}

template<typename Real>
inline void Helper
( const AbstractDistMatrix<Real         >& d,
  const AbstractDistMatrix<Complex<Real>>& dSub,
        AbstractDistMatrix<Real         >& wPre, 
  SortType sort, const HermitianEigSubset<Real>& subset )
{
    ProxyCtrl wCtrl;
    wCtrl.colConstrain = true;
    wCtrl.colAlign = 0;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre, wCtrl ); 
    auto& w = *wPtr;

    // Force the computation to take place with double-precision since PMRRR
    // currently only supports this case
    const Int n = d.Height();
    const Grid& g = d.Grid();
    DistMatrix<double,STAR,STAR> d_STAR_STAR(g);
    DistMatrix<Complex<double>,STAR,STAR> dSub_STAR_STAR(g);
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );

    DistMatrix<double,STAR,STAR> dSubReal(g);
    dSubReal.Resize( n-1, 1, n );

    Complex<double> yLast = 1;
    for( Int j=0; j<n-1; ++j )
    {
        const Complex<double> psi = dSub_STAR_STAR.GetLocal(j,0);
        const double psiAbs = Abs(psi);
        if( psiAbs == double(0) )
            yLast = 1;
        else
            yLast = ComplexFromPolar(double(1),Arg(psi*yLast));
        dSubReal.SetLocal( j, 0, psiAbs );
    }

    herm_tridiag_eig::Info info;
    vector<double> wVector(n);
    if( subset.rangeSubset )
    {
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(),
            wVector.data(), w.ColComm(),
            subset.lowerBound, subset.upperBound );
    }
    else if( subset.indexSubset )
    {
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(),
            wVector.data(), w.ColComm(),
            int(subset.lowerIndex), int(subset.upperIndex) );
    }
    else
    {
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(),
            wVector.data(), w.ColComm() );
    }
    w.Resize( info.numGlobalEigenvalues, 1 );
    for( Int iLoc=0; iLoc<w.LocalHeight(); ++iLoc )
        w.SetLocal( iLoc, 0, Real(wVector[iLoc]) );

    Sort( w, sort );
}

} // namespace herm_tridiag_eig

template<typename F>
void HermitianTridiagEig
( const AbstractDistMatrix<Base<F>>& d,
  const AbstractDistMatrix<F      >& dSub,
        AbstractDistMatrix<Base<F>>& w, 
  SortType sort, const HermitianEigSubset<Base<F>>& subset )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEig"))
    herm_tridiag_eig::Helper( d, dSub, w, sort, subset );
}

// Return eigenpairs
// =================

namespace herm_tridiag_eig {

template<typename Real>
inline void Helper
( Matrix<Real>& d, Matrix<Real>& dSub, Matrix<Real>& w, Matrix<Real>& Z,
  SortType sort, const HermitianEigSubset<Real>& subset )
{
    const Int n = d.Height();
    w.Resize( n, 1 );
    if( subset.rangeSubset )
    {
         Z.Resize( n, n );
         const Int k = lapack::SymmetricTridiagEig
          ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer(), 
            Z.Buffer(), int(Z.LDim()),
            subset.lowerBound, subset.upperBound );
         w.Resize( k, 1 );
         Z.Resize( n, k );
    }
    else if( subset.indexSubset )
    {
        const Int numEig = subset.upperIndex-subset.lowerIndex+1;
        Z.Resize( n, numEig );
        lapack::SymmetricTridiagEig
        ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer(), 
          Z.Buffer(), int(Z.LDim()),
          int(subset.lowerIndex), int(subset.upperIndex) );
        w.Resize( numEig, 1 );
    }
    else
    {
        Z.Resize( n, n );
        lapack::SymmetricTridiagEig
        ( int(n), d.Buffer(), dSub.Buffer(), w.Buffer(), 
          Z.Buffer(), int(Z.LDim()) );
    }
    herm_eig::Sort( w, Z, sort );
}

// (Y^H T Y) ZHat = ZHat Lambda
// T (Y ZHat) = (Y ZHat) Lambda
template<typename Real>
inline void Helper
( Matrix<Real>& d, Matrix<Complex<Real>>& dSub, Matrix<Real>& w, 
  Matrix<Complex<Real>>& Z, SortType sort,
  const HermitianEigSubset<Real>& subset )
{
    typedef Complex<Real> C;
    const Int n = d.Height();
    Matrix<Real> dSubReal( n-1, 1 );
    Matrix<C> y( n, 1 );
    y.Set( 0, 0, 1 );
    for( Int j=0; j<n-1; ++j )
    {
        const C psi = dSub.Get(j,0);
        const Real psiAbs = Abs(psi);
        if( psiAbs == Real(0) )
            y.Set( j+1, 0, 1 );
        else
            y.Set( j+1, 0, ComplexFromPolar(Real(1),Arg(psi*y.Get(j,0))) );
        dSubReal.Set( j, 0, psiAbs );
    }
    Matrix<Real> ZReal;
    HermitianTridiagEig( d, dSubReal, w, ZReal, sort, subset );
    Z.Resize( n, ZReal.Width() );
    for( Int j=0; j<ZReal.Width(); ++j )
        for( Int i=0; i<n; ++i )
            Z.Set( i, j, y.Get(i,0)*ZReal.Get(i,j) );
}

} // namespace herm_tridiag_eig

template<typename F>
void HermitianTridiagEig
( Matrix<Base<F>>& d, Matrix<F>& dSub, Matrix<Base<F>>& w, Matrix<F>& Z, 
  SortType sort, const HermitianEigSubset<Base<F>>& subset )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEig"))
    herm_tridiag_eig::Helper( d, dSub, w, Z, sort, subset );
}

namespace herm_tridiag_eig {

template<typename Real>
inline void Helper
( const AbstractDistMatrix<Real>& d,
  const AbstractDistMatrix<Real>& dSub,
        AbstractDistMatrix<Real>& wPre, 
        AbstractDistMatrix<Real>& ZPre, 
  SortType sort, const HermitianEigSubset<Real>& subset )
{
    // NOTE: The computation forces double-precision due to PMRRR limitations

    const Int n = d.Height();
    const Grid& g = d.Grid();

    ProxyCtrl wCtrl, ZCtrl;
    wCtrl.colConstrain = true;
    wCtrl.colAlign = 0;
    ZCtrl.rowConstrain = true;
    ZCtrl.rowAlign = 0;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre, wCtrl );   
    auto& w = *wPtr;
    auto ZPtr = WriteProxy<double,STAR,VR>( &ZPre, ZCtrl ); 
    auto& Z = *ZPtr;

    DistMatrix<double,STAR,STAR> d_STAR_STAR(g), dSub_STAR_STAR(g);
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );

    Int k;
    if( subset.rangeSubset )
    {
        vector<double> dVector(n), dSubVector(n), wVector(n);
        MemCopy( dVector.data(), d_STAR_STAR.Buffer(), n );
        MemCopy( dSubVector.data(), dSub_STAR_STAR.Buffer(), n-1 );
        auto estimate = herm_tridiag_eig::EigEstimate
          ( int(n), dVector.data(), dSubVector.data(),
            wVector.data(), w.ColComm(),
            subset.lowerBound, subset.upperBound );
        SwapClear( dVector );
        SwapClear( dSubVector );
        k = estimate.numGlobalEigenvalues;
    }
    else if( subset.indexSubset )
        k = ( n==0 ? 0 : subset.upperIndex-subset.lowerIndex+1 );
    else
        k = n;
    Z.Resize( n, k );

    herm_tridiag_eig::Info info;
    vector<double> wVector(n);
    if( subset.rangeSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), Z.Buffer(), Z.LDim(), w.ColComm(),
            subset.lowerBound, subset.upperBound );
    else if( subset.indexSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), Z.Buffer(), Z.LDim(), w.ColComm(),
            int(subset.lowerIndex), int(subset.upperIndex) );
    else
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), 
            wVector.data(), Z.Buffer(), Z.LDim(), w.ColComm() );
    w.Resize( info.numGlobalEigenvalues, 1 );
    Z.Resize( n, info.numGlobalEigenvalues );
    for( Int iLoc=0; iLoc<w.LocalHeight(); ++iLoc )
        w.SetLocal( iLoc, 0, Real(wVector[iLoc]) );

    herm_eig::Sort( w, Z, sort );
}

template<typename Real>
inline void Helper
( const AbstractDistMatrix<Real         >& d,
  const AbstractDistMatrix<Complex<Real>>& dSub,
        AbstractDistMatrix<Real         >& wPre, 
        AbstractDistMatrix<Complex<Real>>& ZPre, 
  SortType sort, const HermitianEigSubset<Real>& subset )
{
    // NOTE: The computation forces double-precision due to PMRRR limitations
    const Int n = d.Height();
    const Grid& g = d.Grid();
    typedef Complex<Real> C;

    DistMatrix<double,STAR,STAR> d_STAR_STAR(g);
    DistMatrix<Complex<double>,STAR,STAR> dSub_STAR_STAR(g);
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );

    DistMatrix<Complex<double>,STAR,STAR> y(n,1,g);
    DistMatrix<double,STAR,STAR> dSubReal(g);
    dSubReal.Resize( n-1, 1, n );

    y.SetLocal(0,0,1);
    for( Int j=0; j<n-1; ++j )
    {
        const Complex<double> psi = dSub_STAR_STAR.GetLocal(j,0);
        const double psiAbs = Abs(psi);
        if( psiAbs == double(0) )
            y.SetLocal( j+1, 0, 1 );
        else
            y.SetLocal
            ( j+1, 0, ComplexFromPolar(double(1),Arg(psi*y.GetLocal(j,0))) );
        dSubReal.SetLocal( j, 0, psiAbs );
    }

    ProxyCtrl wCtrl, ZCtrl;
    wCtrl.colConstrain = true;
    wCtrl.colAlign = 0;
    ZCtrl.rowConstrain = true;
    ZCtrl.rowAlign = 0;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre, wCtrl ); auto& w = *wPtr;
    auto ZPtr = WriteProxy<C,STAR,VR>( &ZPre, ZCtrl );    auto& Z = *ZPtr;

    Int k;
    if( subset.rangeSubset )
    {
        vector<double> dVector(n), dSubVector(n), wVector(n);
        MemCopy( dVector.data(), d_STAR_STAR.Buffer(), n );
        MemCopy( dSubVector.data(), dSubReal.Buffer(), n-1 );
        auto estimate = herm_tridiag_eig::EigEstimate
          ( int(n), dVector.data(), dSubVector.data(),
            wVector.data(), w.ColComm(),
            subset.lowerBound, subset.upperBound );
        SwapClear( dVector );
        SwapClear( dSubVector );
        k = estimate.numGlobalEigenvalues;
    }
    else if( subset.indexSubset )
        k = ( n==0 ? 0 : subset.upperIndex-subset.lowerIndex+1 );
    else
        k = n;
    DistMatrix<double,STAR,VR> ZReal(g);
    ZReal.Resize( n, k );

    herm_tridiag_eig::Info info;
    vector<double> wVector(n);
    if( subset.rangeSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(), 
            wVector.data(), ZReal.Buffer(), ZReal.LDim(), w.ColComm(),
            subset.lowerBound, subset.upperBound );
    else if( subset.indexSubset )
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(), 
            wVector.data(), ZReal.Buffer(), ZReal.LDim(), w.ColComm(),
            int(subset.lowerIndex), int(subset.upperIndex) );
    else
        info = herm_tridiag_eig::Eig
          ( int(n), d_STAR_STAR.Buffer(), dSubReal.Buffer(), 
            wVector.data(), ZReal.Buffer(), ZReal.LDim(), w.ColComm() );

    w.Resize( info.numGlobalEigenvalues, 1 );
    for( Int iLoc=0; iLoc<w.LocalHeight(); ++iLoc )
        w.SetLocal( iLoc, 0, wVector[iLoc] );

    ZReal.Resize( n, info.numGlobalEigenvalues );
    herm_eig::Sort( w, ZReal, sort );

    Z.Resize( n, info.numGlobalEigenvalues );
    for( Int jLoc=0; jLoc<Z.LocalWidth(); ++jLoc )
        for( Int i=0; i<n; ++i )
            Z.SetLocal( i, jLoc, C(y.GetLocal(i,0)*ZReal.GetLocal(i,jLoc)) );
}

} // namespace herm_tridiag_eig

template<typename F>
void HermitianTridiagEig
( const AbstractDistMatrix<Base<F>>& d, const AbstractDistMatrix<F>& dSub,
        AbstractDistMatrix<Base<F>>& w,       AbstractDistMatrix<F>& Z, 
  SortType sort, const HermitianEigSubset<Base<F>>& subset )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEig"))
    herm_tridiag_eig::Helper( d, dSub, w, Z, sort, subset );
}

template<typename Real>
Int HermitianTridiagEigEstimate
( const AbstractDistMatrix<Real>& d, const AbstractDistMatrix<Real>& dSub,
        mpi::Comm wColComm, Real vl, Real vu )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEigEstimate"))
    const Int n = d.Height();
    DistMatrix<double,STAR,STAR> d_STAR_STAR( d.Grid() );
    DistMatrix<double,STAR,STAR> dSub_STAR_STAR( d.Grid() );
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );
    vector<double> dVector(n), dSubVector(n), wVector(n);
    MemCopy( dVector.data(), d_STAR_STAR.Buffer(), n );
    MemCopy( dSubVector.data(), dSub_STAR_STAR.Buffer(), n-1 );
    auto estimate = herm_tridiag_eig::EigEstimate
    ( int(n), dVector.data(), dSubVector.data(), wVector.data(), wColComm,
      vl, vu );
    return estimate.numGlobalEigenvalues;
}

// Z is assumed to be sufficiently large and properly aligned
template<typename Real>
void HermitianTridiagEigPostEstimate
( const AbstractDistMatrix<Real>& d,    const AbstractDistMatrix<Real>& dSub,
        AbstractDistMatrix<Real>& wPre,       AbstractDistMatrix<Real>& ZPre,
  SortType sort, Real vl, Real vu )
{
    DEBUG_ONLY(CallStackEntry cse("HermitianTridiagEigPostEstimate"))

    ProxyCtrl wCtrl, ZCtrl;
    wCtrl.colConstrain = true;
    wCtrl.colAlign = 0;
    ZCtrl.rowConstrain = true;
    ZCtrl.rowAlign = 0;
    auto wPtr = WriteProxy<Real,VR,STAR>( &wPre, wCtrl ); 
    auto& w = *wPtr;
    auto ZPtr = WriteProxy<double,STAR,VR>( &ZPre, ZCtrl ); 
    auto& Z = *ZPtr;

    const Int n = d.Height();
    DistMatrix<double,STAR,STAR> d_STAR_STAR( d.Grid() ),
                                 dSub_STAR_STAR( d.Grid() );
    Copy( d, d_STAR_STAR );
    dSub_STAR_STAR.Resize( n-1, 1, n );
    Copy( dSub, dSub_STAR_STAR );

    vector<double> wVector(n);
    auto info = herm_tridiag_eig::Eig
    ( int(n), d_STAR_STAR.Buffer(), dSub_STAR_STAR.Buffer(), wVector.data(), 
      Z.Buffer(), Z.LDim(), w.ColComm(), vl, vu );
    const Int k = info.numGlobalEigenvalues;

    w.Resize( k, 1 );
    for( Int iLoc=0; iLoc<w.LocalHeight(); ++iLoc )
        w.SetLocal( iLoc, 0, Real(wVector[iLoc]) );

    // Shrink Z
    Z.Resize( n, k );

    herm_eig::Sort( w, Z, sort );
}

#define PROTO(F) \
  template void herm_eig::Sort \
  ( Matrix<Base<F>>& w, Matrix<F>& Z, SortType sort ); \
  template void herm_eig::Sort \
  ( AbstractDistMatrix<Base<F>>& w, AbstractDistMatrix<F>& Z, SortType sort ); \
  template void HermitianTridiagEig \
  ( Matrix<Base<F>>& d, Matrix<F>& dSub, Matrix<Base<F>>& w, SortType sort, \
    const HermitianEigSubset<Base<F>>& subset ); \
  template void HermitianTridiagEig \
  ( Matrix<Base<F>>& d, Matrix<F>& dSub, Matrix<Base<F>>& w, Matrix<F>& Z, \
    SortType sort, const HermitianEigSubset<Base<F>>& subset ); \
  template void HermitianTridiagEig \
  ( const AbstractDistMatrix<Base<F>>& d, const AbstractDistMatrix<F>& dSub, \
          AbstractDistMatrix<Base<F>>& w, \
    SortType sort, const HermitianEigSubset<Base<F>>& subset ); \
  template void HermitianTridiagEig \
  ( const AbstractDistMatrix<Base<F>>& d, const AbstractDistMatrix<F>& dSub, \
          AbstractDistMatrix<Base<F>>& w,       AbstractDistMatrix<F>& Z, \
    SortType sort, const HermitianEigSubset<Base<F>>& subset );

#define PROTO_REAL(Real) \
  PROTO(Real) \
  template Int HermitianTridiagEigEstimate \
  ( const AbstractDistMatrix<Real>& d, \
    const AbstractDistMatrix<Real>& dSub, \
          mpi::Comm wColComm, Real vl, Real vu ); \
  template void HermitianTridiagEigPostEstimate \
  ( const AbstractDistMatrix<Real>& d, const AbstractDistMatrix<Real>& dSub, \
          AbstractDistMatrix<Real>& w,       AbstractDistMatrix<Real>& Z, \
    SortType sort, Real vl, Real vu );

#define PROTO_FLOAT \
  PROTO_REAL(float) \
  template void herm_eig::Sort \
  ( AbstractDistMatrix<float>& w, AbstractDistMatrix<double>& Z, \
    SortType sort );

#define PROTO_COMPLEX_FLOAT \
  PROTO(Complex<float>) \
  template void herm_eig::Sort \
  ( AbstractDistMatrix<float>& w, AbstractDistMatrix<Complex<double>>& Z, \
    SortType sort );

#define EL_NO_INT_PROTO
#include "El/macros/Instantiate.h"

} // namespace El
