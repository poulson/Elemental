/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_SCHUR_QR_HPP
#define EL_SCHUR_QR_HPP

namespace El {
namespace schur {

template<typename F>
inline void
QR( Matrix<F>& A, Matrix<Complex<Base<F>>>& w, bool fullTriangle )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
    const Int n = A.Height();
    w.Resize( n, 1 );
    lapack::Schur( n, A.Buffer(), A.LDim(), w.Buffer(), fullTriangle );
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        DEBUG_ONLY(CheckRealSchur(A))
    }
}

template<typename F>
inline void
QR
( Matrix<F>& A, Matrix<Complex<Base<F>>>& w, Matrix<F>& Q, 
  bool fullTriangle )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
    const Int n = A.Height();
    Q.Resize( n, n );
    w.Resize( n, 1 );
    lapack::Schur
    ( n, A.Buffer(), A.LDim(), w.Buffer(), Q.Buffer(), Q.LDim(), fullTriangle );
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        DEBUG_ONLY(CheckRealSchur(A))
    }
}

template<typename F>
inline void
QR
( BlockDistMatrix<F>& A, AbstractDistMatrix<Complex<Base<F>>>& w,
  bool fullTriangle, const HessQRCtrl& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
#ifdef EL_HAVE_SCALAPACK
    const Int n = A.Height();
    const int bhandle = blacs::Handle( A.DistComm().comm );
    const int context =
        blacs::GridInit
        ( bhandle, A.Grid().Order()==COLUMN_MAJOR, 
          A.ColStride(), A.RowStride() );
    if( A.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( A.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( A.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( A.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");
    auto desca = FillDesc( A, context );

    // Reduce the matrix to upper-Hessenberg form in an elemental form
    DistMatrix<F> AElem( A );
    DistMatrix<F,STAR,STAR> t( A.Grid() );
    Hessenberg( UPPER, AElem, t );
    MakeTrapezoidal( UPPER, AElem, -1 );
    A = AElem;
    if( A.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( A.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( A.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( A.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");

    // Run the QR algorithm in block form
    DistMatrix<Complex<Base<F>>,STAR,STAR> w_STAR_STAR( n, 1, A.Grid() );
    scalapack::HessenbergSchur
    ( n, A.Buffer(), desca.data(), w_STAR_STAR.Buffer(), fullTriangle, 
      ctrl.distAED );
    Copy( w_STAR_STAR, w );

    blacs::FreeGrid( context );
    blacs::FreeHandle( bhandle );
    blacs::Exit();
#else
    LogicError("Distributed schur::QR currently requires ScaLAPACK support");
#endif
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        // NOTE: This routine is not yet implemented
        //DEBUG_ONLY(CheckRealSchur(A))
    }
}

template<typename F>
inline void
QR
( BlockDistMatrix<F>& A, AbstractDistMatrix<Complex<Base<F>>>& w,
  BlockDistMatrix<F>& Q, bool fullTriangle, const HessQRCtrl& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
#ifdef EL_HAVE_SCALAPACK
    const Int n = A.Height();
    const int bhandle = blacs::Handle( A.DistComm().comm );
    const int context =
        blacs::GridInit
        ( bhandle, A.Grid().Order()==COLUMN_MAJOR, 
          A.ColStride(), A.RowStride() );
    Q.AlignWith( A );
    Q.Resize( n, n, A.LDim() );
    if( A.ColStride() != blacs::GridHeight(context) || 
        Q.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( A.RowStride() != blacs::GridWidth(context) || 
        Q.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( A.ColRank() != blacs::GridRow(context) ||
        Q.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( A.RowRank() != blacs::GridCol(context) || 
        Q.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");
    auto desca = FillDesc( A, context );
    auto descq = FillDesc( Q, context );

    // Reduce A to upper-Hessenberg form in an element-wise distribution
    // and form the explicit reflector matrix
    DistMatrix<F> AElem( A ), QElem( A.Grid() );
    DistMatrix<F,STAR,STAR> t( A.Grid() );
    Hessenberg( UPPER, AElem, t );
    // There is not yet a 'form Q'
    Identity( QElem, n, n ); 
    hessenberg::ApplyQ( LEFT, UPPER, NORMAL, AElem, t, QElem );
    MakeTrapezoidal( UPPER, AElem, -1 );
    A = AElem;
    Q = QElem;
    if( A.ColStride() != blacs::GridHeight(context) || 
        Q.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( A.RowStride() != blacs::GridWidth(context) || 
        Q.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( A.ColRank() != blacs::GridRow(context) ||
        Q.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( A.RowRank() != blacs::GridCol(context) || 
        Q.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");
    
    // Compute the Schur decomposition in block form, multiplying the 
    // accumulated Householder reflectors from the right
    DistMatrix<Complex<Base<F>>,STAR,STAR> w_STAR_STAR( n, 1, A.Grid() );
    const bool multiplyQ = true;
    scalapack::HessenbergSchur
    ( n, A.Buffer(), desca.data(), w_STAR_STAR.Buffer(), 
      Q.Buffer(), descq.data(), fullTriangle, multiplyQ, ctrl.distAED );
    Copy( w_STAR_STAR, w );

    blacs::FreeGrid( context );
    blacs::FreeHandle( bhandle );
    blacs::Exit();
#else
    LogicError("Distributed schur::QR currently requires ScaLAPACK support");
#endif
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        // NOTE: This routine is not yet implemented
        //DEBUG_ONLY(CheckRealSchur(A))
    }
}

template<typename F>
inline void
QR
( AbstractDistMatrix<F>& APre, AbstractDistMatrix<Complex<Base<F>>>& w, 
  bool fullTriangle, const HessQRCtrl& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
    auto APtr = ReadWriteProxy<F,MC,MR>( &APre );
    auto& A = *APtr;
#ifdef EL_HAVE_SCALAPACK
    // Reduce the matrix to upper-Hessenberg form in an elemental form
    DistMatrix<F,STAR,STAR> t( A.Grid() );
    Hessenberg( UPPER, A, t );
    MakeTrapezoidal( UPPER, A, -1 );

    // Run the QR algorithm in block form
    // TODO: Create schur::HessenbergQR
    const Int n = A.Height(); 
    const Int mb = ctrl.blockHeight;
    const Int nb = ctrl.blockWidth;
    BlockDistMatrix<F> ABlock( n, n, A.Grid(), mb, nb );
    ABlock = A;
    const int bhandle = blacs::Handle( ABlock.DistComm().comm );
    const int context =
        blacs::GridInit
        ( bhandle, ABlock.Grid().Order()==COLUMN_MAJOR,
          ABlock.ColStride(), ABlock.RowStride() );
    if( ABlock.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( ABlock.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( ABlock.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( ABlock.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");
    blacs::Desc desca = FillDesc( ABlock, context );
    DistMatrix<Complex<Base<F>>,STAR,STAR> w_STAR_STAR( n, 1, A.Grid() );
    scalapack::HessenbergSchur
    ( n, ABlock.Buffer(), desca.data(), w_STAR_STAR.Buffer(), 
      fullTriangle, ctrl.distAED );
    A = ABlock;
    Copy( w_STAR_STAR, w );

    blacs::FreeGrid( context );
    blacs::FreeHandle( bhandle );
    blacs::Exit();
#else
    LogicError("Distributed schur::QR currently requires ScaLAPACK support");
#endif
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        DEBUG_ONLY(CheckRealSchur(A))
    }
}

template<typename F>
inline void
QR
( AbstractDistMatrix<F>& APre, AbstractDistMatrix<Complex<Base<F>>>& w, 
  AbstractDistMatrix<F>& QPre, bool fullTriangle, const HessQRCtrl& ctrl )
{
    DEBUG_ONLY(CallStackEntry cse("schur::QR"))
    auto APtr = ReadWriteProxy<F,MC,MR>( &APre ); auto& A = *APtr;
    auto QPtr = WriteProxy<F,MC,MR>( &QPre );     auto& Q = *QPtr;
#ifdef EL_HAVE_SCALAPACK
    const Int n = A.Height();
    // Reduce A to upper-Hessenberg form in an element-wise distribution
    // and form the explicit reflector matrix
    DistMatrix<F,STAR,STAR> t( A.Grid() );
    Hessenberg( UPPER, A, t );
    // There is not yet a 'form Q'
    Identity( Q, n, n ); 
    hessenberg::ApplyQ( LEFT, UPPER, NORMAL, A, t, Q );
    MakeTrapezoidal( UPPER, A, -1 );

    // Run the Hessenberg QR algorithm in block form
    const Int mb = ctrl.blockHeight;
    const Int nb = ctrl.blockWidth;
    BlockDistMatrix<F> ABlock( n, n, A.Grid(), mb, nb ), 
                       QBlock( n, n, A.Grid(), mb, nb );
    ABlock = A;
    QBlock = Q;
    const int bhandle = blacs::Handle( ABlock.DistComm().comm );
    const int context =
        blacs::GridInit
        ( bhandle, ABlock.Grid().Order()==COLUMN_MAJOR, 
          ABlock.ColStride(), ABlock.RowStride() );
    if( ABlock.ColStride() != blacs::GridHeight(context) || 
        QBlock.ColStride() != blacs::GridHeight(context) )
        LogicError("Grid height did not match BLACS");
    if( ABlock.RowStride() != blacs::GridWidth(context) || 
        QBlock.RowStride() != blacs::GridWidth(context) )
        LogicError("Grid width did not match BLACS");
    if( ABlock.ColRank() != blacs::GridRow(context) ||
        QBlock.ColRank() != blacs::GridRow(context) )
        LogicError("Grid row did not match BLACS");
    if( ABlock.RowRank() != blacs::GridCol(context) || 
        QBlock.RowRank() != blacs::GridCol(context) )
        LogicError("Grid col did not match BLACS");
    auto desca = FillDesc( ABlock, context );
    auto descq = FillDesc( QBlock, context );

    // Compute the Schur decomposition in block form, multiplying the 
    // accumulated Householder reflectors from the right
    DistMatrix<Complex<Base<F>>,STAR,STAR> w_STAR_STAR( n, 1, A.Grid() );
    const bool multiplyQ = true;
    scalapack::HessenbergSchur
    ( n, ABlock.Buffer(), desca.data(), w_STAR_STAR.Buffer(), 
      QBlock.Buffer(), descq.data(), fullTriangle, multiplyQ, ctrl.distAED );
    A = ABlock;
    Q = QBlock;
    Copy( w_STAR_STAR, w );

    blacs::FreeGrid( context );
    blacs::FreeHandle( bhandle );
    blacs::Exit();
#else
    LogicError("Distributed schur::QR currently requires ScaLAPACK support");
#endif
    if( IsComplex<F>::val )
        MakeTrapezoidal( UPPER, A );
    else
    {
        MakeTrapezoidal( UPPER, A, -1 );
        DEBUG_ONLY(CheckRealSchur(A))
    }
}

} // namespace schur
} // namespace El

#endif // ifndef EL_SCHUR_QR_HPP
