/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
#include "El.h"
using namespace El;

extern "C" {

ElError ElQRCtrlFillDefault_s( ElQRCtrl_s* ctrl )
{
    ctrl->colPiv = false;
    ctrl->boundRank = false;
    ctrl->maxRank = 0;
    ctrl->adaptive = false;
    ctrl->tol = 0;
    ctrl->alwaysRecomputeNorms = false;
    return EL_SUCCESS;
}

ElError ElQRCtrlFillDefault_d( ElQRCtrl_d* ctrl )
{
    ctrl->colPiv = false;
    ctrl->boundRank = false;
    ctrl->maxRank = 0;
    ctrl->adaptive = false;
    ctrl->tol = 0;
    ctrl->alwaysRecomputeNorms = false;
    return EL_SUCCESS;
}

#define C_PROTO_FIELD(SIG,SIGBASE,F) \
  /* Cholesky
     ======== */ \
  /* Cholesky (no pivoting) */ \
  ElError ElCholesky_ ## SIG \
  ( ElUpperOrLower uplo, ElMatrix_ ## SIG A ) \
  { EL_TRY( Cholesky( CReflect(uplo), *CReflect(A) ) ) } \
  ElError ElCholeskyDist_ ## SIG \
  ( ElUpperOrLower uplo, ElDistMatrix_ ## SIG A ) \
  { EL_TRY( Cholesky( CReflect(uplo), *CReflect(A) ) ) } \
  /* Reverse Cholesky (no pivoting) */ \
  ElError ElReverseCholesky_ ## SIG \
  ( ElUpperOrLower uplo, ElMatrix_ ## SIG A ) \
  { EL_TRY( ReverseCholesky( CReflect(uplo), *CReflect(A) ) ) } \
  ElError ElReverseCholeskyDist_ ## SIG \
  ( ElUpperOrLower uplo, ElDistMatrix_ ## SIG A ) \
  { EL_TRY( ReverseCholesky( CReflect(uplo), *CReflect(A) ) ) } \
  /* Cholesky (full pivoting) */ \
  ElError ElCholeskyPiv_ ## SIG \
  ( ElUpperOrLower uplo, ElMatrix_ ## SIG A, ElMatrix_i p ) \
  { EL_TRY( Cholesky( CReflect(uplo), *CReflect(A), *CReflect(p) ) ) } \
  ElError ElCholeskyPivDist_ ## SIG \
  ( ElUpperOrLower uplo, ElDistMatrix_ ## SIG A, ElDistMatrix_i p ) \
  { EL_TRY( Cholesky( CReflect(uplo), *CReflect(A), *CReflect(p) ) ) } \
  /* Cholesky low-rank modification */ \
  ElError ElCholeskyMod_ ## SIG \
  ( ElUpperOrLower uplo, ElMatrix_ ## SIG T, \
    Base<F> alpha, ElMatrix_ ## SIG V ) \
  { EL_TRY( \
      CholeskyMod( CReflect(uplo), *CReflect(T), alpha, *CReflect(V) ) ) } \
  ElError ElCholeskyModDist_ ## SIG \
  ( ElUpperOrLower uplo, ElDistMatrix_ ## SIG T, \
    Base<F> alpha, ElDistMatrix_ ## SIG V ) \
  { EL_TRY( \
      CholeskyMod( CReflect(uplo), *CReflect(T), alpha, *CReflect(V) ) ) } \
  /* Hermitian Positive Semi-Definite Cholesky */ \
  ElError ElHPSDCholesky_ ## SIG \
  ( ElUpperOrLower uplo, ElMatrix_ ## SIG A ) \
  { EL_TRY( HPSDCholesky( CReflect(uplo), *CReflect(A) ) ) } \
  ElError ElHPSDCholeskyDist_ ## SIG \
  ( ElUpperOrLower uplo, ElDistMatrix_ ## SIG A ) \
  { EL_TRY( HPSDCholesky( CReflect(uplo), *CReflect(A) ) ) } \
  /* Solve after a Cholesky factorization (without pivoting) */ \
  ElError ElSolveAfterCholesky_ ## SIG \
  ( ElUpperOrLower uplo, ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( \
      cholesky::SolveAfter( \
        CReflect(uplo), CReflect(orientation), \
        *CReflect(A), *CReflect(B) ) ) } \
  ElError ElSolveAfterCholeskyDist_ ## SIG \
  ( ElUpperOrLower uplo, ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( \
      cholesky::SolveAfter( \
        CReflect(uplo), CReflect(orientation), \
        *CReflect(A), *CReflect(B) ) ) } \
  /* Solve after a Cholesky factorization (full pivoting) */ \
  ElError ElSolveAfterCholeskyPiv_ ## SIG \
  ( ElUpperOrLower uplo, ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_i p, ElMatrix_ ## SIG B ) \
  { EL_TRY( \
      cholesky::SolveAfter( \
        CReflect(uplo), CReflect(orientation), \
        *CReflect(A), *CReflect(p), *CReflect(B) ) ) } \
  ElError ElSolveAfterCholeskyPivDist_ ## SIG \
  ( ElUpperOrLower uplo, ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_i p, \
    ElDistMatrix_ ## SIG B ) \
  { EL_TRY( \
      cholesky::SolveAfter( \
        CReflect(uplo), CReflect(orientation), \
        *CReflect(A), *CReflect(p), *CReflect(B) ) ) } \
  /* Generalized QR
     ============== */ \
  ElError ElGQR_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG tA, ElMatrix_ ## SIGBASE dA, \
    ElMatrix_ ## SIG B, ElMatrix_ ## SIG tB, ElMatrix_ ## SIGBASE dB ) \
  { EL_TRY( GQR( *CReflect(A), *CReflect(tA), *CReflect(dA), \
                 *CReflect(B), *CReflect(tB), *CReflect(dB) ) ) } \
  ElError ElGQRDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, \
    ElDistMatrix_ ## SIG tA, ElDistMatrix_ ## SIGBASE dA, \
    ElDistMatrix_ ## SIG B, \
    ElDistMatrix_ ## SIG tB, ElDistMatrix_ ## SIGBASE dB ) \
  { EL_TRY( GQR( *CReflect(A), *CReflect(tA), *CReflect(dA), \
                 *CReflect(B), *CReflect(tB), *CReflect(dB) ) ) } \
  ElError ElGQRExplicitTriang_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( gqr::ExplicitTriang( *CReflect(A), *CReflect(B) ) ) } \
  ElError ElGQRExplicitTriangDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( gqr::ExplicitTriang( *CReflect(A), *CReflect(B) ) ) } \
  /* Generalized RQ
     ============== */ \
  ElError ElGRQ_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG tA, ElMatrix_ ## SIGBASE dA, \
    ElMatrix_ ## SIG B, ElMatrix_ ## SIG tB, ElMatrix_ ## SIGBASE dB ) \
  { EL_TRY( GRQ( *CReflect(A), *CReflect(tA), *CReflect(dA), \
                 *CReflect(B), *CReflect(tB), *CReflect(dB) ) ) } \
  ElError ElGRQDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, \
    ElDistMatrix_ ## SIG tA, ElDistMatrix_ ## SIGBASE dA, \
    ElDistMatrix_ ## SIG B, \
    ElDistMatrix_ ## SIG tB, ElDistMatrix_ ## SIGBASE dB ) \
  { EL_TRY( GRQ( *CReflect(A), *CReflect(tA), *CReflect(dA), \
                 *CReflect(B), *CReflect(tB), *CReflect(dB) ) ) } \
  ElError ElGRQExplicitTriang_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( grq::ExplicitTriang( *CReflect(A), *CReflect(B) ) ) } \
  ElError ElGRQExplicitTriangDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( grq::ExplicitTriang( *CReflect(A), *CReflect(B) ) ) } \
  /* Interpolative Decomposition 
     =========================== */ \
  ElError ElID_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_i p, ElMatrix_ ## SIG Z, \
    ElQRCtrl_ ## SIGBASE ctrl, bool canOverwrite ) \
  { EL_TRY( \
      ID( *CReflect(A), *CReflect(p), *CReflect(Z), \
          CReflect(ctrl), canOverwrite ) ) } \
  ElError ElIDDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_i p, ElDistMatrix_ ## SIG Z, \
    ElQRCtrl_ ## SIGBASE ctrl, bool canOverwrite ) \
  { EL_TRY( \
      ID( *CReflect(A), *CReflect(p), *CReflect(Z), \
          CReflect(ctrl), canOverwrite ) ) } \
  /* LDL factorization
     ================= */ \
  /* Return the inertia given diagonal and subdiagonal from an LDL^H fact */ \
  ElError ElInertiaAfterLDL_ ## SIG \
  ( ElConstMatrix_ ## SIGBASE d, ElConstMatrix_ ## SIG dSub, \
    ElInertiaType* inertia ) \
  { EL_TRY( *inertia = \
      CReflect(ldl::Inertia(*CReflect(d),*CReflect(dSub))) ) } \
  ElError ElInertiaAfterLDLDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIGBASE d, ElConstDistMatrix_ ## SIG dSub, \
    ElInertiaType* inertia ) \
  { EL_TRY( *inertia = CReflect(ldl::Inertia( \
      *CReflect(d), *CReflect(dSub))) ) } \
  /* LQ factorization 
     ================ */ \
  /* Return the packed LQ factorization */ \
  ElError ElLQ_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG t, ElMatrix_ ## SIGBASE d ) \
  { EL_TRY( LQ( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  ElError ElLQDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG t, \
    ElDistMatrix_ ## SIGBASE d ) \
  { EL_TRY( LQ( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  /* Explicitly return both factors */ \
  ElError ElLQExplicit_ ## SIG \
  ( ElMatrix_ ## SIG L, ElMatrix_ ## SIG A ) \
  { EL_TRY( lq::Explicit( *CReflect(L), *CReflect(A) ) ) } \
  ElError ElLQExplicitDist_ ## SIG \
  ( ElDistMatrix_ ## SIG L, ElDistMatrix_ ## SIG A ) \
  { EL_TRY( lq::Explicit( *CReflect(L), *CReflect(A) ) ) } \
  /* Only return the triangular factor */ \
  ElError ElLQExplicitTriang_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( lq::ExplicitTriang( *CReflect(A) ) ) } \
  ElError ElLQExplicitTriangDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( lq::ExplicitTriang( *CReflect(A) ) ) } \
  /* Only return the unitary factor */ \
  ElError ElLQExplicitUnitary_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( lq::ExplicitUnitary( *CReflect(A) ) ) } \
  ElError ElLQExplicitUnitaryDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( lq::ExplicitUnitary( *CReflect(A) ) ) } \
  /* Apply Q after an LQ factorization */ \
  ElError ElApplyQAfterLQ_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG t, \
    ElConstMatrix_ ## SIGBASE d, ElMatrix_ ## SIG B ) \
  { EL_TRY( lq::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B) ) ) } \
  ElError ElApplyQAfterLQDist_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG t, \
    ElConstDistMatrix_ ## SIGBASE d, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( lq::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(B) ) ) } \
  /* Solve against vectors after LQ factorization */ \
  ElError ElSolveAfterLQ_ ## SIG \
  ( ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG t, \
    ElConstMatrix_ ## SIGBASE d, ElConstMatrix_ ## SIG B, ElMatrix_ ## SIG X ) \
  { EL_TRY( lq::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  ElError ElSolveAfterLQDist_ ## SIG \
  ( ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG t, \
    ElConstDistMatrix_ ## SIGBASE d, ElConstDistMatrix_ ## SIG B, \
    ElDistMatrix_ ## SIG X ) \
  { EL_TRY( lq::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  /* LU factorization 
     ================ */ \
  /* LU without pivoting */ \
  ElError ElLU_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( LU( *CReflect(A) ) ) } \
  ElError ElLUDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( LU( *CReflect(A) ) ) } \
  /* LU with partial pivoting */ \
  ElError ElLUPartialPiv_ ## SIG ( ElMatrix_ ## SIG A, ElMatrix_i p ) \
  { EL_TRY( LU( *CReflect(A), *CReflect(p) ) ) } \
  ElError ElLUPartialPivDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_i p ) \
  { EL_TRY( LU( *CReflect(A), *CReflect(p) ) ) } \
  /* LU with full pivoting */ \
  ElError ElLUFullPiv_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_i p, ElMatrix_i q ) \
  { EL_TRY( LU( *CReflect(A), *CReflect(p), *CReflect(q) ) ) } \
  ElError ElLUFullPivDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_i p, ElDistMatrix_i q ) \
  { EL_TRY( LU( *CReflect(A), *CReflect(p), *CReflect(q) ) ) } \
  /* Solve against vectors after LU with no pivoting */ \
  ElError ElSolveAfterLU_ ## SIG \
  ( ElOrientation orientation, ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(B) ) ) } \
  ElError ElSolveAfterLUDist_ ## SIG \
  ( ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(B) ) ) } \
  /* Solve against vectors after LU with partial pivoting */ \
  ElError ElSolveAfterLUPartialPiv_ ## SIG \
  ( ElOrientation orientation, ElConstMatrix_ ## SIG A, ElConstMatrix_i p, \
    ElMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), \
      *CReflect(A), *CReflect(p), *CReflect(B) ) ) } \
  ElError ElSolveAfterLUPartialPivDist_ ## SIG \
  ( ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_i p, \
    ElDistMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(p), *CReflect(B) ) ) } \
  /* Solve against vectors after LU with full pivoting */ \
  ElError ElSolveAfterLUFullPiv_ ## SIG \
  ( ElOrientation orientation, ElConstMatrix_ ## SIG A, \
    ElConstMatrix_i p, ElConstMatrix_i q, ElMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), *CReflect(A), \
      *CReflect(p), *CReflect(q), *CReflect(B) ) ) } \
  ElError ElSolveAfterLUFullPivDist_ ## SIG \
  ( ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_i p, ElConstDistMatrix_i q, \
    ElDistMatrix_ ## SIG B ) \
  { EL_TRY( lu::SolveAfter( \
      CReflect(orientation), *CReflect(A), \
      *CReflect(p), *CReflect(q), *CReflect(B) ) ) } \
  /* QR factorization 
     ================ */ \
  /* Return the packed QR factorization (with no pivoting) */ \
  ElError ElQR_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG t, ElMatrix_ ## SIGBASE d ) \
  { EL_TRY( QR( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  ElError ElQRDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, \
    ElDistMatrix_ ## SIG t, ElDistMatrix_ ## SIGBASE d ) \
  { EL_TRY( QR( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  /* Return the packed QR factorization (with column pivoting) */ \
  ElError ElQRColPiv_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG t, ElMatrix_ ## SIGBASE d, \
    ElMatrix_i p ) \
  { EL_TRY( QR( \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(p) ) ) } \
  ElError ElQRColPivDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, \
    ElDistMatrix_ ## SIG t, ElDistMatrix_ ## SIGBASE d, ElDistMatrix_i p ) \
  { EL_TRY( QR( *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(p) ) ) } \
  /* Return the packed QR factorization (with column pivoting, eXpert) */ \
  ElError ElQRColPivX_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG t, ElMatrix_ ## SIGBASE d, \
    ElMatrix_i p, ElQRCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( QR( \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(p),\
      CReflect(ctrl) ) ) } \
  ElError ElQRColPivXDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, \
    ElDistMatrix_ ## SIG t, ElDistMatrix_ ## SIGBASE d, ElDistMatrix_i p, \
    ElQRCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( QR( \
      *CReflect(A), *CReflect(t), *CReflect(d), \
      *CReflect(p), CReflect(ctrl) ) ) } \
  /* Explicitly return Q and R (with no pivoting) */ \
  ElError ElQRExplicit_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG R ) \
  { EL_TRY( qr::Explicit( *CReflect(A), *CReflect(R) ) ) } \
  ElError ElQRExplicitDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG R ) \
  { EL_TRY( qr::Explicit( *CReflect(A), *CReflect(R) ) ) } \
  /* Explicitly return Q, R, and P (with column pivoting) */ \
  ElError ElQRColPivExplicit_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG R, ElMatrix_i P ) \
  { EL_TRY( qr::Explicit( *CReflect(A), *CReflect(R), *CReflect(P) ) ) } \
  ElError ElQRColPivExplicitDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG R, ElDistMatrix_i P ) \
  { EL_TRY( qr::Explicit( *CReflect(A), *CReflect(R), *CReflect(P) ) ) } \
  /* Return the triangular factor from QR */ \
  ElError ElQRExplicitTriang_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( qr::ExplicitTriang( *CReflect(A) ) ) } \
  ElError ElQRExplicitTriangDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( qr::ExplicitTriang( *CReflect(A) ) ) } \
  /* Return the unitary factor from QR */ \
  ElError ElQRExplicitUnitary_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( qr::ExplicitUnitary( *CReflect(A) ) ) } \
  ElError ElQRExplicitUnitaryDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( qr::ExplicitUnitary( *CReflect(A) ) ) } \
  /* Cholesky-based QR factorization */ \
  ElError ElCholeskyQR_ ## SIG ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG R ) \
  { EL_TRY( qr::Cholesky( *CReflect(A), *CReflect(R) ) ) } \
  ElError ElCholeskyQRDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG R ) \
  { EL_TRY( qr::Cholesky( *CReflect(A), *CReflect(R) ) ) } \
  /* Apply Q after a QR factorization */ \
  ElError ElApplyQAfterQR_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG t, \
    ElConstMatrix_ ## SIGBASE d, ElMatrix_ ## SIG B ) \
  { EL_TRY( qr::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B) ) ) } \
  ElError ElApplyQAfterQRDist_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG t, \
    ElConstDistMatrix_ ## SIGBASE d, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( qr::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(B) ) ) } \
  /* Solve against vectors after QR factorization */ \
  ElError ElSolveAfterQR_ ## SIG \
  ( ElOrientation orientation, ElConstMatrix_ ## SIG A, \
    ElConstMatrix_ ## SIG t, ElConstMatrix_ ## SIGBASE d, \
    ElConstMatrix_ ## SIG B, ElMatrix_ ## SIG X ) \
  { EL_TRY( qr::SolveAfter( \
      CReflect(orientation), *CReflect(A), \
      *CReflect(t), *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  ElError ElSolveAfterQRDist_ ## SIG \
  ( ElOrientation orientation, ElConstDistMatrix_ ## SIG A, \
    ElConstDistMatrix_ ## SIG t, ElConstDistMatrix_ ## SIGBASE d, \
    ElConstDistMatrix_ ## SIG B, ElDistMatrix_ ## SIG X ) \
  { EL_TRY( qr::SolveAfter( \
      CReflect(orientation), *CReflect(A), \
      *CReflect(t), *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  /* RQ factorization 
     ================ */ \
  /* Return the packed RQ factorization */ \
  ElError ElRQ_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG t, ElMatrix_ ## SIGBASE d ) \
  { EL_TRY( RQ( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  ElError ElRQDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG t, \
    ElDistMatrix_ ## SIGBASE d ) \
  { EL_TRY( RQ( *CReflect(A), *CReflect(t), *CReflect(d) ) ) } \
  /* TODO: Explicitly return both factors */ \
  /* Only return the triangular factor */ \
  ElError ElRQExplicitTriang_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( rq::ExplicitTriang( *CReflect(A) ) ) } \
  ElError ElRQExplicitTriangDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( rq::ExplicitTriang( *CReflect(A) ) ) } \
  /* TODO: Only return the unitary factor */ \
  /* Apply Q after an RQ factorization */ \
  ElError ElApplyQAfterRQ_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG t, \
    ElConstMatrix_ ## SIGBASE d, ElMatrix_ ## SIG B ) \
  { EL_TRY( rq::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(B) ) ) } \
  ElError ElApplyQAfterRQDist_ ## SIG \
  ( ElLeftOrRight side, ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG t, \
    ElConstDistMatrix_ ## SIGBASE d, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( rq::ApplyQ( \
      CReflect(side), CReflect(orientation), \
      *CReflect(A), *CReflect(t), *CReflect(d), *CReflect(B) ) ) } \
  /* Solve against vectors after RQ factorization */ \
  ElError ElSolveAfterRQ_ ## SIG \
  ( ElOrientation orientation, \
    ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG t, \
    ElConstMatrix_ ## SIGBASE d, ElConstMatrix_ ## SIG B, ElMatrix_ ## SIG X ) \
  { EL_TRY( rq::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  ElError ElSolveAfterRQDist_ ## SIG \
  ( ElOrientation orientation, \
    ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG t, \
    ElConstDistMatrix_ ## SIGBASE d, ElConstDistMatrix_ ## SIG B, \
    ElDistMatrix_ ## SIG X ) \
  { EL_TRY( rq::SolveAfter( \
      CReflect(orientation), *CReflect(A), *CReflect(t), \
      *CReflect(d), *CReflect(B), *CReflect(X) ) ) } \
  /* Skeleton factorization
     ====================== */ \
  ElError ElSkeleton_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElMatrix_i pR, ElMatrix_i pC, \
    ElMatrix_ ## SIG Z, ElQRCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( Skeleton( *CReflect(A), *CReflect(pR), *CReflect(pC), \
                      *CReflect(Z), CReflect(ctrl) ) ) } \
  ElError ElSkeletonDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElDistMatrix_i pR, ElDistMatrix_i pC, \
    ElDistMatrix_ ## SIG Z, ElQRCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( Skeleton( *CReflect(A), *CReflect(pR), *CReflect(pC), \
                      *CReflect(Z), CReflect(ctrl) ) ) }

#define C_PROTO_REAL(SIG,Real) \
  C_PROTO_FIELD(SIG,SIG,Real) \
  /* LDL factorization 
     ================= */ \
  ElError ElLDLPivotConstant_ ## SIG ( ElLDLPivotType pivotType, Real* gamma ) \
  { EL_TRY( *gamma = LDLPivotConstant<Real>(CReflect(pivotType)) ) } \
  /* Return the packed LDL factorization (without pivoting) */ \
  ElError ElLDL_ ## SIG ( ElMatrix_ ## SIG A ) \
  { EL_TRY( LDL( *CReflect(A), false ) ) } \
  ElError ElLDLDist_ ## SIG ( ElDistMatrix_ ## SIG A ) \
  { EL_TRY( LDL( *CReflect(A), false ) ) } \
  /* Return the packed LDL factorization with pivoting */ \
  ElError ElLDLPiv_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG dSub, ElMatrix_i p ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), false ) ) } \
  ElError ElLDLPivDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG dSub, ElDistMatrix_i p ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), false ) ) } \
  /* Expert versions */ \
  ElError ElLDLPivX_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG dSub, ElMatrix_i p, \
    ElLDLPivotCtrl_ ## SIG ctrl ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), false, \
                 CReflect(ctrl) ) ) } \
  ElError ElLDLPivXDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG dSub, ElDistMatrix_i p, \
    ElLDLPivotCtrl_ ## SIG ctrl ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), false, \
                 CReflect(ctrl) ) ) } \
  /* Multiply vectors after an unpivoted LDL factorization */ \
  ElError ElMultiplyAfterLDL_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( ldl::MultiplyAfter( *CReflect(A), *CReflect(B), false ) ) } \
  ElError ElMultiplyAfterLDLDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( ldl::MultiplyAfter( *CReflect(A), *CReflect(B), false ) ) } \
  /* Multiply vectors after a pivoted LDL factorization */ \
  ElError ElMultiplyAfterLDLPiv_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG dSub, ElConstMatrix_i p, \
    ElMatrix_ ## SIG B ) \
  { EL_TRY( ldl::MultiplyAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      false ) ) } \
  ElError ElMultiplyAfterLDLPivDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG dSub, \
    ElConstDistMatrix_i p, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( ldl::MultiplyAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), false ) ) } \
  /* Solve against vectors after an unpivoted LDL factorization */ \
  ElError ElSolveAfterLDL_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B ) \
  { EL_TRY( ldl::SolveAfter( *CReflect(A), *CReflect(B), false ) ) } \
  ElError ElSolveAfterLDLDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( ldl::SolveAfter( *CReflect(A), *CReflect(B), false ) ) } \
  /* Solve against vectors after a pivoted LDL factorization */ \
  ElError ElSolveAfterLDLPiv_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG dSub, ElConstMatrix_i p, \
    ElMatrix_ ## SIG B ) \
  { EL_TRY( ldl::SolveAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      false ) ) } \
  ElError ElSolveAfterLDLPivDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG dSub, \
    ElConstDistMatrix_i p, ElDistMatrix_ ## SIG B ) \
  { EL_TRY( ldl::SolveAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), false ) ) } \
  /* LU factorization
     ================ */ \
  /* Rank-one LU factorization modification */ \
  ElError ElLUMod_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_i p, \
    ElConstMatrix_ ## SIG u, ElConstMatrix_ ## SIG v, Real tau ) \
  { EL_TRY( LUMod( \
      *CReflect(A), *CReflect(p), \
      *CReflect(u), *CReflect(v), false, tau ) ) } \
  ElError ElLUModDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_i p, \
    ElConstDistMatrix_ ## SIG u, ElConstDistMatrix_ ## SIG v, Real tau ) \
  { EL_TRY( LUMod( \
      *CReflect(A), *CReflect(p), \
      *CReflect(u), *CReflect(v), false, tau ) ) }

#define C_PROTO_COMPLEX(SIG,SIGBASE,F) \
  C_PROTO_FIELD(SIG,SIGBASE,F) \
  /* LDL factorization 
     ================= */ \
  /* Return the packed LDL factorization (without pivoting) */ \
  ElError ElLDL_ ## SIG ( ElMatrix_ ## SIG A, bool conjugate ) \
  { EL_TRY( LDL( *CReflect(A), conjugate ) ) } \
  ElError ElLDLDist_ ## SIG ( ElDistMatrix_ ## SIG A, bool conjugate ) \
  { EL_TRY( LDL( *CReflect(A), conjugate ) ) } \
  /* Return the packed LDL factorization with pivoting */ \
  ElError ElLDLPiv_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG dSub, ElMatrix_i p, bool conjugate ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), conjugate ) ) } \
  ElError ElLDLPivDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG dSub, ElDistMatrix_i p, \
    bool conjugate ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), conjugate ) ) } \
  /* Expert versions */ \
  ElError ElLDLPivX_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_ ## SIG dSub, ElMatrix_i p, bool conjugate, \
    ElLDLPivotCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), conjugate, \
                 CReflect(ctrl) ) ) } \
  ElError ElLDLPivXDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG dSub, ElDistMatrix_i p, \
    bool conjugate, ElLDLPivotCtrl_ ## SIGBASE ctrl ) \
  { EL_TRY( LDL( *CReflect(A), *CReflect(dSub), *CReflect(p), conjugate, \
                 CReflect(ctrl) ) ) } \
  /* Multiply vectors after an unpivoted LDL factorization */ \
  ElError ElMultiplyAfterLDL_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::MultiplyAfter( *CReflect(A), *CReflect(B), conjugate ) ) } \
  ElError ElMultiplyAfterLDLDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::MultiplyAfter( *CReflect(A), *CReflect(B), conjugate ) ) } \
  /* Multiply vectors after a pivoted LDL factorization */ \
  ElError ElMultiplyAfterLDLPiv_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG dSub, ElConstMatrix_i p, \
    ElMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::MultiplyAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      conjugate ) ) } \
  ElError ElMultiplyAfterLDLPivDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG dSub, \
    ElConstDistMatrix_i p, ElDistMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::MultiplyAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      conjugate ) ) } \
  /* Solve against vectors after an unpivoted LDL factorization */ \
  ElError ElSolveAfterLDL_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::SolveAfter( *CReflect(A), *CReflect(B), conjugate ) ) } \
  ElError ElSolveAfterLDLDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElDistMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::SolveAfter( *CReflect(A), *CReflect(B), conjugate ) ) } \
  /* Solve against vectors after a pivoted LDL factorization */ \
  ElError ElSolveAfterLDLPiv_ ## SIG \
  ( ElConstMatrix_ ## SIG A, ElConstMatrix_ ## SIG dSub, ElConstMatrix_i p, \
    ElMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::SolveAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      conjugate ) ) } \
  ElError ElSolveAfterLDLPivDist_ ## SIG \
  ( ElConstDistMatrix_ ## SIG A, ElConstDistMatrix_ ## SIG dSub, \
    ElConstDistMatrix_i p, ElDistMatrix_ ## SIG B, bool conjugate ) \
  { EL_TRY( ldl::SolveAfter( \
      *CReflect(A), *CReflect(dSub), *CReflect(p), *CReflect(B), \
      conjugate ) ) } \
  /* LU factorization
     ================ */ \
  /* Rank-one LU factorization modification */ \
  ElError ElLUMod_ ## SIG \
  ( ElMatrix_ ## SIG A, ElMatrix_i p, \
    ElConstMatrix_ ## SIG u, ElConstMatrix_ ## SIG v, \
    bool conjugate, Base<F> tau ) \
  { EL_TRY( LUMod( \
      *CReflect(A), *CReflect(p), \
      *CReflect(u), *CReflect(v), conjugate, tau ) ) } \
  ElError ElLUModDist_ ## SIG \
  ( ElDistMatrix_ ## SIG A, ElDistMatrix_i p, \
    ElConstDistMatrix_ ## SIG u, ElConstDistMatrix_ ## SIG v, \
    bool conjugate, Base<F> tau ) \
  { EL_TRY( LUMod( \
      *CReflect(A), *CReflect(p), \
      *CReflect(u), *CReflect(v), conjugate, tau ) ) }

#define EL_NO_INT_PROTO
#include "El/macros/CInstantiate.h"

} // extern "C"
