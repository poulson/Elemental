/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

#ifndef EL_IMPORTS_PMRRR_HPP
#define EL_IMPORTS_PMRRR_HPP

#include "El.hpp"

#include <pmrrr/pmrrr.hpp>

namespace El {
namespace herm_tridiag_eig {

struct Estimate {
    int numLocalEigenvalues;
    int numGlobalEigenvalues;
};

struct Info {
    int numLocalEigenvalues;
    int numGlobalEigenvalues;

    int firstLocalEigenvalue;
};

// Return upper bounds on the number of (local) eigenvalues in the given range,
// (lowerBound,upperBound]
template<typename FloatingType>
Estimate EigEstimate
( int n, FloatingType* d, FloatingType* e, FloatingType* w, mpi::Comm comm, 
  FloatingType lowerBound, FloatingType upperBound )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::EigEstimate"))
    Estimate estimate;
    char jobz='C';
    char range='V';
    int il, iu;
    int highAccuracy=0;
    int nz, offset;
    int ldz=1;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm.comm, &nz, &offset, w, static_cast<FloatingType*>(nullptr), &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    estimate.numLocalEigenvalues = nz;
    estimate.numGlobalEigenvalues = mpi::AllReduce( nz, comm );
    return estimate;
}

// Compute all of the eigenvalues
template<typename FloatingType>
Info Eig( int n, FloatingType* d, FloatingType* e, FloatingType* w, mpi::Comm comm )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    char jobz='N';
    char range='A';
    FloatingType vl, vu;
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &highAccuracy, comm.comm,
      &nz, &offset, w, static_cast<FloatingType*>(nullptr), &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=n;
    return info;
}

// Compute all of the eigenpairs
template<typename FloatingType>
Info Eig
( int n, FloatingType* d, FloatingType* e, FloatingType* w, FloatingType* Z, int ldz, mpi::Comm comm )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    char jobz='V';
    char range='A';
    FloatingType vl, vu;
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &highAccuracy, comm.comm,
      &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=n;
    return info;
}

// Compute all of the eigenvalues in (lowerBound,upperBound]
template<typename FloatingType>
Info Eig
( int n, FloatingType* d, FloatingType* e, FloatingType* w, mpi::Comm comm, 
  FloatingType lowerBound, FloatingType upperBound )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    char jobz='N';
    char range='V';
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm.comm, &nz, &offset, w, static_cast<FloatingType*>(nullptr), &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    mpi::AllReduce( &nz, &info.numGlobalEigenvalues, 1, mpi::SUM, comm );
    return info;
}

// Compute all of the eigenpairs with eigenvalues in (lowerBound,upperBound]
template<typename FloatingType>
Info Eig
( int n, FloatingType* d, FloatingType* e, FloatingType* w, FloatingType* Z, int ldz, mpi::Comm comm, 
  FloatingType lowerBound, FloatingType upperBound )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    char jobz='V';
    char range='V';
    int il, iu;
    int highAccuracy=0; 
    int nz, offset;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &lowerBound, &upperBound, &il, &iu, 
      &highAccuracy, comm.comm, &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    mpi::AllReduce( &nz, &info.numGlobalEigenvalues, 1, mpi::SUM, comm );
    return info;
}

// Compute all of the eigenvalues with indices in [lowerBound,upperBound]
template<typename FloatingType>
Info Eig
( int n, FloatingType* d, FloatingType* e, FloatingType* w, mpi::Comm comm, 
  int lowerBound, int upperBound )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    ++lowerBound;
    ++upperBound;
    char jobz='N';
    char range='I';
    FloatingType vl, vu;
    int highAccuracy=0; 
    int nz, offset;
    int ldz=1;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &vl, &vu, &lowerBound, &upperBound, 
      &highAccuracy, comm.comm, &nz, &offset, w, static_cast<FloatingType*>(nullptr), &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=(upperBound-lowerBound)+1;
    return info;
}

// Compute all of the eigenpairs with eigenvalues indices in 
// [lowerBound,upperBound]
template<typename FloatingType>
Info Eig
( int n, FloatingType* d, FloatingType* e, FloatingType* w, FloatingType* Z, int ldz, mpi::Comm comm, 
  int lowerBound, int upperBound )
{
    DEBUG_ONLY(CSE cse("herm_tridiag_eig::Eig"))
    Info info;
    ++lowerBound;
    ++upperBound;
    char jobz='V';
    char range='I';
    FloatingType vl, vu;
    int highAccuracy=0; 
    int nz, offset;
    vector<int> ZSupport(2*n);
    int retval = pmrrr::pmrrr
    ( &jobz, &range, &n, d, e, &vl, &vu, &lowerBound, &upperBound, 
      &highAccuracy, comm.comm, &nz, &offset, w, Z, &ldz, ZSupport.data() );
    if( retval != 0 )
        RuntimeError("pmrrr returned ",retval);

    info.numLocalEigenvalues=nz;
    info.firstLocalEigenvalue=offset;
    info.numGlobalEigenvalues=(upperBound-lowerBound)+1;
    return info;
}

} // namespace herm_tridiag_eig
} // namespace El

#endif
