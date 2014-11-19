/*
   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_AXPYINTERFACE2_HPP
#define EL_AXPYINTERFACE2_HPP

#if MPI_VERSION>=3
namespace El {
template<typename T>
class AxpyInterface2
{
public:
    AxpyInterface2();
    ~AxpyInterface2();

    AxpyInterface2(       DistMatrix<T,MC,MR>& Z );
    AxpyInterface2( const DistMatrix<T,MC,MR>& Z );

    void Attach(       DistMatrix<T,MC,MR>& Z );
    void Attach( const DistMatrix<T,MC,MR>& Z );

    // nonblocking update routines
    // requires flush for completion
    void Iput( Matrix<T>& Z, Int i, Int j );
    void Iput( const Matrix<T>& Z, Int i, Int j );

    void Iget(       Matrix<T>& Z, Int i, Int j );

    void Iacc(       Matrix<T>& Z, Int i, Int j );
    void Iacc( const Matrix<T>& Z, Int i, Int j );

    void Flush(       Matrix<T>& Z );
    void Flush( const Matrix<T>& Z );
    
    void Cflush(       Matrix<T>& Z );
    void Cflush( const Matrix<T>& Z );

    // blocking update routines
    void Put( Matrix<T>& Z, Int i, Int j );
    void Put( const Matrix<T>& Z, Int i, Int j );

    void Get(       Matrix<T>& Z, Int i, Int j );

    void Acc(       Matrix<T>& Z, Int i, Int j );
    void Acc( const Matrix<T>& Z, Int i, Int j );

    void Detach();

private:
   
    static const Int 
        DATA_PUT_TAG      =1, 
        DATA_GET_TAG      =2,
        DATA_ACC_TAG   	  =3,
        REQUEST_GET_TAG   =4,
	COORD_ACC_TAG     =5,
	COORD_PUT_TAG     =6;

    // request statuses
    std::vector<std::deque<bool>> 
        sendDataStatuses_, sendCoordStatuses_, 
	recvDataStatuses_, recvCoordStatuses_;
    
    // request handles
    std::vector<std::deque<mpi::Request>> 
        sendDataRequests_, sendCoordRequests_, 
	recvDataRequests_, recvCoordRequests_;
    
    // data
    std::vector<std::deque<std::vector<T>>>
        sendData_, recvData_;
    
    // coords
    std::vector<std::deque<std::array<Int,3>>>
        sendCoord_, recvCoord_;
   
    // TODO need to add const here...
    DistMatrix<T,MC,MR>* GlobalArrayPut_;
    DistMatrix<T,MC,MR>* GlobalArrayGet_;
   
    bool toBeAttachedForPut_, toBeAttachedForGet_, 
	 attached_, detached_;

    // op count window for read increment
    mpi::Window put_win_, acc_win_, getrq_win_;
    long *put_win_base_, *acc_win_base_, 
	 *getrq_win_base_;

    // next index for data and coord
    Int NextIndexData
    ( Int dataSize,
      std::deque<std::vector<T>>& data,
      std::deque<mpi::Request>& requests, 
      std::deque<bool>& requestStatuses );
    
    Int NextIndexCoord
    ( std::deque<std::array<Int,3>>& coord,
      std::deque<mpi::Request>& requests, 
      std::deque<bool>& requestStatuses );

    // TODO
    Int GetIndexData( Matrix<T>& Z );
    Int GetIndexCoord( Matrix<T>& Z );

    bool TestRequests( Matrix<T>& Z );
    void WaitRequests( Matrix<T>& Z );

    // these are only used for nonblocking
    // update rountines
    void HandleGlobalToLocalData( Matrix<T>& Z );
    void HandleLocalToGlobalData( Matrix<T>& Z, Int source );
    void HandleLocalToGlobalAcc(  Matrix<T>& Z, Int source );

    void HandleGlobalToLocalData( const Matrix<T>& Z );
    void HandleLocalToGlobalData( const Matrix<T>& Z, Int source );
    void HandleLocalToGlobalAcc(  const Matrix<T>& Z, Int source );
};
} // namespace El
#endif // MPI-3
#endif // ifndef EL_AXPYINTERFACE2_HPP
