/*
   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_AXPYINTERFACE2_HPP
#define EL_AXPYINTERFACE2_HPP

namespace El {
template<typename T>
class AxpyInterface2
{
public:
    AxpyInterface2();
    ~AxpyInterface2();

    AxpyInterface2(       DistMatrix<T,MC,MR>& Z );
    AxpyInterface2( const DistMatrix<T,MC,MR>& Z );

    // collective epoch initialization routines
    void Attach(       DistMatrix<T,MC,MR>& Z );
    void Attach( const DistMatrix<T,MC,MR>& Z );
    void Detach();

    // remote update routines
    
    // requires Flush for local+remote 
    // completion
    void Iput( Matrix<T>& Z, Int i, Int j );
    void Iput( const Matrix<T>& Z, Int i, Int j );

    void Iget(       Matrix<T>& Z, Int i, Int j );

    void Iacc(       Matrix<T>& Z, Int i, Int j );
    void Iacc( const Matrix<T>& Z, Int i, Int j );
    
    // locally blocking update routines
    // currently not implemented
    void Put( Matrix<T>& Z, Int i, Int j );
    void Put( const Matrix<T>& Z, Int i, Int j );

    void Get(       Matrix<T>& Z, Int i, Int j );

    void Acc(       Matrix<T>& Z, Int i, Int j );
    void Acc( const Matrix<T>& Z, Int i, Int j );

    // synchronization routines
    void Flush(          Matrix<T>& Z );
    void Flush(    const Matrix<T>& Z ); 
 
private:
   
    static const Int 
        DATA_PUT_TAG      =1, 
        DATA_GET_TAG      =2,
        DATA_ACC_TAG   	  =3,
        REQUEST_GET_TAG   =4,
	COORD_ACC_TAG     =5,
	COORD_PUT_TAG     =6;	

    // struct for passing data
    struct matrix_params_
    {
	T *base_;
	std::vector<std::deque<std::vector<T>>>
	    data_;
	std::vector<std::deque<mpi::Request>> 
	    requests_;
	std::vector<std::deque<bool>> 
	    statuses_;
    };
        	
    std::vector<struct matrix_params_> matrices_;

    // struct for passing coordinates
    struct coord_params_
    {
	T *base_;
	std::vector<std::deque<std::array<Int, 3>>>
	    coord_;
	std::vector<std::deque<mpi::Request>> 
	    requests_;
	std::vector<std::deque<bool>> 
	    statuses_;
    };
        	
    std::vector<struct coord_params_> coords_;

    // TODO need to add const here...
    DistMatrix<T,MC,MR>* GlobalArrayPut_;
    DistMatrix<T,MC,MR>* GlobalArrayGet_;
   
    bool toBeAttachedForPut_, toBeAttachedForGet_, 
	 attached_, detached_;

    // next index for data and coord
    Int NextIndexData (
	Int target,
	Int dataSize, 
	T * base_address,
	Int *mindex);
   
    Int NextIndexCoord (
	Int i, Int j,
	Int target,
	T * base_address,
	Int *cindex);

    bool Testall();
    bool Test(          Matrix<T>& Z );
    bool Test(    const Matrix<T>& Z );  
    bool TestAny(          Matrix<T>& Z );
    bool TestAny(    const Matrix<T>& Z ); 

    void Waitall();
    void Wait(          Matrix<T>& Z );
    void Wait(    const Matrix<T>& Z );    
    void WaitAny(          Matrix<T>& Z );
    void WaitAny(    const Matrix<T>& Z );   

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
#endif // ifndef EL_AXPYINTERFACE2_HPP
