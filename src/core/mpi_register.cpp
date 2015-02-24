/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace {
El::mpi::Datatype typeIntInt;
El::mpi::Datatype typeFloatInt;
El::mpi::Datatype typeDoubleInt;

El::mpi::Op maxLocIntOp;
El::mpi::Op maxLocFloatOp;
El::mpi::Op maxLocDoubleOp;

El::mpi::Op minLocIntOp;
El::mpi::Op minLocFloatOp;
El::mpi::Op minLocDoubleOp;

El::mpi::Datatype typeIntIntPair;
El::mpi::Datatype typeFloatIntPair;
El::mpi::Datatype typeDoubleIntPair;

El::mpi::Op maxLocPairIntOp;
El::mpi::Op maxLocPairFloatOp;
El::mpi::Op maxLocPairDoubleOp;

El::mpi::Op minLocPairIntOp;
El::mpi::Op minLocPairFloatOp;
El::mpi::Op minLocPairDoubleOp;
} // anonymouse namespace   

namespace El {
namespace mpi {

template<typename T>
void
MaxLocFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueInt<T>* inData = static_cast<ValueInt<T>*>(inVoid);
    ValueInt<T>* outData = static_cast<ValueInt<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd = inData[j].index;
        const Int outInd = outData[j].index; 
        if( inVal > outVal || (inVal == outVal && inInd < outInd) )
            outData[j] = inData[j];
    }
}

template<typename T>
void
MinLocFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueInt<T>* inData = static_cast<ValueInt<T>*>(inVoid);
    ValueInt<T>* outData = static_cast<ValueInt<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd = inData[j].index;
        const Int outInd = outData[j].index; 
        if( inVal < outVal || (inVal == outVal && inInd < outInd) )
            outData[j] = inData[j];
    }
}

template<typename T>
void
MaxLocPairFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueIntPair<T>* inData = static_cast<ValueIntPair<T>*>(inVoid);
    ValueIntPair<T>* outData = static_cast<ValueIntPair<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd0 = inData[j].indices[0];
        const Int inInd1 = inData[j].indices[1];
        const Int outInd0 = outData[j].indices[0];
        const Int outInd1 = outData[j].indices[1];
        const bool inIndLess = 
            ( inInd0 < outInd0 || (inInd0 == outInd0 && inInd1 < outInd1) );
        if( inVal > outVal || (inVal == outVal && inIndLess) )
            outData[j] = inData[j];
    }
}

template<typename T>
void
MinLocPairFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueIntPair<T>* inData = static_cast<ValueIntPair<T>*>(inVoid);
    ValueIntPair<T>* outData = static_cast<ValueIntPair<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd0 = inData[j].indices[0];
        const Int inInd1 = inData[j].indices[1];
        const Int outInd0 = outData[j].indices[0];
        const Int outInd1 = outData[j].indices[1];
        const bool inIndLess = 
            ( inInd0 < outInd0 || (inInd0 == outInd0 && inInd1 < outInd1) );
        if( inVal < outVal || (inVal == outVal && inIndLess) )
            outData[j] = inData[j];
    }
}

template<>
Datatype& ValueIntType<Int>()        { return ::typeIntInt; }
template<>
Datatype& ValueIntType<float>()      { return ::typeFloatInt; }
template<>
Datatype& ValueIntType<double>()     { return ::typeDoubleInt; }
template<>
Datatype& ValueIntPairType<Int>()    { return ::typeIntIntPair; }
template<>
Datatype& ValueIntPairType<float>()  { return ::typeFloatIntPair; }
template<>
Datatype& ValueIntPairType<double>() { return ::typeDoubleIntPair; }

template<typename T>
void CreateValueIntType()
{
    DEBUG_ONLY(CallStackEntry cse("CreateValueIntType"))
    Datatype typeList[2];
    typeList[0] = TypeMap<T>();
    typeList[1] = TypeMap<Int>();
    
    int blockLengths[2];
    blockLengths[0] = 1;
    blockLengths[1] = 1; 

    ValueInt<T> v;
    MPI_Aint startAddr, valueAddr, indexAddr;
    MPI_Get_address( &v,       &startAddr );
    MPI_Get_address( &v.value, &valueAddr );
    MPI_Get_address( &v.index, &indexAddr );

    MPI_Aint displs[2];
    displs[0] = valueAddr - startAddr;
    displs[1] = indexAddr - startAddr;

    Datatype& type = ValueIntType<T>();
    MPI_Type_create_struct( 2, blockLengths, displs, typeList, &type );
    MPI_Type_commit( &type );
}
template void CreateValueIntType<Int>();
template void CreateValueIntType<float>();
template void CreateValueIntType<double>();

template<typename T>
void DestroyValueIntType()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyValueIntType"))
    Datatype& type = ValueIntType<T>();
    MPI_Type_free( &type );
}
template void DestroyValueIntType<Int>();
template void DestroyValueIntType<float>();
template void DestroyValueIntType<double>();

template<typename T>
void CreateValueIntPairType()
{
    DEBUG_ONLY(CallStackEntry cse("CreateValueIntPairType"))
    Datatype typeList[2];
    typeList[0] = TypeMap<T>();
    typeList[1] = TypeMap<Int>();
    
    int blockLengths[2];
    blockLengths[0] = 1;
    blockLengths[1] = 2; 

    ValueIntPair<T> v;
    MPI_Aint startAddr, valueAddr, indexAddr;
    MPI_Get_address( &v,        &startAddr );
    MPI_Get_address( &v.value,  &valueAddr );
    MPI_Get_address( v.indices, &indexAddr );

    MPI_Aint displs[2];
    displs[0] = valueAddr - startAddr;
    displs[1] = indexAddr - startAddr;

    Datatype& type = ValueIntPairType<T>();
    MPI_Type_create_struct( 2, blockLengths, displs, typeList, &type );
    MPI_Type_commit( &type );
}
template void CreateValueIntPairType<Int>();
template void CreateValueIntPairType<float>();
template void CreateValueIntPairType<double>();

template<typename T>
void DestroyValueIntPairType()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyValueIntPairType"))
    Datatype& type = ValueIntPairType<T>();
    MPI_Type_free( &type );
}
template void DestroyValueIntPairType<Int>();
template void DestroyValueIntPairType<float>();
template void DestroyValueIntPairType<double>();

template<>
void CreateMaxLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocOp<Int>"))
    Create( (UserFunction*)MaxLocFunc<Int>, true, ::maxLocIntOp );
}

template<>
void CreateMaxLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocOp<float>"))
    Create( (UserFunction*)MaxLocFunc<float>, true, ::maxLocFloatOp );
}

template<>
void CreateMaxLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocOp<double>"))
    Create( (UserFunction*)MaxLocFunc<double>, true, ::maxLocDoubleOp );
}

template<>
void CreateMinLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocOp<Int>"))
    Create( (UserFunction*)MinLocFunc<Int>, true, ::minLocIntOp );
}

template<>
void CreateMinLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocOp<float>"))
    Create( (UserFunction*)MinLocFunc<float>, true, ::minLocFloatOp );
}

template<>
void CreateMinLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocOp<double>"))
    Create( (UserFunction*)MinLocFunc<double>, true, ::minLocDoubleOp );
}

template<>
void CreateMaxLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocPairOp<Int>"))
    Create( (UserFunction*)MaxLocPairFunc<Int>, true, ::maxLocPairIntOp );
}

template<>
void CreateMaxLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocPairOp<float>"))
    Create( (UserFunction*)MaxLocPairFunc<float>, true, ::maxLocPairFloatOp );
}

template<>
void CreateMaxLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMaxLocPairOp<double>"))
    Create( (UserFunction*)MaxLocPairFunc<double>, true, ::maxLocPairDoubleOp );
}

template<>
void CreateMinLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocPairOp<Int>"))
    Create( (UserFunction*)MinLocPairFunc<Int>, true, ::minLocPairIntOp );
}

template<>
void CreateMinLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocPairOp<float>"))
    Create( (UserFunction*)MinLocPairFunc<float>, true, ::minLocPairFloatOp );
}

template<>
void CreateMinLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("CreateMinLocPairOp<double>"))
    Create( (UserFunction*)MinLocPairFunc<double>, true, ::minLocPairDoubleOp );
}

template<>
void DestroyMaxLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocOp<Int>"))
    Free( ::maxLocIntOp );
}

template<>
void DestroyMaxLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocOp<float>"))
    Free( ::maxLocFloatOp );
}

template<>
void DestroyMaxLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocOp<double>"))
    Free( ::maxLocDoubleOp );
}

template<>
void DestroyMinLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocOp<Int>"))
    Free( ::minLocIntOp );
}

template<>
void DestroyMinLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocOp<float>"))
    Free( ::minLocFloatOp );
}

template<>
void DestroyMinLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocOp<double>"))
    Free( ::minLocDoubleOp );
}

template<>
void DestroyMaxLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocPairOp<Int>"))
    Free( ::maxLocPairIntOp );
}

template<>
void DestroyMaxLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocPairOp<float>"))
    Free( ::maxLocPairFloatOp );
}

template<>
void DestroyMaxLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMaxLocPairOp<double>"))
    Free( ::maxLocPairDoubleOp );
}

template<>
void DestroyMinLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocPairOp<Int>"))
    Free( ::minLocPairIntOp );
}

template<>
void DestroyMinLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocPairOp<float>"))
    Free( ::minLocPairFloatOp );
}

template<>
void DestroyMinLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("DestroyMinLocPairOp<double>"))
    Free( ::minLocPairDoubleOp );
}

template<>
Op MaxLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocOp<Int>"))
    return ::maxLocIntOp;
}

template<>
Op MaxLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocOp<float>"))
    return ::maxLocFloatOp;
}

template<>
Op MaxLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocOp<double>"))
    return ::maxLocDoubleOp;
}

template<>
Op MinLocOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocOp<Int>"))
    return ::minLocIntOp;
}

template<>
Op MinLocOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocOp<float>"))
    return ::minLocFloatOp;
}

template<>
Op MinLocOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocOp<double>"))
    return ::minLocDoubleOp;
}

template<>
Op MaxLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocPairOp<Int>"))
    return ::maxLocPairIntOp;
}

template<>
Op MaxLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocPairOp<float>"))
    return ::maxLocPairFloatOp;
}

template<>
Op MaxLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("MaxLocPairOp<double>"))
    return ::maxLocPairDoubleOp;
}

template<>
Op MinLocPairOp<Int>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocPairOp<Int>"))
    return ::minLocPairIntOp;
}

template<>
Op MinLocPairOp<float>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocPairOp<float>"))
    return ::minLocPairFloatOp;
}

template<>
Op MinLocPairOp<double>()
{
    DEBUG_ONLY(CallStackEntry cse("MinLocPairOp<double>"))
    return ::minLocPairDoubleOp;
}

template void
MaxLocFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocFunc<double>( void* in, void* out, int* length, Datatype* datatype );

template void
MinLocFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MinLocFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MinLocFunc<double>( void* in, void* out, int* length, Datatype* datatype );

template void
MaxLocPairFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocPairFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocPairFunc<double>( void* in, void* out, int* length, Datatype* datatype );

template void
MinLocPairFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MinLocPairFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MinLocPairFunc<double>( void* in, void* out, int* length, Datatype* datatype );

} // namespace mpi
} // namespace El
