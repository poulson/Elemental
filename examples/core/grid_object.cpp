#include "elemental.hpp"

using namespace std;
using namespace elem;

typedef double R;
typedef float F;

void Foo( Grid** grid, mpi::Comm comm );
void Bar( mpi::Comm comm );

int nprocs;
int commRank;

int main( int argc, char* argv[] ){

  Initialize( argc, argv );
  mpi::Comm comm = mpi::COMM_WORLD;
  commRank = mpi::CommRank( comm );
  nprocs = mpi::CommSize(comm );
  
  Bar(comm);

     
  Finalize();
  return 0;
 }


void Foo( Grid** grid, mpi::Comm comm )
{
    *grid = new Grid( comm );
    DistMatrix<R> M (10,10,**grid);
}

void Bar( mpi::Comm comm )
{
    Grid* grid;
    Foo( &grid, comm );
    DistMatrix<R> N (10,10,*grid);
    delete grid;
}
