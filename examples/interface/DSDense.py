#
#  Copyright (c) 2009-2015, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License, 
#  which can be found in the LICENSE file in the root directory, or at 
#  http://opensource.org/licenses/BSD-2-Clause
#
import El
import time

m = 200
n = 400
numLambdas = 5
startLambda = 0.01
endLambda = 1
display = True
worldRank = El.mpi.WorldRank()

def Rectang(height,width):
  A = El.DistMatrix()
  El.Uniform( A, height, width )
  return A

A = Rectang(m,n)
b = El.DistMatrix()
El.Gaussian( b, m, 1 )
if display:
  El.Display( A, "A" )
  El.Display( b, "b" )

ctrl = El.LPAffineCtrl_d()
ctrl.mehrotraCtrl.progress = True

for j in xrange(0,numLambdas):
  lambd = startLambda + j*(endLambda-startLambda)/(numLambdas-1.)
  if worldRank == 0:
    print "lambda =", lambd

  startDS = time.clock()
  x = El.DS( A, b, lambd, ctrl )
  endDS = time.clock()
  if worldRank == 0:
    print "DS time: ", endDS-startDS

  if display:
    El.Display( x, "x" )

  xOneNorm = El.EntrywiseNorm( x, 1 )
  r = El.DistMatrix()
  El.Copy( b, r )
  El.Gemv( El.NORMAL, -1., A, x, 1., r )
  rTwoNorm = El.Nrm2( r )
  t = El.DistMatrix()
  El.Zeros( t, n, 1 )
  El.Gemv( El.TRANSPOSE, 1., A, r, 0., t )
  tTwoNorm = El.Nrm2( t )
  tInfNorm = El.MaxNorm( t )
  if display:
    El.Display( r, "r" )
    El.Display( t, "t" )
  if worldRank == 0:
    print "|| x ||_1       =", xOneNorm
    print "|| b - A x ||_2 =", rTwoNorm
    print "|| A^T (b - A x) ||_2 =", tTwoNorm
    print "|| A^T (b - A x) ||_oo =", tInfNorm

# Require the user to press a button before the figures are closed
commSize = El.mpi.Size( El.mpi.COMM_WORLD() )
El.Finalize()
if commSize == 1:
  raw_input('Press Enter to exit')
