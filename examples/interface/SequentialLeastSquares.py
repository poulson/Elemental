#
#  Copyright (c) 2009-2015, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License, 
#  which can be found in the LICENSE file in the root directory, or at 
#  http://opensource.org/licenses/BSD-2-Clause
#
import El

n0 = n1 = 20
display = False

def ExtendedLaplacian(xSize,ySize):
  A = El.SparseMatrix()
  n = xSize*ySize
  A.Resize(2*n,n)
  A.Reserve(6*n)
  hxInvSq = (1.*(xSize+1))**2
  hyInvSq = (1.*(ySize+1))**2
  for s in xrange(2*n):
    if s < xSize*ySize:
      x = s % xSize
      y = s / xSize
      A.QueueUpdate( s, s, 2*(hxInvSq+hyInvSq) )
      if x != 0:       A.QueueUpdate( s, s-1,     -hxInvSq )
      if x != xSize-1: A.QueueUpdate( s, s+1,     -hxInvSq )
      if y != 0:       A.QueueUpdate( s, s-xSize, -hyInvSq )
      if y != ySize-1: A.QueueUpdate( s, s+xSize, -hyInvSq )
    else:
      A.QueueUpdate( s, s-xSize*ySize, 2*(hxInvSq+hyInvSq) )

  A.MakeConsistent()
  return A

A = ExtendedLaplacian(n0,n1)
if display:
  El.Display( A, "A" )
  El.Display( A.Graph(), "Graph of A" )

y = El.Matrix()
El.Uniform( y, 2*n0*n1, 1 )
if display:
  El.Display( y, "y" )
yNrm = El.Nrm2(y)
print "|| y ||_2 =", yNrm

x = El.LeastSquares(A,y)
xNrm = El.Nrm2(x)
if display:
  El.Display( x, "x" )
print "|| x ||_2 =", xNrm
El.SparseMultiply(El.NORMAL,-1.,A,x,1.,y)
if display:
  El.Display( y, "A x - y" )
eNrm = El.Nrm2(y)
print "|| A x - y ||_2 / || y ||_2 =", eNrm/yNrm

# Require the user to press a button before the figures are closed
El.Finalize()
raw_input('Press Enter to exit')
