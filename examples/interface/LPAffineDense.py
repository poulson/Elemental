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

m = 500
n = 1000
k = 750
testMehrotra = True
testIPF = False
manualInit = False
display = False
progress = True
worldRank = El.mpi.WorldRank()

# Make a dense matrix
def RectangDense(height,width):
  A = El.DistMatrix()
  El.Gaussian( A, height, width )
  return A

A = RectangDense(m,n)
G = RectangDense(k,n)

# Generate a (b,h) which implies a primal feasible (x,s)
# ======================================================
xGen = El.DistMatrix()
# b := A xGen
# -----------
El.Gaussian(xGen,n,1)
b = El.DistMatrix()
El.Zeros( b, m, 1 )
El.Gemv( El.NORMAL, 1., A, xGen, 0., b )
# h := G xGen + sGen
# ------------------
sGen = El.DistMatrix()
El.Uniform(sGen,k,1,0.5,0.5)
h = El.DistMatrix()
El.Copy( sGen, h )
El.Gemv( El.NORMAL, 1., G, xGen, 1., h )

# Generate a c which implies a dual feasible (y,z)
# ================================================
yGen = El.DistMatrix()
El.Gaussian(yGen,m,1)
zGen = El.DistMatrix()
El.Uniform(zGen,k,1,0.5,0.5)
c = El.DistMatrix()
El.Zeros( c, n, 1 )
El.Gemv( El.TRANSPOSE, -1., A, yGen, 1., c )
El.Gemv( El.TRANSPOSE, -1., G, zGen, 1., c )

if display:
  El.Display( A, "A" )
  El.Display( G, "G" )
  El.Display( b, "b" )
  El.Display( c, "c" )
  El.Display( h, "h" )

# Set up the control structure (and possibly initial guesses)
# ===========================================================
ctrl = El.LPAffineCtrl_d()
xOrig = El.DistMatrix()
yOrig = El.DistMatrix()
zOrig = El.DistMatrix()
sOrig = El.DistMatrix()
if manualInit:
  El.Uniform(xOrig,n,1,0.5,0.4999)
  El.Uniform(yOrig,m,1,0.5,0.4999)
  El.Uniform(zOrig,k,1,0.5,0.4999)
  El.Uniform(sOrig,k,1,0.5,0.4999)
x = El.DistMatrix()
y = El.DistMatrix()
z = El.DistMatrix()
s = El.DistMatrix()

if testMehrotra:
  ctrl.approach = El.LP_MEHROTRA
  ctrl.mehrotraCtrl.primalInitialized = manualInit
  ctrl.mehrotraCtrl.dualInitialized = manualInit
  ctrl.mehrotraCtrl.progress = progress
  El.Copy( xOrig, x )
  El.Copy( yOrig, y )
  El.Copy( zOrig, z )
  El.Copy( sOrig, s )
  startMehrotra = time.clock()
  El.LPAffine(A,G,b,c,h,x,y,z,s,ctrl)
  endMehrotra = time.clock()
  if worldRank == 0:
    print "Mehrotra time:", endMehrotra-startMehrotra

  if display:
    El.Display( x, "x Mehrotra" )
    El.Display( y, "y Mehrotra" )
    El.Display( z, "z Mehrotra" )
    El.Display( s, "s Mehrotra" )

  obj = El.Dot(c,x)
  if worldRank == 0:
    print "Mehrotra c^T x =", obj

if testIPF:
  ctrl.approach = El.LP_IPF
  ctrl.ipfCtrl.primalInitialized = manualInit
  ctrl.ipfCtrl.dualInitialized = manualInit
  ctrl.ipfCtrl.progress = progress
  ctrl.ipfCtrl.lineSearchCtrl.progress = progress
  El.Copy( xOrig, x )
  El.Copy( yOrig, y )
  El.Copy( zOrig, z )
  El.Copy( sOrig, s )
  startIPF = time.clock()
  El.LPAffine(A,G,b,c,h,x,y,z,s,ctrl)
  endIPF = time.clock()
  if worldRank == 0:
    print "IPF time:", endIPF-startIPF

  if display:
    El.Display( x, "x IPF" )
    El.Display( y, "y IPF" )
    El.Display( z, "z IPF" )
    El.Display( s, "s IPF" )

  obj = El.Dot(c,x)
  if worldRank == 0:
    print "IPF c^T x =", obj

# Require the user to press a button before the figures are closed
commSize = El.mpi.Size( El.mpi.COMM_WORLD() )
El.Finalize()
if commSize == 1:
  raw_input('Press Enter to exit')
