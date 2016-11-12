import numpy as np


def misc():
	dt = 0.025
	alpha = 1
	
	#print t
	
	xsol = np.zeros(niter)
	xsol[0] = xT[0]
	
	x = xT[0]
	
	beta = 1-alpha*dt
	
	# Forward Euler scheme
	
	xsol[1] = 0.5*xsol[0] 
	
	for n in range(1,niter-1):
		xx = 0.5*(xsol[n] + xT[n])
		xsol[n+1] = xx*beta
	
	xsol2 = xsol.copy()
	
	#-------------------
	xsol[1] = 0.5*(xsol[0] + xT[0])
	
	for n in range(1,niter-1):
		xx = 0.5*(xsol[n] + xT[n])
		xsol[n+1] = xx*beta
	
	xsol3 = xsol.copy()
	
	#print xsol2 - xsol3
	
	#--------------------------
	#Forward Euler
	
	for n in range(0,niter-1):
		xx = xsol[n]
		xsol[n+1] = xx*beta
	
	xsol4 = xsol.copy()
	
	#print xsol2 - xsol4
	
#----------------------------------------------------------------------
# computation of loss function at the first time step, and of dL/d(alpha)
	
dt = 0.025
lr = 10      # learning rate
alpha = 1.5
niter = 300
t = np.linspace(0., niter, niter+1) * dt
xT = np.exp(-alpha*t)
xsol = np.zeros(niter)
xsol[0] = 0.

# start with it=1
def one_step(xsol, xT,  it):
	global alpha, dt
	beta = (1 - alpha * dt)
	xsol[it] = 0.5*(xsol[it-1] + xT[it-1]) * beta
	dLda = (xsol[it] - xT[it]) * (xsol[it-1] + xT[it-1]) * (-dt)
	dactdp = 0.5*(xsol[it-1]+xT[it-1])*(-dt)
	#print "Iteration ", it, ": dLda= ", dLda
	#print "  0.5*xT[0]= ",   0.5*xT[0]
	#print "0.5*xsol[0]= ", 0.5*xsol[0]
	#print "     sum   = ", 0.5*(xT[0]+xsol[0])
	#print "xsol[1]= ", xsol[1]
	#print "  xT[1]= ",   xT[1]
	#print "loss[0]= ", (xsol[1]-xT[1])**2

	alpha = alpha - lr * dLda
	print "alpha= ", alpha
	print

for i in range(1,100):
	one_step(xsol, xT, i)

# First iteration is correct

# Next iteration


#beta = (1 - alpha * dt)
#xsol[2] = 0.5*(xsol[1] + xT[1]) * beta
#dLda = (xsol[2] - xT[2]) * (xsol[1] + xT[1]) * (-dt)
#dactdp = 0.5*(xsol[1]+xT[1])*(-dt)
#print "d(act)/d(param)= ", dactdp
#print "Iteration 1: dLda= ", dLda
#print "  0.5*xT[1]= ",   0.5*xT[1]
#print "0.5*xsol[1]= ", 0.5*xsol[1]
#print "     sum   = ", 0.5*(xT[1]+xsol[1])
#print "    xsol[2]= ", xsol[2]
#print "      xT[2]= ",   xT[2]
#print "loss[1]= ", (xsol[2]-xT[2])**2
#
#alpha = alpha - lr * dLda
#print "alpha(2)= ", alpha
#print


# NOTES
# loss at iteration 1 does not match cpp code

#pred0= 0.4812
#exact0 = 0.9753

#pred1 = 0.7101
#exact1 = 0.9512

#dLda = (pred1-exact1) * (pred0+exact0) * (-dt)
#print "dLda = ", dLda
