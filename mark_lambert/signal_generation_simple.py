import numpy as np
import math
import matplotlib.pyplot as plt

def single_plot(dt,nsteps,x,x_target, epoch):
		t_plot=np.linspace(0, dt*nsteps, (nsteps+1))
		plt.plot(t_plot,x)
		plt.plot(t_plot,x_target)
		plt.suptitle('Epoch %d' %(epoch+1))
		plt.xlabel('t')
		plt.ylabel('x')
		plt.show()

def signal(t,alpha=1.0):
	x=np.exp(-alpha*t)
	return(x)

def activation_exp(alpha,dt):
	return(1-alpha*dt)
	
#set signal type
target_signal='exp'

def activation_exp(alpha,dt):
	return(1-alpha*dt)
	
#set signal type
set_signal='exp'	
target_signal='exp'
	
# set timesteps
# note: the answer comes out better with a smaller dt and a larger number for nsteps
if(set_signal=='exp'):
	dt=0.001
	alpha=2.0
	alpha_tgt=1.0 #target alpha
	
#initialize starting value at t=0
x_0=1.0
# length of signal
nsteps=int(math.ceil(2000))
nepochs=100

# set learning rate
lr = 2.0
decay = lr/nepochs #if this is 0, learning rate is constant
decay = 0  # for testing without scheduling of lr

for epoch in xrange(nepochs):
	loss=0
	x=np.zeros(nsteps+1)
	x_star=np.zeros(nsteps+1)
	x_target=np.zeros(nsteps+1)
	t=0
	x[0]=x_0
	x_star[0]=x_0
	x_target[0]=x_0
	lr = lr / (1+decay*epoch) #change learning rate each epoch
	
	for i in xrange(nsteps):
		dalpha=0
		x_target[i+1]=signal(t+dt,alpha_tgt)
		# forward propagation
		x[i+1]=x[i]*(1-alpha*dt)
			
		# update weights
		dalpha += 2*(x[i+1]-x_target[i])*-dt*x[i] #backward pass

		#Loss function
		loss += (x[i+1]-x_target[i+1])**2
		t += dt
		alpha -= lr*dalpha 
	
	loss=loss/(nsteps) #mean square error of x and x_target
	
	print(x)
	print "After epoch",epoch+1,":"
	print "alpha = ",alpha,"  target = ",alpha_tgt
	print "loss  = ",loss

	if(epoch<3 or epoch==(nepochs-1)):
		single_plot(dt,nsteps,x,x_target, epoch)
	
#-----------------------------------
# FINAL PLOTS
t_plot=np.linspace(0, dt*nsteps, (nsteps+1))
x_plot=np.zeros(len(t_plot))
x_plot[0]=x_0
x_plot2=np.zeros(len(t_plot))
x_plot2[0]=x_0
loss_sum=0

for i in xrange(len(t_plot)-1):
	x_plot[i+1]=x_plot[i]-dt*alpha*x_plot[i]
	loss_sum+=(x_plot[i]-x_target[i])**2

loss_mse_final=loss_sum/len(t_plot)
plt.suptitle('alpha = %f \nalpha_target= 2.0' %(alpha))
plt.plot(t_plot,x_plot)
plt.plot(t_plot,x_target)
plt.xlabel('t')
plt.ylabel('x')
plt.show()
print "final mse = ",loss_mse_final
