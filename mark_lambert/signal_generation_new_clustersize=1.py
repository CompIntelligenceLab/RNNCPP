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

def signal(t,dt,set_signal,x=1.0,alpha=1.0):
	if(set_signal=='exp'):
		return(signal_exp(t,dt,alpha=alpha))

def signal_exp(t,dt,alpha=1.0): #exponential decay
	x=np.exp(-alpha*t)
	dx=-alpha*x*dt
	#return(x,dx)
	#return(x)
	return(x)

def activation_exp(alpha,dt):
	return(1-alpha*dt)
	
#set signal type
set_signal='exp'	
#target_signal='new_signal'
target_signal='exp'
	
# set timesteps
# note: the answer comes out better with a smaller dt and a larger number for nsteps
if(set_signal=='exp'):
	dt=0.001
	alpha=1.0
	alpha_tgt=2.0 #target alpha
	
#initialize starting value at t=0
x_0=1.0
lambd=1.0 #lagrange multiplier
# length of signal
cluster_size=1
nsteps=int(math.ceil(2000))
nepochs=100

# set learning rate
lr=.01
decay=lr/nepochs #if this is 0, learning rate is constant

for epoch in xrange(nepochs):
	loss=0
	x=np.zeros((nsteps+1))
	x_star=np.zeros((nsteps+1))
	x_target=np.zeros((nsteps+1))
	t=0
	x[0]=x_0
	x_star[0]=x_0
	x_target[0]=x_0
	lr = lr*1/(1+decay*epoch) #change learning rate each epoch
	
	#set initial cluster
	for j in xrange(1):
		if(j>0): #x[0] already set
			x[j]=signal(t,dt,set_signal,x[j-1],alpha,beta,gamma,delta)
			x_star[j]=signal(t,dt,set_signal,x_star[j-1],a,b,c,d)
			x_target[j]=signal(t,dt,target_signal,x_target[j-1],alpha_tgt,beta_tgt,gamma_tgt,delta_tgt)
			t+=dt
	
	for i in xrange(nsteps): #i is the current cluster number
		dalpha=0
		for j in xrange(1): #j is the position within the cluster
			x_target[(i+1)+j]=signal(t+dt,dt,target_signal,x_target[(i+1)+j-1],alpha_tgt)
			# forward propagation
			if(set_signal=='exp'):
				hidden_node = activation_exp(alpha,dt) #cluster size >1 means that the dt distance between the previous point that had a different alpha is increased by a factor of cluster_size (1)
				x[(i+1)+j]=x[(i+1)+j-1]*hidden_node
			

			# update weights
			
			#Loss function
			loss += (x[(i+1)+j]-x_target[(i+1)+j])**2
			
			t += dt
		
		alpha -= lr*dalpha #weights update after each cluster
		
	loss=loss/(nsteps) #mean square error of x and x_target

	
	print "After epoch",epoch+1,":"
	print "alpha = ",alpha,"  target = ",alpha_tgt
	print "loss  = ",loss

	if(epoch<3 or epoch==(nepochs-1)):
		single_plot(dt,nsteps,x,x_target, epoch)
	
#-----------------------------------
# FINAL PLOTS
x_plot=np.zeros(len(t_plot))
x_plot[0]=x_0
x_plot2=np.zeros(len(t_plot))
x_plot2[0]=x_0
loss_sum=0
for i in xrange(len(t_plot)-1):
	if(set_signal=='exp'):
		x_plot[i+1]=x_plot[i]-dt*alpha*x_plot[i]
	loss_sum+=(x_plot[i]-x_target[i])**2
loss_mse_final=loss_sum/len(t_plot)
if(set_signal=='exp'):
	plt.suptitle('alpha = %f \nalpha_target= 2.0' %(alpha))
plt.plot(t_plot,x_plot)
plt.plot(t_plot,x_target)
#plt.plot(t_plot,x_plot2) #x_plot2 uses a,b,c,d instead of alpha,beta,gamma,delta to generate curve
plt.xlabel('t')
plt.ylabel('x')
plt.show()
print "final mse = ",loss_mse_final
