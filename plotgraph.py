import numpy as np
import matplotlib.pyplot as plt

plt.subplot(2,2,1)
a = np.loadtxt("params.out")
sz = a.shape[1]
plt.title("activation params")
for i in range(1,sz):
	plt.plot(a[:,0], a[:,i])

plt.subplot(2,2,2)
a = np.loadtxt("weights.out")
sz = a.shape[1]
print sz
plt.title("weights")
for i in range(1,sz):
	plt.plot(a[:,0], a[:,i])

plt.subplot(2,2,3)
plt.title("loss")
a = np.loadtxt("loss.out")
plt.plot(a[:,0], a[:,1])

plt.subplot(2,2,4)
plt.title("x")
a = np.loadtxt("x.out")
sz = a.shape[1]
plt.plot(a[:,0], a[:,1], label='target')
plt.plot(a[:,0], a[:,2], label='predict')
plt.legend()
#for i in range(1,sz):
#	plt.plot(a[:,0], a[:,i])

plt.show()
