import numpy as np
import matplotlib.pyplot as plt

#a = np.loadtxt("s.out")
a = np.loadtxt("stats.out")
sz = a.shape[1];

nb = 3

plt.subplot(2,2,1)
for i in range(0,nb):
	plt.plot(a[:,i])
plt.title("amplitude")

plt.subplot(2,2,2)
for i in range(nb,2*nb):
	plt.plot(a[:,i])
plt.title("mean")

plt.subplot(2,2,3)
for i in range(2*nb,3*nb):
	plt.plot(a[:,i])
plt.title("std dev")

plt.savefig("stats.pdf")
#plt.show()
#print "gordon"

