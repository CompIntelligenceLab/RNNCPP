import numpy as np
import matplotlib.pyplot as plt

#a = np.loadtxt("s.out")
a = np.loadtxt("stats.out")
nb_gmm = a.shape[1] / 3

plt.subplot(2,2,1)
for i in range(0,nb_gmm):
	plt.plot(a[:,i])
plt.title("amplitude")

plt.subplot(2,2,2)
for i in range(nb_gmm,2*nb_gmm):
	plt.plot(a[:,i])
plt.title("mean")

plt.subplot(2,2,3)
for i in range(2*nb_gmm,3*nb_gmm):
	plt.plot(a[1000:,i])
plt.title("std dev")

plt.savefig("stats.pdf")
#plt.show()
#print "gordon"

