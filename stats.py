import numpy as np
import matplotlib.pyplot as plt

#a = np.loadtxt("s.out")
a = np.loadtxt("stats.out")
sz = a.shape[1];

plt.subplot(2,2,1)
for i in range(0,3):
	plt.plot(a[:,i])
plt.title("amplitude")

plt.subplot(2,2,2)
for i in range(3,6):
	plt.plot(a[:,i])
plt.title("mean")

plt.subplot(2,2,3)
for i in range(6,9):
	plt.plot(a[:,i])
plt.title("std dev")

plt.show()

print a[1]
