import numpy as np
import matplotlib.pyplot as plt

plt.subplot(2,2,1)
a = np.loadtxt("params.out")
plt.plot(a[:,0], a[:,1])

plt.subplot(2,2,2)
a = np.loadtxt("weights.out")
plt.plot(a[:,0], a[:,1])
plt.plot(a[:,0], a[:,2])

plt.subplot(2,2,3)
a = np.loadtxt("loss.out")
plt.plot(a[:,0], a[:,1])

plt.show()
