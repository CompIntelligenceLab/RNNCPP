import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("out")
print a.shape
plt.plot(a[:,0], a[:,1])

#Original sine
npts = a.shape[0];

plt.plot(a[:,0], a[:,2], 'r')

plt.show()
