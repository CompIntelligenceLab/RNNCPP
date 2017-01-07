import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("out")
print a.shape
plt.plot(a[:,0], a[:,1])

#Original sine
dt = 0.5
f = np.zeros(310)
for i in range(310):
	x = i * dt;
	f[i] = np.sin(x)

plt.plot(a[:,0], f, 'r')

plt.show()
