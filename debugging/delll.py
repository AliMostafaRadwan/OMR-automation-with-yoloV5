import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(4, -4, 1000)
y = np.cosh(x)

plt.plot(x, y)

plt.show()