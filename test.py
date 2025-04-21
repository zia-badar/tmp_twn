import matplotlib.pyplot as plt
import numpy as np

delta = 1
epsilon = 1
max_r = 3
min_r = -3

def fw_(x):
       if x >= -0.5 and x <= 0.5:
           return x * (epsilon)
       elif x < -0.5:
           return (x + 0.5) * (epsilon) + (-1 + epsilon/2)
       elif x > 0.5:
           return (x - 0.5) * (epsilon) + (1 - epsilon/2)

# Data for plotting
xs = np.arange(-1.5, 1.5, 0.001)
ys = []
for x in xs:
       ys.append(fw_(x))

fig, ax = plt.subplots()
ax.plot(xs, ys, 'bo')

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
ax.set_xticks(np.arange(-2, 2, 0.1))
ax.set_yticks(np.arange(-2, 2, 0.1))

fig.savefig("test.png")
plt.show()