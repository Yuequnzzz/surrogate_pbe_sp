# try to draw a 2d function
import numpy as np
import matplotlib.pyplot as plt


# define the 2d function
def f(x):
    return 2 * np.sin(x) - 0.2 * 2 * x ** 2 + np.exp(0.2*x)


# define the x range
x = np.linspace(-5, 5, 100)
# plot the function
plt.plot(x, f(x))
plt.show()

# plot another figure with the same function but marked the random points
# define the random points
x_random = np.random.uniform(-5, 5, 50)
# plot the random points with red cross
plt.plot(x_random, f(x_random), 'rx')
plt.show()

# plot the dotted line for the function
plt.plot(x, f(x), ls='--')
# with the dotted line, plot the random points with red cross
plt.plot(x_random, f(x_random), 'rx')
plt.show()

