import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# generate a skewed normal distribution
def skewed_normal(x, mu, sigma, alpha):
    """
    generate a skewed normal distribution
    :param x: the x values
    :param mu: the mean
    :param sigma: the standard deviation
    :param alpha: the skewness
    :return: the skewed normal distribution
    """
    t = (x - mu) / sigma
    return 2 * sigma * norm.pdf(t) * norm.cdf(alpha * t)


# define the x range
x = np.linspace(0, 500, 100)
# define the parameters
mu = 140
sigma = 50
alpha = 0.5
# plot the function
plt.plot(x, skewed_normal(x, mu, sigma, alpha))
# limit the x axis
plt.xlim(0, 500)
plt.show()

# plot another figure by moving the mean
# define the x range
x = np.linspace(0, 500, 100)
# define the parameters
mu = 260
sigma = 50
alpha = 0.5
# plot the function
plt.plot(x, skewed_normal(x, mu, sigma, alpha))
# limit the x axis
plt.xlim(0, 500)
plt.show()

# plot another figure by moving the mean
# define the x range
x = np.linspace(0, 800, 100)
# define the parameters
mu = 380
sigma = 50
alpha = 0.5
# plot the function
plt.plot(x, skewed_normal(x, mu, sigma, alpha))
# limit the x axis
plt.xlim(0, 500)
plt.show()
