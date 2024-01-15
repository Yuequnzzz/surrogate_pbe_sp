# generate different distributions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def f(x, mu_1, mu_2, sigma_1, sigma_2, alpha):
    """
    generate a skewed normal distribution plus a normal distribution
    :param x: the x values
    :param mu_1: the mean of the skewed normal distribution
    :param sigma_1: the standard deviation of the skewed normal distribution
    :param alpha: the skewness
    :param mu_2: the mean of the normal distribution
    :param sigma_2: the standard deviation of the normal distribution
    :return: the combined distribution
    """
    t1 = (x - mu_1) / sigma_1
    t2 = (x - mu_2) / sigma_2
    return 2 * sigma_1 * norm.pdf(t1) * norm.cdf(alpha * t1) + 50 * norm.pdf(t2)


def g(x, mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3, alpha):
    """
    generate a skewed normal distribution plus two normal distributions
    :param x: the x values
    :param mu_1: the mean of the skewed normal distribution
    :param sigma_1: the standard deviation of the skewed normal distribution
    :param alpha: the skewness
    :param mu_2: the mean of the normal distribution
    :param sigma_2: the standard deviation of the normal distribution
    :param mu_3: the mean of the normal distribution
    :param sigma_3: the standard deviation of the normal distribution
    :return: the combined distribution
    """
    t1 = (x - mu_1) / sigma_1
    t2 = (x - mu_2) / sigma_2
    t3 = (x - mu_3) / sigma_3
    return 2 * sigma_1 * norm.pdf(t1) * norm.cdf(alpha * t1) + 10 * norm.pdf(t2) + 40 * norm.pdf(t3)


def h_nucleation(x, mu_1, mu_2, sigma_1, sigma_2, alpha, step_height, step_length):
    """
    generate a skewed normal distribution plus a normal distribution plus a step function (just at the start)
    :param x: the x values
    :param mu_1: the mean of the skewed normal distribution
    :param sigma_1: the standard deviation of the skewed normal distribution
    :param alpha: the skewness
    :param mu_2: the mean of the normal distribution
    :param sigma_2: the standard deviation of the normal distribution
    :param step_height: the height of the step function
    :param step_length: the length of the step function
    :return: the combined distribution
    """
    t1 = (x - mu_1) / sigma_1
    t2 = (x - mu_2) / sigma_2
    step = np.zeros(x.size)
    # step starts at 0 and ends at step_length
    step[np.where(x < step_length)] = step_height
    # the function without the step function
    h_no_step = 2 * sigma_1 * norm.pdf(t1) * norm.cdf(alpha * t1) + 50 * norm.pdf(t2)
    # the function with the step function
    h = h_no_step + step
    return h_no_step, h


# define the x range
x = np.linspace(0, 500, 100)
# define the parameters
mu_1 = 140
mu_2 = 260
sigma_1 = 50
sigma_2 = 10
alpha = 0.5
# plot the function
plt.plot(x, f(x, mu_1, mu_2, sigma_1, sigma_2, alpha))
# limit the x axis
plt.xlim(0, 500)
# remove the axis and the ticks
plt.axis('off')
plt.show()

# plot another figure with g(x)
# define the x range
x = np.linspace(0, 500, 100)
# define the parameters
mu_1 = 140
mu_2 = 260
mu_3 = 380
sigma_1 = 50
sigma_2 = 10
sigma_3 = 20
alpha = 1
# plot the function
plt.plot(x, g(x, mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3, alpha))
# limit the x axis
plt.xlim(0, 500)
plt.show()

# plot another figure with h(x)
# define the x range
x = np.linspace(0, 500, 100)
# define the parameters
mu_1 = 160
mu_2 = 220
sigma_1 = 20
sigma_2 = 15
alpha = 0.5
step_height = 3
step_length = 40
# plot the function
plt.plot(x, h_nucleation(x, mu_1, mu_2, sigma_1, sigma_2, alpha, step_height, step_length)[1])
# limit the x axis
plt.xlim(0, 500)
plt.show()








