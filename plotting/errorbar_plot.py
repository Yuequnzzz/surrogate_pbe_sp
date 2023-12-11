import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')

# load data
path = "D:/PycharmProjects/GMM/data/Prediction_hyperparameter/"
file = pd.read_csv(path + "hyperparamOpt_231201_0928.csv")  # for "InputMat_231021_0805"
# file = pd.read_csv(path + "hyperparamOpt_231201_1251.csv")  # for "InputMat_231115_1455"

# the mean value of the RMSE
fx = file['nodes'].values
fy = file['layers'].values
fz = file['RMSE_tot_mean'].values

# the standard deviation of the RMSE
error = file['RMSE_tot_std'].values

# plot
ax.plot(fx, fy, fz, linestyle="None", color='r', marker="o")

for i in np.arange(0, len(fx)):
    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i] + error[i], fz[i] - error[i]], marker="_", color="b")

# configure axes
ax.set_xlim3d(0, 105)
ax.set_ylim3d(0, 12)
ax.set_zlim3d(10, 30)
ax.set_xlabel('nodes')
ax.set_ylabel('layers')
ax.set_zlabel('RMSE total mean')

plt.show()