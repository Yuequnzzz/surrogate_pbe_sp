# #########################################################

# 1D Population Balance Solver
# Particle Size Distribution Class

# #########################################################
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import json
from scipy.stats import skewnorm

sns.set_style("whitegrid")


def convertToJsonFriendly(val):
    if isinstance(val, (np.ndarray, list)):
        return [convertToJsonFriendly(x) for x in val]
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    else:
        return val


class PSD:
    def __init__(
        self,
        dist_shape="gaussian",
        mu=None,
        sigma=None,
        skewness=None,
        weights=None,
        dL=None,
        L_max=None,
        load_file=None,
        f_vec=None,
        tot_number=5000,
    ):
        if load_file is not None:
            # Load size distribution from json file
            with open(f"{load_file}.json", "r") as f:
                PSD_dict = json.load(f)
            self.mu = mu  # [um]
            self.sigma = sigma  # [um]
            self.load_file = load_file
            self.dL = PSD_dict["dL"]  # [um]
            self.L_max = PSD_dict["L_max"]  # [um]
            self.tot_number = PSD_dict["tot_number"]

            self.L_bounds = np.array(PSD_dict["L_bounds"])  # [um]
            self.L_mid = np.array(PSD_dict["L_bounds"])  # [um]
            self.nbins = PSD_dict["nbins"]

            self.f = np.array(PSD_dict["f"])
        
        if f_vec is not None:
            # Generate population from a population density vector
            self.f = f_vec
            self.nbins = len(f_vec)
            self.L_max = L_max
            self.dL = L_max/self.nbins
            self.L_bounds = np.arange(0, L_max + self.dL, self.dL)  # [um]
            self.L_mid = np.mean(
                [self.L_bounds[:-1], self.L_bounds[1:]], axis=0
            )  # [um]
            self.computeMoments()

        else:
            # Generate new gaussian size distribution
            self.dL = dL  # [um]
            self.L_max = L_max  # [um]
            self.tot_number = tot_number

            # Create bins
            self.L_bounds = np.arange(0, L_max + dL, dL)  # [um]
            self.L_mid = np.mean(
                [self.L_bounds[:-1], self.L_bounds[1:]], axis=0
            )  # [um]
            self.nbins = len(self.L_mid)

            # Create size distribution
            if dist_shape == "gaussian":
                self.f = (
                    1
                    / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-((self.L_mid - mu) ** 2) / (2 * sigma**2))
                )  # gaussian shape

            elif dist_shape == "skewed":
                self.f = skewnorm.pdf(self.L_mid, skewness, mu, sigma)

            elif dist_shape == "multiple":
                if weights is None:
                    weights = [1]*len(mu)
                fs = []
                for m, sig, sk, w in zip(mu, sigma, skewness, weights):
                    fs.append(w * skewnorm.pdf(self.L_mid, sk, m, sig))
                self.f = np.sum(np.array(fs), axis=0)

            # Normalize to requested number of crystals
            self.computeMoments()
            self.f = (
                self.f * tot_number / self.moments[0]
            )  # normalize so that total count is equal to total number of crystals
            self.computeMoments()

    def __str__(self):
        return f"1D PSSD with dL={self.dL}, and {self.nbins} bins"

    def plot(self, color="b", ls="-", label="", cumulative=False):
        """Plot distribution

        Args:
            color (str, optional): Line color. Defaults to "b".
            ls (str, optional): Line style. Defaults to "-".
            label (str, optional): Label for legend. Defaults to "".
            cumulative (bool, optional): Cumulative or differential representation. Defaults to False.
        """        
        if not cumulative:
            plt.plot(self.L_mid, self.f, label=label, color=color, ls=ls)
        else:
            plt.plot(self.L_mid, np.cumsum(self.f), label=label, color=color, ls=ls)

        plt.xlabel(f"$L$ [$\mu$m]")

        if not cumulative:
            plt.ylabel(r"$f$ [m$^{-3}\mu$m$^{-1}$]")
        else:
            plt.ylabel(r"$f$ [m$^{-3}\mu$m$^{-1}$]")

        plt.xlim([0, self.L_max])

    def computeMoments(self, imax=5):
        """Compute moments of the distribution

        Args:
            imax (int, optional): Maximum degree of moment to be calculated. Defaults to 5.
        """        
        moments = np.zeros([imax])
        for i in range(imax):
            moments[i] = np.nansum((self.f * self.L_mid**i) * self.dL)

        self.moments = moments

    def save(self, save_name):
        """Save population to a file.

        Args:
            save_name (str): file name. Defaults to "".
        """        
        PSSD_dict = self.__dict__
        for key, val in PSSD_dict.items():
            PSSD_dict[key] = convertToJsonFriendly(val)

        with open(f"{save_name}.json", "w") as f:
            json.dump(PSSD_dict, f)

    def extend_grid(self, n):
        """Extend grid with zeros towards larger sizes

        Args:
            n (int): Number of cells to be added.
        """        
        self.nbins += n
        self.L_max += n*self.dL
        self.f = np.concatenate([self.f, np.zeros(n)])
        self.L_bounds = np.arange(0, self.L_max + self.dL, self.dL)  # [um]
        self.L_mid = np.mean(
            [self.L_bounds[:-1], self.L_bounds[1:]], axis=0
        )  # [um]
        

if __name__ == "__main__":
    # Examples of use

    # Generate populations
    pop1 = PSD(mu=50, sigma=10, dL=1, L_max=500)
    print(pop1)
    pop2 = PSD(mu=250, sigma=50, skewness=6, dist_shape="skewed", dL=1, L_max=500)
    pop3 = PSD(
        mu=[100, 150],
        sigma=[30, 15],
        skewness=[4, 1],
        weights=[0.5, 1],
        dist_shape="multiple",
        dL=1,
        L_max=500,
    )

    # Plot populations
    pop1.plot()
    pop2.plot(color="r")
    pop3.plot(color="g")
    plt.savefig("test.png")

    # Save and load populations
    pop1.save("ini_pop")
    loaded_pop = PSD(load_file="ini_pop")
    print(pop1)
