import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

from ohqk.project_directories import GRAPHICS_DIR

plt.rcParams["text.usetex"] = True
# make the y axis invisible
plt.rcParams["ytick.left"] = False
plt.rcParams["ytick.labelleft"] = False

if __name__ == "__main__":
    x = np.linspace(-5, 5, 100).reshape(-1, 1)

    gamma_values = [0.1, 0.5, 1, 2, 5]

    # first plot: multiple gammas
    for gamma in gamma_values:
        kappa = rbf_kernel(x, np.zeros(shape=(1, 1)), gamma=gamma)

        plt.plot(x, kappa, label=rf"$\gamma={gamma}$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\kappa(x, 0)$")

    plt.legend()
    plt.savefig(GRAPHICS_DIR / "rbf_1d_gammas.pdf")

    # second plot: "optimal" gamma
    kappa_opt = rbf_kernel(x, np.zeros(shape=(1, 1)), gamma=0.5)
    plt.figure()
    plt.plot(x, kappa_opt, label=r"$\gamma=\gamma_{opt}$", color="C1")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\kappa(x, 0)$")
    plt.legend()
    plt.savefig(GRAPHICS_DIR / "rbf_1d_gamma_opt.pdf")
