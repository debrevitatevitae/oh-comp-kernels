import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from ohqk.data import StrainsStressesLabels
from ohqk.project_directories import GRAPHICS_DIR, RAW_DATA_DIR

plt.rcParams["text.usetex"] = True
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12

if __name__ == "__main__":

    np.random.seed(42)

    file_name = "Ea10000_Eb0_Es0.txt"

    ds = StrainsStressesLabels(
        RAW_DATA_DIR / file_name, labeling_threshold=0.85
    )

    dl = DataLoader(ds, batch_size=200, shuffle=True)

    eps, sigma, label = next(iter(dl))
    sns.scatterplot(x=eps[:, 0], y=sigma[:, 0], hue=label, palette="coolwarm")
    plt.xlabel(r"$\varepsilon [-]$")
    plt.ylabel(r"$\sigma$ [MPa]")
    plt.legend(["failed", "intact"])  # Add legend labels
    plt.savefig(GRAPHICS_DIR / "labelled_eps_sigma.pdf")
