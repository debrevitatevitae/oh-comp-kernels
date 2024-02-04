import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ohqk.data import LabelledDataset
from ohqk.kta_classical import KernelTargetAlignmentLoss, rbf_kernel
from ohqk.project_directories import GRAPHICS_DIR
from ohqk.train import train
from ohqk.utils import relabel_to_m1p1, running_average_filter

plt.rcParams["text.usetex"] = True
# make the y axis invisible
plt.rcParams["ytick.left"] = False
plt.rcParams["ytick.labelleft"] = False

if __name__ == "__main__":

    # initialize pseudo-random number generators
    np.random.seed(42)
    torch.manual_seed(42)

    # load the first two classes of the iris dataset
    X, y = load_iris(return_X_y=True)
    X = X[y < 2]
    y = relabel_to_m1p1(y[y < 2])

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ds = LabelledDataset(X, y)

    # set up the optimization configuration
    num_epochs = 100
    num_checkpoints = 50
    batch_size = 50
    lr = 1e-1
    gamma = 10 * torch.rand(1)  # initial gamma
    gamma.requires_grad = True

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam([gamma], lr)
    loss_function = KernelTargetAlignmentLoss(rbf_kernel)

    print("initial gamma:", gamma.item())
    # train the model
    trained_gamma, losses = train(
        gamma,
        loss_function,
        opt,
        num_epochs,
        dl,
        num_checkpoints=num_checkpoints,
    )
    print("trained gamma:", trained_gamma.item())

    # plot the losses
    smooth_losses = running_average_filter(losses, factor=0.6)
    plt.plot([-s for s in smooth_losses])  # negative sign for kta
    plt.xlabel("epoch")
    plt.ylabel("KTA")
    plt.savefig(GRAPHICS_DIR / "rbf_kta_opt_iris.pdf")
