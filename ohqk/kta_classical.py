import torch
import torch.nn as nn


def rbf_kernel(x, y, gamma):
    diff = x - y
    norm_squared = torch.dot(diff, diff)
    return torch.exp(-gamma * norm_squared)


def square_kernel_matrix(X, kernel_function):
    N = X.shape[0]
    K = torch.eye(N)

    for i in range(N - 1):
        for j in range(i + 1, N):
            # Compute kernel value and fill in both (i, j) and (j, i) entries
            K_i_j = kernel_function(X[i], X[j])
            K[i, j] = K_i_j
            K[j, i] = K_i_j

    return K


def frobenius_ip(A, B):
    return torch.sum(A * B)


class KernelTargetAlignmentLoss(nn.Module):
    def __init__(self, kernel_function):
        super(KernelTargetAlignmentLoss, self).__init__()
        self.kernel_function = kernel_function

    def forward(self, X, target, params):
        # Compute kernel and target kernel
        N = X.shape[0]

        def kf(x, y):
            return self.kernel_function(x, y, params)

        K = square_kernel_matrix(X, kf)
        K_target = torch.outer(target, target)

        # Compute the alignment loss
        kta = frobenius_ip(K, K_target) / (N * torch.sqrt(frobenius_ip(K, K)))

        return -kta
