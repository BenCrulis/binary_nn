"""
Adapted from https://github.com/romanpogodin/plausible-kernelized-bottleneck/blob/master/kernels.py
"""

import numpy as np
from scipy.spatial.distance import pdist


def estimate_hsic(a_matrix, b_matrix, mode='biased', normalize=False):
    """
    Estimates HSIC (if mode='biased') or pHSIC (if mode='plausible') between variables A and B.
    :param a_matrix:    torch.Tensor, a_matrix_ij = k(a_i,a_j), symmetric
    :param b_matrix:    torch.Tensor, b_matrix_ij = k(b_i,b_j), symmetric, must be the same size as a_matrix
    :param mode:        str, 'biased' (HSIC) or 'plausible' (pHSIC)
    :param normalize:   bool, use centered kernel alignment (biased HSIC only)
    :return: float, HSIC or pHSIC estimate
    """
    if mode == 'biased':
        a_vec = a_matrix.mean(axis=0)
        b_vec = b_matrix.mean(axis=0)

        normalization = 1.0
        if normalize:
            a_norm = (a_matrix * a_matrix).mean() - 2 * (a_vec * a_vec).mean() + a_vec.mean() * a_vec.mean()
            b_norm = (b_matrix * b_matrix).mean() - 2 * (b_vec * b_vec).mean() + b_vec.mean() * b_vec.mean()
            a_norm = 0.0 if a_norm < 0.0 else a_norm
            b_norm = 0.0 if b_norm < 0.0 else b_norm
            normalization = np.sqrt(a_norm * b_norm)
            if normalization <= 0.0:
                normalization = 1.0

        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return ((a_matrix * b_matrix).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean())/normalization
    if mode == 'plausible':
        if normalize:
            raise NotImplementedError("normalization is not available for pHSIC")
        # same as tr((A - mean(A))(B - mean(B)))/m^2
        return ((a_matrix - a_matrix.mean()) * b_matrix).mean()

    raise NotImplementedError('mode must be either biased or plausible, but %s was given' % mode)


def estimate_hsic_zy_objective(z, y, z_kernel, y_kernel, gamma=2.0, mode='biased', normalize=False):
    """
    Estimates the kernelized bottleneck objective between activations z and labels y.
    :param z:         torch.Tensor, activations, shape (batch_size, ...)
    :param y:         torch.Tensor, labels, shape (batch_size, ...)
    :param z_kernel:  Kernel, kernel to use for z
    :param y_kernel:  Kernel, kernel to use for y
    :param gamma:     float, balance parameter (float)
    :param mode:      str, 'biased' (HSIC) or 'plausible' (pHSIC)
    :return: float, HSIC(z,z) - gamma * HSIC(z, y) (or pHSIC, if mode='plausible')
    """
    z_matrix = z_kernel.compute(z)
    y_matrix = y_kernel.compute(y)
    return estimate_hsic(z_matrix, z_matrix, mode, normalize) - gamma * estimate_hsic(y_matrix, z_matrix, mode, normalize)


def estimate_hsic_ker(x, y, x_kernel, y_kernel, mode="biased", normalize=False):
    x_matrix = x_kernel.compute(x)
    y_matrix = y_kernel.compute(y)
    return estimate_hsic(x_matrix, y_matrix, mode, normalize)


def compute_linear_kernel(batch):
    """
    Computes the linear kernel between input vectors.
    :param batch: torch.Tensor, input vectors
    :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] (for flattened batch[i] and batch[j])
    """
    return batch.reshape(batch.shape[0], -1) @ batch.reshape(batch.shape[0], -1).T


def compute_pdist_matrix(batch, p=2.0):
    """
    Computes the matrix of pairwise distances w.r.t. p-norm
    :param batch: torch.Tensor, input vectors
    :param p:     float, norm parameter, such that ||x||p = (sum_i |x_i|^p)^(1/p)
    :return: numpy array, matrix A such that A_ij = ||batch[i] - batch[j]||_p (for flattened batch[i] and batch[j])
    """
    mat = np.zeros((batch.shape[0], batch.shape[0]))
    ind = np.triu_indices(batch.shape[0], k=1)
    mat[ind[0], ind[1]] = pdist(batch.reshape(batch.shape[0], -1)*1, metric="minkowski", p=p)

    return mat + mat.T


class Kernel:
    """
    Base class for different kernels.
    """
    def __init__(self):
        pass

    def compute(self, batch):
        raise NotImplementedError()


class LinearKernel(Kernel):
    """
    Linear kernel: k(a_i, a_j) = a_i^T a_j
    """
    def __init__(self):
        super().__init__()

    def compute(self, batch):
        """
        Computes the linear kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] (for flattened batch[i] and batch[j])
        """
        return compute_linear_kernel(batch)


class GaussianKernel(Kernel):
    """
    Gaussian kernel: k(a_i, a_j) = epx(-||a_i - a_j||^2 / (2 sigma^2))
    """
    def __init__(self, sigma=5.0):
        """
        :param sigma: float, Gaussian kernel sigma
        """
        super().__init__()
        self.sigma = sigma

    def compute(self, batch):
        """
        Computes the Gaussian kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = exp(-||batch[i] - batch[j]||^2 / (2 sigma^2))
            (for flattened batch[i] and batch[j])
        """
        return np.exp(-(compute_pdist_matrix(batch, p=2.0)) ** 2 / (2.0 * self.sigma ** 2))


class CosineSimilarityKernel(Kernel):
    """
    Cosine similarity kernel: k(a_i, a_j) = a_i^T a_j / (||a_i||_2 ||a_j||_2)
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6  # s.t. a_i / ||a_i||_2 == 0.0 if ||a_i||_2 == 0.0

    def compute(self, batch):
        """
        Computes the cosine similarity kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] / (||batch[i]||_2 ||batch[j]||_2)
            (for flattened batch[i] and batch[j])
        """
        normalization = np.linalg.norm(batch.reshape(batch.shape[0], -1), axis=-1)[:, None]
        normalization = normalization + self.eps * (normalization <= 0.0)
        return compute_linear_kernel(batch.reshape(batch.shape[0], -1) / normalization)


class HammingKernel(Kernel):
    def compute(self, batch):
        return 1.0 - compute_pdist_matrix(batch, p=1.0)/batch.shape[-1]