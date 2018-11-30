""" The Semi Non-negative Matrix Factorization (non-GPU accelerate) """
from numpy.testing import *
import numpy as np
import logging
import logging.config
import scipy.sparse
import scipy.io as spio
import matplotlib.pyplot as plt
import random
import dist

__all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps  # In real case, this eps should be larger.


class PyMFBase:
    """ PyMF base class used in (almost) all matrix factorization methods
    PyMF Base Class. Does nothing useful apart from providing some basic methods. """
    # some small value
    _EPS = _EPS

    def __init__(self, data, num_bases=4, **kwargs):
        """
        """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases

        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """ Returns the residual in % of the total amount of data
            Returns: residual (float)
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank approximation given by WH.
            Minimizing the Fnorm ist the most common optimization criterion for matrix factorization methods.
            Returns: frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self, 'H') and hasattr(self, 'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10 ** -4

    def _init_h(self):
        """ Initalize H to random values [0,1].
        """
        self.H = np.random.random((self._num_bases, self._num_samples)) + 0.2

    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

            # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, 'W') and compute_w:
            self._init_w()

        if not hasattr(self, 'H') and compute_h:
            self._init_h()

            # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('FN: %s (%s/%s)' % (self.ferr[i], i + 1, niter))
            else:
                self._logger.info('Iteration: (%s/%s)' % (i + 1, niter))

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class SNMF(PyMFBase):
    """
    SNMF(data, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        self.W = np.dot(W1, np.linalg.inv(W2))

    def _update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10 ** -9

        self.H *= np.sqrt(H1 / H2)


def _test():
    import doctest
    doctest.testmod()


class Kmeans(PyMFBase):
    """
    Kmeans(data, num_bases=4)

    K-means clustering. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to unary vectors, W
    is simply the mean over the corresponding samples in "data".

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())
    """

    def _init_h(self):
        # W has to be present for H to be initialized
        self.H = np.zeros((self._num_bases, self._num_samples))
        self._update_h()

    def _init_w(self):
        # set W to some random data samples
        sel = random.sample(range(self._num_samples), self._num_bases)

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(sel)]

    def _update_h(self):
        # and assign samples to the best matching centers
        self.assigned = dist.vq(self.W, self.data)
        self.H = np.zeros(self.H.shape)
        self.H[self.assigned, range(self._num_samples)] = 1.0

    def _update_w(self):
        for i in range(self._num_bases):
            # cast to bool to use H as an index for data
            idx = np.array(self.H[i, :], dtype=np.bool)
            n = np.sum(idx)
            if n > 0:
                self.W[:, i] = np.sum(self.data[:, idx], axis=1) / n


class TestNMF:

    mat = spio.loadmat('cereb_avrgDataStruct.mat')
    betaValueStructure = mat['T']
    betaValueMatrix = betaValueStructure[0, 0]
    print(betaValueMatrix['data'].shape)

    #data = np.random.rand(13, 25)
    data = betaValueMatrix['data']

    def test_snmf(self):

        for a in range(50):
            mdl = SNMF(self.data, num_bases=10)
            # nmf forms a cone in the input space, but it is unlikely to hit the
            # cone exactly.
            mdl.factorize(niter=70)
            residual_y = mdl.ferr
            plot_x = len(residual_y)
            print("ferr: ", residual_y, "\n")
            print(plot_x)
            plt.plot(residual_y)

        plt.ylabel('residual value by iteration')
        plt.show()

        plt.imshow(mdl.H, aspect='auto')
        plt.show()

        # the reconstruction quality should be close to perfect
        rec = mdl.frobenius_norm()
        #assert_array_almost_equal(0.0, rec, decimal=1)

        # and H is not allowed to have <0 values
        l = np.where(mdl.H < 0)[0]
        assert(len(l) == 0)

        print("Basis Vector F: ", mdl.W, "\n")
        print("Coefficients Matrix G: ", mdl.H, "\n")
        print("Residual Value: ", rec)


class TestKMeans:
    mat = spio.loadmat('cereb_avrgDataStruct.mat')
    betaValueStructure = mat['T']
    betaValueMatrix = betaValueStructure[0, 0]
    print(betaValueMatrix['data'].shape)

    # data = np.random.rand(13, 25)
    data = betaValueMatrix['data']

    def test_compute_w(self):
        mdl = Kmeans(self.data, num_bases=3)
        mdl.H = self.H
        mdl.factorize(niter=10, compute_h=False)
        assert_almost_equal(mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        mdl = Kmeans(self.data, num_bases=3)
        mdl.W = self.W
        mdl.factorize(niter=10, compute_w=False)
        assert_almost_equal(mdl.H, self.H, decimal=2)

    def test_compute_wh(self):
        mdl = Kmeans(self.data, num_bases=3)

        # init W to some reasonable values, otherwise the random init can
        # lead to a changed order of basis vectors.
        mdl.W = np.array([[0.1, 0.7, 0.4],
                          [0.3, 0.6, 0.5]])
        mdl.factorize(niter=10)
        assert_almost_equal(mdl.H, self.H, decimal=2)
        assert_almost_equal(mdl.W, self.W, decimal=2)

    def test_kmeans(self):
        mdl = Kmeans(self.data, num_bases=10)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=100)
        residual_y = mdl.ferr
        plot_x = len(residual_y)
        print("ferr: ", residual_y, "\n")
        print(plot_x)

        plt.plot(residual_y)
        plt.ylabel('residual value by iteration')
        plt.show()

        plt.imshow(mdl.H, aspect='auto')
        plt.show()

        # the reconstruction quality should be close to perfect
        rec = mdl.frobenius_norm()
        #assert_array_almost_equal(0.0, rec, decimal=1)

        # and H is not allowed to have <0 values
        l = np.where(mdl.H < 0)[0]
        assert(len(l) == 0)

        print("Basis Vector F: ", mdl.W, "\n")
        print("Coefficients Matrix G: ", mdl.H, "\n")
        print("Residual Value: ", rec)


def main():

    A = TestNMF()
    A.test_snmf()

    B = TestKMeans()
    B.test_kmeans()


if __name__ == '__main__':
    main()