# -*- coding: utf-8 -*-

from typing import Iterable
import numpy as np

from pytrol.model.Metrics import Metrics


def compute_iav(idlss: Iterable):
    r"""Computes the average graph idleness.

    Args:
        idlss: 2D iterable of shape `(T, N)`, where `T` is the duration of
        the mission and `N` the number of nodes
    """

    return np.mean(np.mean(idlss))


class MetricComputer:
    r"""
    Args:
        cycles_nb:
        intvls:
    """
    def __init__(self, cycles_nb: int, intvls: np.ndarray = None):
        # Value of metrics at the time t
        self.metrics = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
        self.cycles_nb = cycles_nb

        # Numpy array of intervals
        if intvls is None:
            self.intvls = np.array([], dtype=np.int32)
        else:
            self.intvls = intvls

    def compute_metrics(self, idlss: np.ndarray):
        r"""Computes the metrics iteratively.

        Args:
            idlss:
        """

        self.metrics[Metrics.IAV] = compute_iav(idlss)
        self.metrics[Metrics.MI] = self.compute_mi()
        self.metrics[Metrics.MSI] = self.compute_msi()
        self.metrics[Metrics.MAXI] = self.compute_max_i()
        self.metrics[Metrics.VARI] = self.compute_var_i()
        self.metrics[Metrics.SDI] = np.sqrt(self.metrics[Metrics.VARI])

    def get_intvls(self, idlss: np.ndarray):
        r"""Retrieves the intervals from the list of idlenesses. This method is
        called from `t = 1` and not `t = 0`.

        Args:
            idlss:
        """

        # Previous latest idlenesses
        pl_idls = np.array(idlss[-2], dtype=np.int16) if len(idlss) > 1 \
            else np.ones(len(idlss[0]), dtype=np.int16) # if len(idlss) > 1
        # the penultimate list of idleness corresponding to the time step t
        # - 1 is retrieved, else idlenesses are set to 1 for cope with the
        # case where at t = 1 (first call of this method) new nodes are
        # already visited. This will enable to set the first interval for
        # these nodes.

        # Latest Idlenesses (indice -1 i.e. the last one)
        idls = np.array(idlss[-1], dtype=np.int16)

        # Indices of idleness reinitialised at the current time
        ris = np.where(idls == 0)[0]

        self.intvls = np.append(self.intvls, pl_idls[ris], axis=0)

    def compute_mi(self) -> float:
        r"""Computes the mean interval."""
        return np.sum(self.intvls) / len(self.intvls)

    def compute_msi(self) -> float:
        r"""Computes the mean square interval."""
        return np.sqrt(np.dot(self.intvls, self.intvls) / len(
            self.intvls))

    def compute_max_i(self) -> float:
        r"""Computes the maximum interval."""
        return np.max(self.intvls)

    def compute_var_i(self) -> float:
        r"""Computes the variance of interval."""
        mi = self.compute_mi()
        msi = self.compute_msi()

        return msi * msi - mi * mi
