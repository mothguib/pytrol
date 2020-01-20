# -*- coding: utf-8 -*-

r"""`randidlenest.py`: useful functions to perform random idleness estimation
"""
from typing import Iterable

import numpy as np


def constraint(b: float, m: float, iidl: int):
    r"""Computes the left-hand side of the constraint
    :math:`\sum^{iidl}_{i = 0} (i-m) b^i = 0`.

    Returns:
        The constraint.

    Args:
        b: the variable `b` such as :math:`p_i = a b^i`
        m: the estimated idleness
        iidl: the individual idleness
    """

    c = 0.0
    for i in range(iidl + 1):
        c += (i - m) * (b ** i)
    return c


def constraint_2(b: float, m: float, iidl: int) -> float:
    r"""Computes the left-hand side of the constraint
    :math:`\sum^{iidl}_{i = 0} (i-m) b^{iidl - i} = 0` resulting in expressing
    :math:`p_i` such as :math:`p_i = a b^{iidl - i}` in the previous constraint.

    Returns;
        The constraint.

    Args:
        b: the variable `b` such as $p_i = a b^{iidl - i}$
        m: the estimated idleness
        iidl: the individual idleness
    """

    c = 0.0
    for i in range(iidl + 1):
        c += (i - m) * (b ** (iidl - i))

    return c


def bin_search_distribution(m: float, iidl) -> np.ndarray:
    r"""Computes a node's idleness probability distribution from its estimated
    idleness `m`

    Args:
        m: idleness estimated value
        iidl:

    Returns:
        The computed node's idleness probability distribution.
    """

    probs = np.zeros(iidl + 1)
    binf = 0.0
    bsup = 1.0

    if m == iidl / 2:
        for i in range(iidl + 1):
            probs[i] = 1.0 / (iidl + 1.0)
        return probs

    elif m < iidl / 2:
        cinf = constraint(binf, m, iidl)
        csup = constraint(bsup, m, iidl)

        for i in range(100):
            btry = (binf + bsup) / 2.0
            ctry = constraint(btry, m, iidl)

            if ctry < 0.0:
                binf = btry
                cinf = ctry
            else:
                bsup = btry
                csup = ctry

        b = binf - cinf * (bsup - binf) / (csup - cinf)
        a = (1.0 - b) / (1.0 - (b ** (iidl + 1)))

        for i in range(iidl + 1):
            probs[i] = a * b ** i

    elif m > iidl / 2:
        cinf = iidl - m
        csup = constraint_2(bsup, m, iidl)

        for i in range(100):
            btry = (binf + bsup) / 2.0
            ctry = constraint_2(btry, m, iidl)

            if ctry < 0.0:
                bsup = btry
                csup = ctry
            else:
                binf = btry
                cinf = ctry

        b = binf - cinf * (bsup - binf) / (csup - cinf)
        a = (1.0 - b) / (1.0 - (b ** (iidl + 1)))

        for i in range(iidl + 1):
            probs[i] = a * b ** (iidl - i)

    return probs


def expectation(support, probs):
    r"""
    Args:
        support: 1D iterable
        probs: 1D iterable, the probability distribution

    Returns:
        The expectation of `probs`.
    """

    e = 0

    for i in range(len(probs)):
        e += support[i] * probs[i]

    return e


def draw_rand_idls(estm_idls, iidls) -> np.ndarray:
    """
    Args:
        estm_idls:
        iidls:
    """
    ps = compute_distributions(estm_idls, iidls)

    # Random estimated idlenesses
    res = np.zeros(len(estm_idls), dtype=np.int16)

    for i, p in enumerate(ps):
        res[i] = np.random.choice(len(p), p=p)

    return res


def compute_distributions(ms: Iterable, iidls: Iterable) -> list:
    r"""Computes the probability distribution of the idleness values of a node
    from its estimated idleness `m`.

    Args:
        ms: estimated idlenesses, one per node
        iidls: vector of individual idlenesses
    """

    nb_nodes = len(ms)
    probs = []

    for i in range(nb_nodes):
        probs.append(
            compute_distribution(ms[i], iidls[i]).tolist())

    return probs


def compute_distribution(m: float, iidl) -> np.ndarray:
    r"""
    Args:
        m (float):
        iidl:
    """
    return approximate_distribution(m, iidl)
    # return bin_search_distribution(m, iidl)


def approximate_distribution(m: float, iidl):
    """
    Args:
        m (float):
        iidl:
    """
    probs = np.zeros(iidl + 1)

    # The uniform distribution
    probs_u = np.ones(iidl + 1) * (1.0 / (iidl + 1.0))

    if m == iidl / 2:
        return probs_u

    elif m < iidl / 2:
        b = m / (1 + m)
        a = (1.0 - b) / (1.0 - (b ** (iidl + 1)))
        for i in range(iidl + 1):
            probs[i] = a * b ** i
    elif m > iidl / 2:
        b = (iidl - m) / (1 + iidl - m)
        a = (1.0 - b) / (1.0 - (b ** (iidl + 1)))
        for i in range(iidl + 1):
            probs[i] = a * b ** (iidl - i)

    # Expectation of the uniform distribution
    m_u = iidl / 2

    m_e = expectation(list(range(iidl + 1)), probs)
    alpha = (m - m_u) / (m_e - m_u)
    alpha = 1 if alpha > 1 else alpha

    # print("alpha: ", alpha, ", 1 - alpha: ", 1 - alpha, ", iidl: ", iidl,
    # ", m: ", m, ", m_u: ", m_u, ", m_e: ", m_e)
    probs = alpha * probs + (1 - alpha) * probs_u

    return probs
