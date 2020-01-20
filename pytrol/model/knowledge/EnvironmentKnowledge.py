# -*- coding: utf-8 -*-

import numpy as np

from pytrol.model.network.Network import Network


class EnvironmentKnowledge:
    r"""Data regarding the knowledge of environment. In MAP environment =
    topology (graph) + society.

    Args:
        ntw: the network, i.e. the topology
        idls: the individual idlenesses
        speeds: the speeds for each edge
        agts_pos: a `numpy.ndarray` containing the known positions of the
            other agents. A position is defined as a 3-element `[<vertex>,
            <edge>, <unit of edge>]`. The default values of <edge> and
            <unit> are -1 and 0, respectively, i.e. agent is not travelling
            any <edge>
        nagts: number of agents; special agents, such as a coordinator for
            example, are not taken into account
        t_nagts: total number of agents, taking into account the coordinator
            for example, if present
    """

    def __init__(self, ntw: Network, idls: np.ndarray, speeds: dict,
                 agts_pos: np.ndarray, ntw_name: str = '', nagts: int = -1,
                 t_nagts: int = -1):

        r"""
        Args:
            ntw (Network):
            idls (np.ndarray):
            speeds (dict):
            agts_pos (np.ndarray):
            ntw_name (str):
            nagts (int):
            t_nagts (int):
        """
        self.ntw_name = ntw_name
        self.nagts = nagts
        self.t_nagts = t_nagts

        # Network
        self.ntw = ntw
        # Individual idleness
        self.idls = idls
        # Shared idleness
        self.shared_idls = idls
        self.speeds = speeds

        # Time
        self.t = 0

        # Society

        #
        # Individual perception of the others agents' position
        self.agts_pos = agts_pos
        # self.dists_to_agts = self.ntw.dists_to_agts(self.agt_id,
        # self.ntw, self.agts_pos)

        self.vertices_to_agents = \
            np.ones(len(self.ntw.graph), np.int16) * (-1)
        self.edges_to_agents = \
            np.ones(len(self.ntw.graph), np.int16) * (-1)

        for i in range(len(self.agts_pos)):
            self.vertices_to_agents[self.agts_pos[i][0]] = i
            # TODO: set an array for probabilities of edges or other

    def tick(self):
        self.t += 1
        self.idls += 1
        self.shared_idls += 1

    def reset_idl(self, p: tuple):
        r"""Resets the idleness of the node corresponding to `p` , if `p`
        corresponds to a node.

        Args:
            p (tuple): idleness' position to reset to a 3D vector (v, e, u)
        """

        if p[1] == -1:
            self.idls[p[0]] = 0
            self.shared_idls[p[0]] = 0

    def target(self, agt_id: int):
        r"""
        Args:
            agt_id (int):
        """
        return self.ntw.target(self.agts_pos[agt_id])
