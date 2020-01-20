# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC

import pytrol.util.graphprocessor as gp
from pytrol.control.agent.Agent import Agent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.model.action.GoingToAction import GoingToAction


# Heuristic Pathfinder Agent
class HPAgent(Agent, ABC):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 interaction: bool = True):
        """
        Args:
            id_ (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            variant (str):
            depth (float):
            interaction (bool):
        """

        Agent.__init__(self, id_=id_, original_id=original_id, env_knl=env_knl,
                       connection=connection, agts_addrs=agts_addrs,
                       variant=variant, depth=depth, interaction=interaction)

        # Estimated idlenesses: idlenesses estimated by any procedure or
        # simply real idlenesses
        self.estimated_idls = np.zeros(len(env_knl.idls), dtype=np.float32)

        # Previous estimated idlenesses: in some cases it is needed to retain
        # the previous estimated idlenesses
        self.prev_estimated_idls = np.zeros(len(env_knl.idls),
                                            dtype=np.float32)

        # Normalised estimated idlenesses: Pathfinder and Heuristic methods'
        # normalisation of idlenesses
        self.nrm_estimated_idls = np.zeros(len(env_knl.idls), dtype=np.float32)

        # Goal position's node id selected by the heuristic method for each
        # agent. Strategies wherefor agents select themselves their Heuristic
        # goal (the opposite of the coordinator-based strategies) will only
        # use `self.hr_agt_goal_pos[self.id]`.

        self.hr_agt_goal_pos = None
        self.set_hr_agt_goal_pos()

        self.r = float(variant)

    def set_hr_agt_goal_pos(self):
        """Populates the `self.hr_agt_goal_pos` list"""

        self.hr_agt_goal_pos = [(-1, -1, 0)] * self.env_knl.t_nagts
        self.hr_agt_goal_pos[self.id] = self.pos

    def strategy_decide(self):
        """Strategy-specific decision stage. By default the agent makes the
        decision for itself: this method is designed for an agent which makes
        the next goal node choice directly by itself from its estimated
        idlenesses unlike a coordinator which make the decision for coordinated
        agent
        """

        super().strategy_decide()

        self.prev_estimated_idls = self.estimated_idls

        self.estimated_idls = self.estimate_idls()

        if self.hr_agt_goal_pos[self.id] == self.pos:
            # The heuristic method is applied in `process_idls`
            prcssd_idls = self.process_idls(self.estimated_idls,
                                            self.pos)
            self.hr_agt_goal_pos[self.id] = \
                self.select_goal_pos(prcssd_idls, self.hr_agt_goal_pos,
                                     self.pos)

        # The Pathfinder path is determined at each decision step i.e. at
        # each node owing to the fact that idleness evolving continuously.
        pathfinder_path = self.pathfinder(idls=self.estimated_idls,
                                          edges_lgts=self. \
                                          env_knl.ntw.edges_lgts,
                                          agt_pos=self.pos,
                                          goal_pos=self. \
                                          hr_agt_goal_pos[self.id],
                                          r=self.r,
                                          graph=self.env_knl.ntw.graph)

        next_pos = self.pathfinder_next_pos(pathfinder_path)

        self.PLAN. \
            append(GoingToAction(next_pos))

    @staticmethod
    def select_goal_pos(prcssd_idls, hr_agt_goal_pos: list, agt_pos: tuple) \
            -> tuple:
        """Selects the next position from the processed idlenesses passed as
        argument and updates the list of Heuristic goal positions
        `hr_agt_goal_pos` pertaining to this selection

        procedure and any other further methods :type prcssd_idls: iterable
        :param hr_agt_goal_pos: Goal position's node id selected by the
        heuristic method for each agent. :type hr_agt_goal_pos: list :param
        agt_pos: position of the agent requesting for a goal node :type agt_pos:
        tuple :return: the goal position vector :rtype: tuple

        Args:
            prcssd_idls: the processed idleness with the Heuristic
            hr_agt_goal_pos (list):
            agt_pos (tuple):
        """

        # "Mountainous" processed idleness vector, because the nodes already
        # assigned as goal to an agent will be assigned the infinity as
        # its idleness level to avoid to be selected one more time
        mount_prcssed_idls = np.copy(prcssd_idls)

        for i in range(len(hr_agt_goal_pos)):
            # `hr_agt_goal_pos[i][0]` corresponds to the goal node id of the
            #  agent `i`
            if hr_agt_goal_pos[i][0] > -1:
                mount_prcssed_idls[hr_agt_goal_pos[i][0]] = np.infty
            else:
                mount_prcssed_idls[agt_pos[0]] = np.infty

        return np.argmin(mount_prcssed_idls), -1, 0

    def estimate_idls(self) -> np.ndarray:
        """Returns an estimation of the idleness vector if needed. By default it
        returns the individual idleness `self.env_knl.idls`
        """

        return self.env_knl.idls

    def process_idls(self, idls, agt_pos) -> np.ndarray:
        """Processing to perform over the current idlenesses `idls` . The
        procedure Heuristic is applied in this method and any other processing
        can be performed upon the Heuristic idlenesses.

        Args:
            idls (iterable):
            agt_pos (3D tuple): the requesting agent position

        Returns:
            np.ndarray: the idleness vector transformed by the heuristic method
        """

        min_idl = min(idls)
        max_idl = max(idls)

        min_ttg = self.env_knl.ntw.min_dist
        max_ttg = self.env_knl.ntw.max_dist

        # Times-to-go between the agent's position and every node
        ttgs = self.env_knl.ntw.v_dists[agt_pos[0]]

        hr_idls, self.nrm_estimated_idls = self.apply_heuristic(idls, ttgs,
                                                                self.pos[0],
                                                                self.r,
                                                                max_idl,
                                                                min_idl,
                                                                max_ttg,
                                                                min_ttg)

        return hr_idls

    @classmethod
    def apply_heuristic(cls, idls,
                        ttgs,
                        current_vtx: int,
                        r,
                        max_idl: int = -1,
                        min_idl: int = -1,
                        max_ttg: int = -1,
                        min_ttg: int = -1
                        ) -> (np.ndarray, np.ndarray):
        r"""
        Args:
            idls:
            ttgs: times-to-go between the agent's position `current_vtx`
            current_vtx (int):
            r:
            max_idl (int):
            min_idl (int):
            max_ttg (int):
            min_ttg (int):

        Returns:
            (heuristicified_idls, nrm_estimated_idls):
            `heuristicified_idls` is the idleness vector transformed by the
            heuristic method with `heuristicified_idls[current_vtx]` having the
            infinity as value given that the current node shall not be selected,
            and `nrm_estimated_idls` the original idleness vector normalised.
        """

        # "Heuristicified" idlenesses
        heuristicified_idls, nrm_estimated_idls = \
            cls.heuristicify_idls(idls, ttgs, r, max_idl, min_idl, max_ttg,
                                  min_ttg)

        heuristicified_idls[current_vtx] = np.inf  # To avoid to select the
        # current node as the next goal node

        return heuristicified_idls, nrm_estimated_idls

    @classmethod
    def heuristicify_idls(cls,
                          idls,
                          ttgs,
                          r,
                          max_idl: int = -1,
                          min_idl: int = -1,
                          max_ttg: int = -1,
                          min_ttg: int = -1) -> (np.ndarray, np.ndarray):
        r"""
        :math:`r * I(v,t) * (1-r) * d(v0 , v)` where v is a node, and v0 the
        agent's node, for each `v` in `V` .

        Args:
            idls (iterable):
            ttgs (iterable): times-to-go between the agent's position and every
                node
            r:
            max_idl (int):
            min_idl (int):
            max_ttg (int):
            min_ttg (int):

        Returns:
            (np.ndarray, np.ndarray): the "heuristicified" idlenesses and the
            normalised idlenesses
        """

        if max_idl == -1:
            max_idl = idls.max()
        if min_idl == -1:
            min_idl = idls.min()
        if max_ttg == -1:
            max_ttg = ttgs.max()
        if min_ttg == -1:
            min_ttg = ttgs.min()

        nrm_estimated_idls = cls.normalised_idls(idls, max_idl, min_idl)

        return r * nrm_estimated_idls + \
                (1 - r) * cls.normalised_times_to_go(ttgs, max_ttg, min_ttg), \
                nrm_estimated_idls

    @staticmethod
    def normalised_idls(idls, max_idl: int = -1, min_idl: int = -1) \
            -> np.ndarray:
        r"""
        Args:
            idls (iterable):
            max_idl (int):
            min_idl (int):

        Returns:
            np.ndarray:
        """

        idls = np.array(idls, dtype=np.float32)

        if max_idl == -1:
            max_idl = idls.max()
        if min_idl == -1:
            min_idl = idls.min()

        if max_idl - min_idl == 0:
            return np.zeros(len(idls), dtype=np.float32)

        return (max_idl - idls) / (max_idl - min_idl)

    @staticmethod
    def normalised_times_to_go(ttgs, max_ttg: int = -1, min_ttg: int = -1):
        r"""Normalised times-to-go in [0,1] standing for the normalised
        length of the distance. The closer to 1, the larger the distance `d`
        is.

        Args:
            ttgs (iterable): times-to-go
            max_ttg (int):
            min_ttg (int):

        Returns:
            np.ndarray:
        """

        ttgs = np.array(ttgs, dtype=np.float32)

        if max_ttg == -1:
            max_ttg = ttgs.max()
        if min_ttg == -1:
            min_ttg = ttgs.min()

        if max_ttg - min_ttg == 0:
            return np.zeros(len(ttgs), dtype=np.float32)

        return (ttgs - min_ttg) / (max_ttg - min_ttg)

    @classmethod
    def pathfinder(cls,
                   idls,
                   edges_lgts,
                   agt_pos,
                   goal_pos,
                   r: float,
                   graph) \
            -> list:
        r"""
        Args:
            idls (iterable): the idleness vector, normalised or not
            edges_lgts (iterable):
            agt_pos (int or tuple): the requesting agent's position
            goal_pos (int or tuple): the goal agent's position
            r (float):
            graph (iterable):

        Returns:
            list: a list of tuples standing for the 3D position path
        """

        # Test of whether `idls` is a normalised idleness vector or not
        if np.array(idls, dtype=np.float32).sum() > len(idls):
            nrm_idls = cls.normalised_idls(idls)
        else:
            nrm_idls = idls

        pf_dist, pf_path = cls.pf_dijkstra_dist_to_target(agt_pos,
                                                          goal_pos,
                                                          graph,
                                                          edges_lgts,
                                                          nrm_idls,
                                                          r)

        # DBG
        # print("DBG: pathfinder:", '\n'
        #      "\tagt_pos: ", agt_pos, '\n',
        #      "\tgoal_pos: ", goal_pos, '\n',
        #      "\tpf_path: ", pf_path)

        # Pathfinder path
        return gp.tc_path(agt_pos,
                               goal_pos,
                               pf_path,
                               graph,
                               edges_lgts)

    @staticmethod
    def pathfinder_next_pos(pthf_path: list) -> tuple:
        r"""The next node where to go according to the planned Pathfinder
        path represented as a list of position vectors.

        Args:
            pthf_path (list): Pathfinder's shortest path passed to the method
        """

        # Among the 3D position path, it is the next node which is selected
        # because the agent applies the path selection process on each
        # crossed node and not only on each goal node
        for i in range(1, len(pthf_path)):
            if pthf_path[i][1] == -1:
                return pthf_path[i]

    @staticmethod
    def pf_dijkstra_dist_to_target(s, t, graph, edges_lgts, nrm_idls, r) \
            -> tuple:
        r"""Pathfinder's Dijkstra distance and path from the source `s` to the
        target `t`

        Args:
            s: the source node
            t: the target node
            graph: 2D array representing the graph whose `s` and `t` are nodes
            edges_lgts: 1D array returning the edge length from its id
            nrm_idls: normalised idleness from the Heuristic of Pathfinder
                procedure
            r: weighting factor controlling the impact of both features
                in the transformation of the edge lengths. `r = 0.5` implies
                that idlenesses and edge lengths are equivalently regarded
        """

        if type(s) != int:
            if len(s) != 3:
                raise ValueError("s must be an iterable of length 3.")
            s = int(s[0])
        if type(t) != int:
            if len(t) != 3:
                raise ValueError("t must be an iterable of length 3.")
            t = int(t[0])

        n = len(graph)

        ngbs = gp.build_neighbours(graph)

        # Minimum edge length
        min_el = edges_lgts.min()

        # Maximum edge length
        max_el = edges_lgts.max()

        fathers = [-1] * len(graph)
        fathers[s] = s  # By convention, the father of `s` is `s` itself

        dists = np.ones(n, dtype=np.float32) * np.inf
        dists[s] = 0

        for v in range(n):
            if v in ngbs[s]:
                dists[v] = r * nrm_idls[v] \
                           + (1 - r) * (edges_lgts[gp.edge(s, v, graph)] -
                                        min_el) / (max_el - min_el)
                fathers[v] = s

        open_ = set(range(n))
        open_.remove(s)

        closed = {s}

        path = np.empty(0, dtype=np.int16)

        while len(open_) != 0:

            i = np.argmin(
                dists[list(open_)])  # `dists[list(open_)]` corresponds
            #  to the list of distances from `s` only for the nodes populating
            # `open_`

            i = list(open_)[i]  # retrieval of the true node id given that from
            # previous instruction, before assignment,`i` corresponds here
            # to the ith node in `open_`
            open_.remove(i)
            closed.add(i)

            if not (dists[i] != np.inf and dists[i] >= dists[t]):
                for j in open_ & set(ngbs[i]):
                    # Edge length transformed by the method Pathfinder
                    pf_edge_lgt_i_j = \
                        r * nrm_idls[j] + \
                        (1 - r) * (edges_lgts[
                                       gp.edge(i, j, graph)] - min_el) / \
                        (max_el - min_el)

                    if dists[i] + pf_edge_lgt_i_j < dists[j]:
                        dists[j] = dists[i] + pf_edge_lgt_i_j
                        fathers[j] = i

        i = t
        path = np.append(path, i)
        while i != s:
            i = fathers[i]
            path = np.append(path, i)

        return dists[t], np.flip(path, axis=0)
