# -*- coding: utf-8 -*-

from typing import Iterable, Union
import numpy as np

import pytrol.util.graphprocessor as gp


class Network:
    def __init__(self, graph: np.ndarray, edges_to_vertices: np.ndarray,
                 edges_lgts: np.ndarray, edge_activations,
                 vertices: dict, edges: dict, locations: np.ndarray):

        r"""The network, i.e. the graph, to patrol.

        Args:
            graph (np.ndarray):
            edges_to_vertices (np.ndarray):
            edges_lgts (np.ndarray):
            edge_activations:
            vertices (dict):
            edges (dict):
            locations (np.ndarray):
        """
        self.graph = graph
        self.edges_to_vertices = edges_to_vertices

        self.edges_lgts = edges_lgts

        self.max_units_of_edge = self.edges_lgts.max()

        self.vertices = vertices

        self.edges = edges

        self.locations = locations

        self.edge_activations = np.array(edge_activations, dtype=np.int16)

        # Neighbours, distances and paths

        self.ngbrs = gp.build_tc_neighbours(self.graph)

        self.v_dists, self.v_paths = gp.fw_distances(graph, edges_lgts)

        self.paths = gp.v_to_v_tc_paths(graph, edges_lgts,
                                          edges_to_vertices, self.v_paths)

        self.min_dist, self.max_dist = gp.min_and_max_dists(self.v_dists)

    def target(self, pos: Union[int, Iterable], e: int = -1) -> int:
        r"""The heading target.

        Args:
            pos (int | Iterable): the current position from which the target is
                required, can be a node id (int) or a position vector (3D
                vector)
            e (int): the current edge

        """

        if len(pos) == 3:
            s = pos[0]
            e = pos[1]
        else:
            s = pos

        if e < 0:
            raise ValueError("Value -1 forbidden. Edge's id must be higher "
                             "than -1 to determine a target")

        return gp.target(s, e, self.edges_to_vertices)

    def edge(self, s: Union[Iterable, int], t: int = -1) -> int:
        r"""The edge corresponding to `s` if `s` is a position vecteur,
        or `{s, t}` if `s` and `t` are node ids.

        Args:
            s (Iterable | int): the position vector or source node
            t: the target node if `s` is a node id
        """

        if t != -1:
            return gp.edge(s, t, self.graph)
        else:
            return self.edge(s[0], s[1])

    """"
    def dist(self, pos1: tuple, pos2: tuple) -> int:
        if pos1[1] == -1:
            pos1 = (pos1[0], self.graph[pos1[0]][self.graph[pos1[0]] > -1][0],
                    pos1[2])
        if pos2[1] == -1:
            pos2 = (pos2[0], self.graph[pos2[0]][self.graph[pos2[0]] > -1][0],
                    pos2[2])

        return self.dists[pos1[0]][pos1[1]][pos1[2]][pos2[0]][pos2[1]][pos2[2]]
    """

    def v_dist(self, pos1: Iterable, pos2: Iterable) -> int:
        r"""Distance between `pos1` and `pos2`.

        Args:
            pos1:
            pos2:
        """
        return self.v_dists[pos1[0]][pos2[0]]

    def eucl_dist(self, pos1: Iterable, pos2: Iterable) -> float:
        r"""Euclidean distance between `pos1` and `pos2`.

        Args:
            pos1:
            pos2:
        """

        vec1 = np.array([self.locations[pos1[0]][0]
                         - self.locations[self.target(pos1)][0],
                         self.locations[pos1[0]][1]
                         - self.locations[self.target(pos1)][1],
                         self.locations[pos1[0]][2]
                         - self.locations[self.target(pos1)][2]]) \
            if pos1[1] > -1 \
            else np.zeros(3)

        vec2 = np.array([self.locations[pos2[0]][0]
                        - self.locations[self.target(pos2)][0],
                        self.locations[pos2[0]][1]
                        - self.locations[self.target(pos2)][1],
                        self.locations[pos2[0]][2]
                        - self.locations[self.target(pos2)][2]]) \
            if pos2[1] > -1 \
            else np.zeros(3)

        unit1 = pos1[2] if pos1[2] != -1 else 0
        unit2 = pos2[2] if pos2[2] != -1 else 0

        coords1 = self.locations[pos1[0]] \
                  + (unit1 / self.edges_lgts[pos1[1]]) * vec1
        coords2 = self.locations[pos2[0]] \
                  + (unit2 / self.edges_lgts[pos2[1]]) * vec2

        return np.linalg.norm(coords1 - coords2)

    def path(self, pos1: Iterable, pos2: Iterable) -> list:
        r"""Shortest path between `pos1` and `pos2`.

        Args:
            pos1:
            pos2:
        """

        if pos1[1] == -1 and pos1[2] != 0 \
                or pos2[1] == -1 and pos2[2] != 0:
            raise ValueError(
                "A vector of the the 3D space of positions with an edge "
                "coordinate (2nd coordinate) valued to -1 cannot have a unit "
                "coordinate non equal to 0.")

        if pos1[1] == -1:
            pos1 = (pos1[0], self.graph[pos1[0]][self.graph[pos1[0]] > -1][0],
                    pos1[2])
        else:
            if pos1[1] not in self.graph[pos1[0]]:
                raise ValueError("A vector of the the 3D space of positions "
                                 "cannot have an edge non connected to the "
                                 "vertex of its first coordinate.")
        if pos2[1] == -1:
            pos2 = (pos2[0], self.graph[pos2[0]][self.graph[pos2[0]] > -1][
                0], pos2[2])
        else:
            if pos2[1] not in self.graph[pos2[0]]:
                raise ValueError("A vector of the the 3D space of positions "
                                 "cannot have an edge non connected to the "
                                 "vertex of its first coordinate.")

        return self.paths[pos1[0]][pos2[0]]

    def neighbours(self, p: Iterable) -> list:
        r"""The neighbours of `p`

        Args:
            p: the position as a 3D vector
        Returns:
            The list of the neighbours of `p`.
        """
        return self.ngbrs[p[0]]
