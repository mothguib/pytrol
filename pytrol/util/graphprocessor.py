# -*- coding: utf-8 -*-

from typing import Iterable, Union
import numpy as np


def retrieve_neighbours(v: int, graph: Iterable) -> np.ndarray:
    r"""Retrieves the neighbours of `v`.

    Args:
        v:
        graph:
    """
    t = np.nonzero(graph[v] + 1)
    # + 1 because the indexing of vertices starts from 0. So adding 1
    # enables to get only the indices whose the values are not equal to -1

    return t[0]


def build_neighbours(g: Iterable) -> list:
    r"""
    Args:
        g:
    """
    l = []

    for i in range(len(g)):
        # Neighbours of `i`
        i_neighbours = retrieve_neighbours(i, g)
        l += [i_neighbours.tolist()]

    return l


def build_tc_neighbours(g: Iterable) -> list:
    r"""
    Args:
        g: the graph

    Returns:
        The list of neighbour of each node of `g` in the form of 3D positions
    """

    tc_ngbrs = []
    ngbrs = build_neighbours(g)

    for ns in ngbrs:
        tc_n = []
        for v in ns:
            tc_n += [(v, -1, 0)]
        tc_ngbrs += [tc_n]

    return tc_ngbrs


# Floyd-Warshall Algorithm
def fw_distances(g, edges_lgts: Iterable) -> (np.ndarray, list):
    r"""Distances between the nodes of `g` computed by using the Floyd-Warshall
    Algorithm.

    Args:
        g: 2D array representing the graph
        edges_lgts: 1D array representing the edge length for each edge id

    Returns:
        A 2-uplet consisting of a `numpy.ndarray` of all distances between the
        nodes and a list of all the paths.
    """

    # `dists` is a |V| × |V| array of minimum distances initialised to
    # infinity
    dists = np.ones(g.shape, dtype=np.float32) * np.inf
    paths = [[[i] for _ in range(len(g))] for i in range(len(g))]

    # for each vertex i
    for i in range(len(g)):
        for j in range(len(g)):
            if g[i][j] != -1:
                dists[i][j] = edges_lgts[g[i][j]]
                paths[i][j] += [j]

        paths[i][i] += [i]
        dists[i][i] = 0

    for k in range(len(g)):
        for i in range(len(g)):
            for j in range(len(g)):
                if dists[i][j] > dists[i][k] + dists[k][j]:
                    dists[i][j] = dists[i][k] + dists[k][j]
                    paths[i][j] = paths[i][k] + paths[k][j][1:]

    # np.place(dists, dists==np.inf, -1)

    return dists, paths


def dijkstra_dist(s, graph, edges_lgts: Iterable) -> (np.ndarray, list):
    r"""Dijkstra distance and path from the source `s` to each node of
    `graph` .

    Args:
        s: the source node
        graph: 2D array representing the graph
        edges_lgts: 1D array representing the edge length for each edge id
    """

    if type(s) != int:
        if len(s) != 3:
            raise ValueError("s must be an iterable of length 3.")
        s = s[0]

    n = len(graph)

    ngbs = build_neighbours(np.array(graph, dtype=np.int16))

    fathers = [-1] * len(graph)
    fathers[s] = s  # By convention, the father of `s` is `s` itself

    dists = np.ones(n, dtype=np.float32) * np.inf
    dists[s] = 0

    for v in range(n):
        if v in ngbs[s]:
            dists[v] = edges_lgts[edge(s, v, graph)]
            fathers[v] = s

    open_ = set(range(n))
    open_.remove(s)

    closed = {s}

    paths = [[]] * n

    while len(open_) != 0:

        i = np.argmin(dists[list(open_)])  # `dists[list(open_)]` corresponds
        #  to the list of distances from `s` only for the nodes populating
        # `open_`

        i = list(open_)[i]  # retrieval of the true node id given that from
        # previous instruction, `i` corresponds here, before assignment,
        # to the ith node in `open_`
        open_.remove(i)
        closed.add(i)

        for j in open_ & set(ngbs[i]):
            if dists[i] + edges_lgts[edge(i, j, graph)] < dists[j]:
                dists[j] = dists[i] + edges_lgts[edge(i, j, graph)]
                fathers[j] = i

    for i in range(n):
        j = i
        paths[i] += [j]
        while j != s:
            j = fathers[j]
            paths[i] += [j]

        paths[i] = list(reversed(paths[i]))

    return dists, paths


def dijkstra_dist_to_target(s: int, t: int, graph: Iterable,
                            edges_lgts: Iterable) \
        -> (np.ndarray, np.ndarray):
    r"""Dijkstra distance and path from the source `s` to the target `t`

    Args:
        s: the source node
        t: the target node
        graph: 2D array representing the graph
        edges_lgts: 1D array representing the edge length for each edge id
    """

    if type(s) != int:
        if len(s) != 3:
            raise ValueError("s must be an iterable of length 3.")
        s = s[0]
    if type(t) != int:
        if len(t) != 3:
            raise ValueError("t must be an iterable of length 3.")
        t = t[0]

    n = len(graph)

    ngbs = build_neighbours(np.array(graph, dtype=np.int16))

    fathers = [-1] * len(graph)
    fathers[s] = s  # By convention, the father of `s` is `s` itself

    dists = np.ones(n, dtype=np.float32) * np.inf
    dists[s] = 0

    for v in range(n):
        if v in ngbs[s]:
            dists[v] = edges_lgts[edge(s, v, graph)]
            fathers[v] = s

    open_ = set(range(n))
    open_.remove(s)

    closed = {s}

    path = np.empty(0, dtype=np.int16)

    while len(open_) != 0:

        i = np.argmin(dists[list(open_)])  # `dists[list(open_)]` corresponds
        #  to the list of distances from `s` only for the nodes populating
        # `open_`

        i = list(open_)[i]  # retrieval of the true node id given that from
        # previous instruction, `i` corresponds here, before assignment,
        # to the ith node in `open_`
        open_.remove(i)
        closed.add(i)

        if not (dists[i] != np.inf and dists[i] >= dists[t]):
            for j in open_ & set(ngbs[i]):
                if dists[i] + edges_lgts[edge(i, j, graph)] < dists[j]:
                    dists[j] = dists[i] + edges_lgts[edge(i, j, graph)]
                    fathers[j] = i

    i = t
    path = np.append(path, i)
    while i != s:
        i = fathers[i]
        path = np.append(path, i)

    return dists[t], np.flip(path, axis=0)


def tc_path(pos1: Union[int, Iterable],
            pos2: Union[int, Iterable],
            v_path: Iterable,
            graph: Iterable,
            edges_lgts: Iterable) -> list:

    r"""Computes the *3-coordinate path* between `pos1` and `pos2` . A
    3-coordinate path is one where the units of all the edges of the path are
    considered, as well as nodes, and represented as *position vector*, i.e. in
    the form of a 3-coordinated vector `(<node>, <edge>, <unit>)`

    Args:
        pos1: the source node
        pos2: the target node
        v_path: iterable of the vertices' path to convert into a list of
        graph: 2D iterable representing the graph
        edges_lgts: 1D iterable representing the edge length for each edge id

    Returns:
        The 3-coordinate path of the successive positions between pos1 and pos2.
        It works only with the positions representing nodes and not with those
        standing for units of edge.
    """

    if type(pos1) == int:
        pos1 = (pos1, -1, 0)
    else:
        if len(pos1) != 3:
            raise ValueError("s must be an iterable of length 3.")

    if type(pos2) == int:
        pos2 = (pos2, -1, 0)
    else:
        if len(pos2) != 3:
            raise ValueError("s must be an iterable of length 3.")

    path = [pos1]

    # Current pos
    pos = pos1

    if pos1[0] != pos2[0]:
        for i in range(len(v_path) - 1):
            # Current edge
            ce = graph[pos[0]][v_path[i + 1]]

            pos = (pos[0], ce, pos[2])

            for u in range(pos[2] + 1, int(edges_lgts[ce])):
                path += [(pos[0], ce, u)]

            pos = (v_path[i + 1], -1, 0)

            path += [pos]
    else:
        if pos1[1] == pos2[1]:
            path += [pos2]
        else:
            path += [pos2]
            # TODO: fixing the issue corresponding to the above case where
            #   pos1[0] == pos2[0]

    return path


def target(s: int, e: int, edges_to_vertices: Iterable) -> int:
    r"""Returns the target of `s` when standing upon `e` .

    Args:
        s:
        e:
        edges_to_vertices:
    """

    # e = graph[s][graph[s] > -1][0] if e == -1 else e

    if e == -1:
        raise ValueError("Value -1 forbidden. Edge's id must be higher than "
                         "-1")

    return edges_to_vertices[e][edges_to_vertices[e] != s][0]


def edge(s: Union[int, Iterable], t: Union[int, Iterable], graph: Iterable) \
        -> int:
    r"""Returns the edge id, if it exists, corresponding to nodes `s` and `t` ,
    otherwise -1.

    Args:
        s:
        t:
        graph:
    """

    if type(s) != int:
        if len(s) != 3:
            raise ValueError("s must be an iterable of length 3.")
        s = s[0]
    if type(t) != int:
        if len(t) != 3:
            raise ValueError("t must be an iterable of length 3.")
        t = t[0]

    return graph[s][t]


def dists_to_agts(agt_id: int,
                  agts_pos: Iterable,
                  edges_lgts: Iterable,
                  edges_to_vertices: Iterable,
                  v_dists: Iterable):
    r"""Computes the distances between the current agent `agt_id` 's
    position and all the nodes of the graph/network.

    Args:
        agt_id:
        agts_pos:
        edges_lgts:
        edges_to_vertices:
        v_dists:
    """

    agts_pos = np.array(agts_pos, dtype=np.int16)

    if agts_pos[agt_id][1] == -1 and agts_pos[agt_id][2] != 0:
        raise ValueError(
            "A vector of the the 3D space of positions with an edge "
            "coordinate (2nd coordinate) valued to -1 cannot have a unit "
            "coordinate (3rd coordinate) higher than 0.")

    # Distances of agt_id to the whole vertices
    agt_id_to_vertices = v_dists[agts_pos[agt_id][0]]

    # If the agent agt_id is travelling an edge
    if agts_pos[agt_id][1] > -1 and agts_pos[agt_id][2] > 0:
        # Distance to the others agents computed by crossing the source
        # of its edge
        dists_from_source = v_dists[agts_pos[agt_id][0]] + \
                            agts_pos[agt_id][2]

        # Distance to the others agents computed by crossing the target
        # of its edge
        dists_from_target = v_dists[target(agts_pos[agt_id][0],
                                           agts_pos[agt_id][1],
                                           edges_to_vertices)] + edges_lgts[
                                agts_pos[agt_id][1]] - agts_pos[agt_id][2]

        # Distance to the others agents dists_to_agts is finally the min of
        #  both
        agt_id_to_vertices = \
            dists_from_source * (dists_from_source < dists_from_target) \
            + dists_from_target * (1 - (dists_from_source <
                                        dists_from_target))
        # Another way to carry out the previous operation:
        # agt_id_to_vertices =\
        # np.where( (dists_from_source < dists_from_target),
        #           dists_from_source,
        #           dists_from_target )

    # Taking the 1st value (index 0) of the second axis (axis 1) of
    # agts_pos, to get the distance of each agent from the current agent
    #  `agt_id`
    dis_to_agts = agt_id_to_vertices[agts_pos.take(0, axis=1)]
    # Another way to carry out the previous operation:
    # dists_to_agts = agt_id_to_vertices[agts_pos[:, 0]]

    # At that stage dists_to_agts represents the distance from
    # agt_id to others agents regardless of its position (travelling
    # an edge or on a vertex)

    agts_on_edge = np.where(agts_pos[:, 1] > -1)[0]
    # Others ways to carry out the previous operation:
    # agts_on_edge = np.where(agts_pos.take(1, axis=1) > -1)[0]

    for a in agts_on_edge:
        if agts_pos[a][1] == agts_pos[agt_id][1]:
            dis_to_agts[a] = np.absolute(agts_pos[a][2] - agts_pos[
                agt_id][2])
        else:
            dis_to_agts[a] = \
                np.minimum(agt_id_to_vertices[agts_pos[a][0]] + agts_pos[a][2],
                           agt_id_to_vertices[target(agts_pos[a][0],
                                                     agts_pos[a][1],
                                                     edges_to_vertices)] +
                           edges_lgts[agts_pos[a][1]] - agts_pos[a][2])

    return dists_to_agts


def v_to_v_dist(pos1: Union[int, Iterable], pos2: Union[int, Iterable],
                graph: Iterable, edges_lgts: Iterable,
                edges_to_vertices: Iterable, v_dists: Iterable) -> int:
    r"""Computes the distance between two nodes provided as position vectors,
    i.e. as 3D vectors. Regardless the positions passed through argument, only
    the vertex, i.e. the 1st coordinate, is taken into account.

    Args:
        pos1:
        pos2:
        graph:
        edges_lgts:
        edges_to_vertices:
        v_dists:

    Returns:
        The distance between `pos1[0]` and `pos2[0]`
    """

    if type(pos1) == int:
        pos1 = (pos1, -1, 0)
    else:
        if len(pos1) != 3:
            raise ValueError("s must be an iterable of length 3.")

    if type(pos2) == int:
        pos2 = (pos2, -1, 0)
    else:
        if len(pos2) != 3:
            raise ValueError("s must be an iterable of length 3.")

    if pos1[1] == -1 and pos1[2] != 0 \
            or pos2[1] == -1 and pos2[2] != 0:
        raise ValueError(
            "A vector of the the 3D space of positions with an edge "
            "coordinate (2nd coordinate) valued to -1 cannot have a unit "
            "coordinate higher than 0.")

    if pos1[1] not in graph[pos1[0]]:
        raise ValueError("A vector of the the 3D space of positions "
                         "cannot have an edge non connected to the "
                         "vertex of its first coordinate.")

    if pos2[1] not in graph[pos2[0]]:
        raise ValueError("A vector of the the 3D space of positions "
                         "cannot have an edge non connected to the "
                         "vertex of its first coordinate.")

    if pos1[2] == edges_lgts[pos1[1]]:
        pos1 = (target(pos1[0], pos1[1], edges_to_vertices), -1, 0)

    if pos2[2] == edges_lgts[pos2[1]]:
        pos2 = (target(pos2[0], pos2[1], edges_to_vertices), -1, 0)

    return v_dists[pos1[0]][pos2[0]]


def v_to_v_tc_path(pos1: Union[int, Iterable], pos2: Union[int, Iterable],
                   graph: Iterable, edges_lgts: Iterable,
                   edges_to_vertices: Iterable, v_paths: Iterable) -> list:
    r"""Computes the path between the two nodes `pos1` and `pos2` provided as
    position vectors, i.e. as 3-coordinate vectors. Regardless the positions
    passed through argument, only the vertex, i.e. the 1st coordinate, is taken
    into account.

    Args:
        pos1: 3-D tuple or int
        pos2: 3-D tuple or int
        graph (np.ndarray):
        edges_lgts:
        edges_to_vertices:
        v_paths (list):

    Returns:
        The path between `pos1[0]` and `pos2[0]`
    """

    if pos1[1] == -1 and pos1[2] != 0 \
            or pos2[1] == -1 and pos2[2] != 0:
        raise ValueError(
            "A vector of the the 3D space of positions with an edge "
            "coordinate (2nd coordinate) valued to -1 cannot have a unit "
            "coordinate higher than 0.")

    if pos1[1] != -1:
        if pos1[1] not in graph[pos1[0]]:
            raise ValueError("A vector of the the 3D space of positions "
                             "cannot have an edge non connected to the "
                             "vertex of its first coordinate.")

        if pos2[1] not in graph[pos2[0]]:
            raise ValueError("A vector of the the 3D space of positions "
                             "cannot have an edge non connected to the "
                             "vertex of its first coordinate.")

    if pos1[2] == edges_lgts[pos1[1]]:
        pos1 = (target(pos1[0], pos1[1], edges_to_vertices), -1, 0)

    if pos2[2] == edges_lgts[pos2[1]]:
        pos2 = (target(pos2[0], pos2[1], edges_to_vertices), -1, 0)

    return tc_path(pos1, pos2, v_paths[pos1[0]][pos2[0]], graph,
                   edges_lgts)


def v_to_v_tc_paths(graph: Iterable, edges_lgts: Iterable,
                    edges_to_vertices: Iterable, v_paths: Iterable) \
        -> list:
    r"""Computes the 3-coordinate paths between all the nodes of graph
    `graph` .
    Regardless the positions passed through argument, only the vertex, i.e. the
    1st coordinate, is taken into account.

    Args:
        graph:
        edges_lgts:
        edges_to_vertices:
        v_paths: a list of the 3-coordinate paths between .
    """

    tc_paths = [[[] for _ in graph] for _ in graph]

    for i in range(len(graph)):
        for j in range(len(graph)):
            tc_paths[i][j] += v_to_v_tc_path((i, -1, 0), (j, -1, 0), graph,
                                             edges_lgts, edges_to_vertices,
                                             v_paths)

    return tc_paths


def tc_u_to_u_dist(pos1: Union[Iterable, int], pos2: Union[Iterable, int],
                   graph: Iterable, edges_lgts: Iterable,
                   edges_to_vertices: Iterable,
                   v_dists: Iterable) -> np.ndarray:
    r"""Computes the 3-coordinate path between all positions `pos1` and
    `pos2`, whether they are edge units or nodes.

    Args:
        pos1: 3-D tuple or int
        pos2: 3-D tuple or int
        graph: the graph to patrol
        edges_lgts: length of edges of `graph`
        edges_to_vertices: mapping between the edge ids and vertex ids
        v_dists: distances between vertices from their ids

    Returns:
        The distance between the position vectors `pos1` and `pos2` . If `pos1`
        is a position located on an edge leading to the vertex of `pos2` (
        `pos2[0]` ) then the returned path will be empty.
    """

    # Handling of impossible values
    if pos1[1] == -1 and pos1[2] > 0 \
            or pos2[1] == -1 and pos2[2] > 0:
        raise ValueError(
            "A vector of the the 3D space of positions with an edge "
            "coordinate (2nd coordinate) valued to -1 cannot have a unit "
            "coordinate higher than 0.")

    if pos1[1] == -1:
        pos1 = (pos1[0], graph[pos1[0]][graph[pos1[0]] > -1][0], pos1[2])
    else:
        if pos1[1] not in graph[pos1[0]]:
            raise ValueError("A vector of the the 3D space of positions "
                             "cannot have an edge non connected to the "
                             "vertex of its first coordinate.")
    if pos2[1] == -1:
        pos2 = (pos2[0], graph[pos2[0]][graph[pos2[0]] > -1][0], pos2[2])
    else:
        if pos2[1] not in graph[pos2[0]]:
            raise ValueError("A vector of the the 3D space of positions "
                             "cannot have an edge non connected to the "
                             "vertex of its first coordinate.")

    # Handling of the case where the positions are on the same edge
    if pos1[1] == pos2[1]:
        # If they are directed in the same way
        if pos1[0] == pos2[0]:
            return np.absolute(pos1[2] - pos2[2])
        # Else
        else:
            return np.absolute(pos1[2] - edges_lgts[pos2[1]] + pos2[2])

    # Marginal distance from pos1's source
    m_pos1_s = pos1[2]
    # Marginal distance from pos1's target
    m_pos1_t = edges_lgts[pos1[1]] - pos1[2]
    # Marginal distance from pos2's source
    m_pos2_s = pos2[2]
    # Marginal distance from pos2's target
    m_pos2_t = edges_lgts[pos2[1]] - pos2[2]

    pos1_t = target(pos1[0], pos1[1], edges_to_vertices)
    pos2_t = target(pos2[0], pos2[1], edges_to_vertices)

    # Array of transient distances to minimise
    t_dists = np.array(
        [v_dists[pos1[0]][pos2[0]] + m_pos1_s + m_pos2_s,
         v_dists[pos1[0]][pos2_t] + m_pos1_s + m_pos2_t,
         v_dists[pos1_t][pos2[0]] + m_pos1_t + m_pos2_s,
         v_dists[pos1_t][pos2_t] + m_pos1_t + m_pos2_t],
        dtype=np.int16
    )

    i = np.argmin(t_dists)

    return t_dists[i]


# 3-coordinate ( (v, e, u) ) unit-to-unit distances
def tc_u_to_u_dists(graph: Iterable, edges_lgts: Iterable,
                    edges_to_vertices: Iterable, v_dists: Iterable) -> dict:
    r"""
    Args:
        graph:
        edges_lgts:
        edges_to_vertices:
        v_dists:
    """

    tc_dists = {}

    for i in range(len(graph)):
        tc_dists[i] = {}
        for e in graph[i][graph[i] > -1]:
            tc_dists[i][e] = {}
            for u in range(edges_lgts[e]):
                tc_dists[i][e][u] = {}
                for j in range(len(graph)):
                    tc_dists[i][e][u][j] = {}
                    for f in graph[j][graph[j] > -1]:
                        tc_dists[i][e][u][j][f] = {}
                        for v in range(edges_lgts[f]):
                            tc_dists[i][e][u][j][f][v] = {}

    for i in range(len(graph)):
        for e in graph[i][graph[i] > -1]:
            for u in range(edges_lgts[e]):
                for j in range(i, len(graph)):
                    for f in graph[j][graph[j] > -1]:
                        for v in range(edges_lgts[f]):
                            tc_dists[i][e][u][j][f][v] = tc_u_to_u_dist(
                                (i, e, u), (j, f, v), graph, edges_lgts,
                                edges_to_vertices, v_dists)
                            tc_dists[j][f][v][i][e][u] = tc_u_to_u_dist(
                                (i, e, u), (j, f, v), graph, edges_lgts,
                                edges_to_vertices, v_dists)

    return tc_dists


def min_and_max_dists(v_dists: Iterable):
    r"""Minimum and maximum distance among `v_dists` .

    Args:
        v_dists: a numpy ndarray containing the dists between all
            nodes

    Returns:
        The maximum and the minimum distance between two different nodes of the
        graph corresponding to `v_dists` .
    """
    max_d = -1
    min_d = -1

    for i in range(len(v_dists)):
        for j in range(len(v_dists)):
            if i != j:
                dist_i_j = v_dists[i][j]
                if dist_i_j > max_d or max_d == -1:
                    max_d = dist_i_j
                elif dist_i_j < min_d or min_d == -1:
                    min_d = dist_i_j

    return min_d, max_d
