# -*- coding: utf-8 -*-

import pickle

import networkx as nx
import numpy as np

import pytrol.util.misc as misc
from pytrol.control.Communicating import Communicating
from pytrol.control.agent.Agent import Agent
from pytrol.control.agent.MAPTrainerModelAgent import MAPTrainerModelAgent
from pytrol.model.Data import Data
from pytrol.model.action.Action import Action
from pytrol.model.action.Actions import Actions
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.model.network.Network import Network
from pytrol.util.SimPreprocessor import SimPreprocessor
from pytrol.util.net.Connection import Connection
from pytrol.util.net.SimulatedConnection import SimulatedConnection
from pytrol.model.AgentTypes import AgentTypes


class Ananke(Communicating):
    r"""`Ananke` is the core of the simulator, i.e. the structure that
    concretely handles the simulation running. It is also a *communicating*.
    In the ancient Greek religion, the goddess Ananke is a personification
    of inevitability, compulsion and necessity. She is often depicted as
    holding a spindle.

    The life cycle of `Ananke` starts with the mission's initialisation which
    takes place in `Ananke.__init__` , where it loads the graph, all
    information relative to the current mission, as well as the agents.
    Then, in `Ananke.run` the main loop of simulation over the time steps is
    executed. There is as many iterations in this loop as the mission's
    duration `duration` . This loop stands for the running of simulation: at
    each period the strategy of agents simulated herein is deployed. More
    precisely, at each iteration Ananke executes the main procedure of the
    strategy by calling, for every agent, the methods described above that
    constitutes their life cycle.

    Args:
        cnt: the connection used by Ananke to communicate
        exec_path: path of the execution file defining a particular
            instanciation of a MAP configuration. An instanciation of a MAP
            configuration consists in an initial position for every patrolling
            agent
        archivist_addr: archivist's address
        duration: duration of the simulation run
        depth: range of communication
        ntw_name: name of the network, i.e. the graph, to patrol
        nagts: number of agents
        variant: simulated strategy's variant
        trace_agent: `True` if the archivist logs also the individual idlenesses
            of the agents, `False` otherwise
        trace_agents_estms: `True` if the archivist logs also the estimated
            idlenesses of every agent, in the case of patrolling strategies
            using
        estimators,:
        interaction: `True` if interaction is enabled, `False` otherwise
    """

    def __init__(self,
                 cnt: Connection,
                 exec_path: str,
                 archivist_addr,
                 duration: int,
                 depth: float,
                 ntw_name: str = '',
                 nagts: int = 0,
                 variant: str = '',
                 trace_agents: bool = False,
                 trace_agents_estms: bool = False,
                 interaction: bool = False):

        Communicating.__init__(self, cnt)

        self.duration = duration
        self.t = 0
        self.archivist_addr = archivist_addr
        self.depth = depth
        self.trace_agents_estms = trace_agents_estms
        self.trace_agents = trace_agents

        # I. Configuration

        # I.1. Environment

        # I.1.1. Network initialisation
        exec_path = exec_path

        self.graph, fl_edges_lgts, vertices, edges, edges_to_vertices, \
        locations, edge_activations, idls, socs, speeds_societies, \
        agts_pos_societies, socs_to_ids = \
            SimPreprocessor.load_run(exec_path)

        edges_lgts = np.round(fl_edges_lgts)

        for s in speeds_societies:
            if s["id"] != "InactiveSociety":
                self.speeds = s["speeds"]

        self.network = Network(self.graph, edges_to_vertices, edges_lgts,
                               edge_activations, vertices, edges, locations)

        # I.2. Agents

        # I.2.1 Total number of agents calculation

        # Total number of agents takes also into account the coordinator in the
        # context of coordinated or multi-society strategies
        self.t_nagts = 0
        for s in socs:
            self.t_nagts += len(s['agents'])

        # I.2.2 Agent generation
        self.agts_addrs, self.agts_pos = self. \
            load_societies(socs=socs,
                           idls=idls,
                           agts_pos_societies=agts_pos_societies,
                           nagts=nagts,
                           t_nagts=self.t_nagts,
                           depth=depth,
                           ntw_name=ntw_name,
                           variant=variant,
                           interaction=interaction)

        # Raising an exception if forbidden values are set
        if len(self.agts_pos[self.agts_pos[:, 2] < 0]) > 0:
            raise ValueError("Negative unit of edge forbidden")

        edges_with_agts = self.agts_pos[self.agts_pos[:, 1] < 0]
        if len(edges_with_agts[edges_with_agts[:, 2] > 0]) > 0:
            raise ValueError("Forbidden value(s) for position("
                             "s) ", edges_with_agts[edges_with_agts[:, 2] > 0],
                             "of `agts_pos`. Some positions have units that "
                             "are not equal to zero whereas they correspond "
                             "to positions not in an edge (edge `-1`).")

        self.pcvd_agt_matrix = np.zeros(self.agts_pos.shape)

        # I.3 Idlenesses

        # I.3.1 Real idlenesses
        self.idls = idls

        # I.3.2 Idlenesses of agents
        self.idls_agts = np.zeros((len(self.agts_pos), len(self.graph)),
                                  dtype=np.int16)

        self.agts_estm_idls = np.zeros((len(self.agts_pos), len(self.graph)),
                                       dtype=np.float32)

    def run(self):
        r"""Runs the main simulation loop over the time steps. There is as many
        iterations in this loop as the mission's duration `self.duration` . This
        loop stands for the running of simulation: at each period the strategy
        of agents simulated herein is deployed. More precisely, at each
        iteration Ananke the main procedure of the strategy is executed by
        calling, for every agent, the methods that constitutes their life cycle.
        """

        print("{}: Ananke: starts".format(misc.timestamp()))

        self.start_archivist()
        print("{}: Archivist: starts".format(misc.timestamp()))

        while not self.stop_working:
            while self.t < self.duration:
                # Creation of the Completed List which is a dictionary
                c_list = np.zeros(self.t_nagts, dtype=np.uint8)

                if self.trace_agents:
                    self.collect_idls_agts()

                # Archiving the current period
                self.send_to_archivist()

                # TDP
                # print("Cycle ", self.t, ':\n\n')
                # print('Agents position:', self.agts_pos, '\n')

                # Preparation of agent position perception,
                # `pcp_mat`: perception matrix
                pcp_mat = self.compute_pos_pcp(
                    self.compute_range_pos_pcp_matrix())

                for a in self.agts_addrs:
                    a.prepare()

                for a in self.agts_addrs:
                    # Generating and flowing perceptions to agents

                    # Agents' perception
                    agts_pos = self.gen_prcved_agts_pos(a, pcp_mat)

                    # Subgraph' perception if needed
                    # subgraph = self.build_subgraph(a)

                    # Agent's own position perception if needed
                    # pos = self.build_current_vertex(a)

                    # Perceiving
                    a.perceive(agts_pos)

                while np.sum(c_list) != self.t_nagts:
                    # Main procedure of agents

                    # 1. Communications when needed at the beginning of the
                    # strategy
                    for a in self.agts_addrs:
                        a.communicate()
                    # 2. Analysing received messages as long as the current
                    # cycle is not complete (`c_list` is not empty)
                    for a in self.agts_addrs:
                        a.analyse()
                    # 3. Deciding
                    for a in self.agts_addrs:
                        a.decide()
                    # 4. Acting as long as the current cycle is not complete (
                    # `c_list` is not empty)
                    for a in self.agts_addrs:
                        action = a.act()  # After each
                        # procedure `act`, the agent must activate its
                        # attribute `ac`
                        self.process_action(action, a)
                    # 5. Checking the cycle termination
                    for a in self.agts_addrs:
                        if len(a.messages) != 0:
                            a.c = True
                    for a in self.agts_addrs:
                        # Testing whether a change must be done in the
                        # Completed List
                        if a.ac and not a.c:  # and c_list[a.id] == 0:
                            c_list[a.id] = 1
                        # Agents reactivated if needed, for example to
                        # process a received message
                        elif not a.ac or a.c:  # and c_list[a.id] == 1:
                            c_list[a.id] = 0

                # TDP
                # print("Real idlenesses ", self.idls)

                for a in self.agts_addrs:
                    a.update_knowledge()

                for a in self.agts_addrs:
                    a.post_process()

                # Environment updating for the next time step
                self.update_environment()

            '''
            # Last call of `send_to_archivist` to log the last idlenesses
            # and next position resulting from the last move of agents
            # self.send_to_archivist()
            '''

            # The simulation is over, other communicatings are then stopped
            '''
            if self.display_addr is not None:
                self.send("stop_working", self.display_addr)
                print("Display: stops to work")
            '''

            print("{}: Latest idlenesses:\n {}" \
                  .format(misc.timestamp(), self.idls))

            # The log is saved before stopping the archivist
            self.stop_archivist()

            self.stop_agts()

            self.stop_working = True

            print("{}: Ananke: stops to work".format(misc.timestamp()))

    @staticmethod
    def compute_pos_pcp(am: np.ndarray) -> np.ndarray:
        r"""The binary relation that represents whether agents are within range
        is called *neighbouring relation*, and its transitive closure, which
        represents whether agents are transitively within range using one
        another as relay, is called *connection relation*. With connection
        relation, two agents are regarded as connected iff. there exists a path
        between them.

        `compute_pos_pcp` converts the matrix of the neighbouring relation
        into that of the connection relation.

        Args:
            am (np.ndarray): an `np.ndarray` representing the matrix of the
                neighbouring relation

        Returns:
            m: an `np.ndarray` representing the matrix of the connection
                relation
        """

        m = np.zeros(am.shape, dtype=np.uint8)

        graph = nx.from_numpy_matrix(am)
        subgraphs = nx.connected_component_subgraphs(graph)

        for g in subgraphs:
            # Submatrix
            sm = Ananke.expand_nx_subgraphs(g, graph.number_of_nodes())
            for n in g.nodes():
                m[n] = sm[n]

        # TDP
        # print("Ananke.compute_pos_pcp:\n",
        #      "am:\n", am,
        #      "m:\n", m)

        return m

    def compute_range_pos_pcp_matrix(self) -> np.ndarray:
        r"""Computes the adjacent matrix of agents that perceive each others
        according to the range of perception `depth` .

        Returns:
            m: an `np.ndarray` standing for the adjacent matrix of agents
                that perceive each others according to the range of perception
                `depth`
        """

        # if self.agts_addrs[0].name == "coordinator":

        nagts = len(self.agts_pos)

        # `m`: the adjacent matrix of agents perceiving each others
        # according to the range of perception `depth`
        m = np.zeros((nagts, nagts), dtype=np.uint8)

        for i, p_1 in enumerate(self.agts_pos):
            for j, p_2 in enumerate(self.agts_pos):
                if self.network.eucl_dist(p_1, p_2) <= self.depth:
                    m[i][j] = 1

        return m

    @staticmethod
    def expand_nx_subgraphs(g: nx.Graph, nvtcs: int) -> np.ndarray:
        r"""Expands a NetworkX subgraph so that it have the same size as its
        parent graph. The new components are populated with 0.

        Args:
            g (nx.Graph): subgraph
            nvtcs (int): number of vertices in the parent graph

        Returns:
            an `np.ndarray` representing subgraph connection adjacent matrix
        """

        m = np.zeros((nvtcs, nvtcs), dtype=np.uint8)

        k = 1
        for u in g.nodes:
            m[u][u] = 1
            for v in list(g.nodes)[k:]:
                m[u][v] = 1
                m[v][u] = 1
            k += 1

        return m

    def gen_prcved_agts_pos(self, a: Agent, pcp_mat: np.ndarray) -> np.ndarray:
        r"""Generates all agents' perceived positions for the current agent
        `a` , except the perceiving agent itself

        Args:
            a (Agent): current agent
            pcp_mat (np.ndarray): matrix of perception

        Returns:
            a_agts_pos: an `np.ndarray` populated with the perceived
                positions of the other agents
        """

        # Perceived agents' positions
        p_agts_pos = np.ones(self.agts_pos.shape, dtype=np.int16) * -1

        for i in np.where(pcp_mat[a.id] == 1):
            p_agts_pos[i] = self.agts_pos[i]

        # The agent does not perceive itself
        p_agts_pos[a.id] = np.ones(3, dtype=np.int16) * -1

        # TDP
        # print("Ananke.gen_prcved_agts_pos: ",
        #      "Agent id: " + str(a.id) + " p_agts_pos:\n",
        #      p_agts_pos, '\n')

        return p_agts_pos

    def collect_idls_agts(self):
        for i in range(len(self.agts_addrs)):
            self.idls_agts[i] = self.agts_addrs[i].env_knl.idls

        if self.trace_agents_estms:
            for i in range(len(self.agts_addrs)):
                self.agts_estm_idls[i] = self.agts_addrs[i].model_estm_idls

    def build_subgraph(self, a: Agent):
        r"""
        Args:
            a (Agent):
        """
        return self.graph

    def build_current_vertex(self, a: Agent):
        r"""
        Args:
            a (Agent):
        """
        return self.agts_pos[a.id]

    def process_action(self, action: Action, a: Agent):
        r"""
        Args:
            action (Action):
            a (Agent):
        """
        if action.type == Actions.Moving_to:
            if not np.array_equal(self.agts_pos[a.id], action.frm):
                print(self.agts_pos[a.id], action.frm)
                raise ValueError(
                    "Inconsistent start vertex in this MovingTo "
                    "Action")

            self.agts_pos[a.id] = action.to

    def update_environment(self):
        self.t += 1
        self.idls += 1

        for a in self.agts_addrs:
            self.idls = np.where(a.env_knl.idls < self.idls,
                                 a.env_knl.idls, self.idls)

    def load_societies(self,
                       socs: dict,
                       idls: np.ndarray,
                       agts_pos_societies: dict,
                       nagts: int,
                       t_nagts: int,
                       depth: float,
                       ntw_name: str = '',
                       variant: str = '',
                       interaction: bool = True) -> (list, np.ndarray):

        r"""
        Args:
            socs (dict):
            idls (np.ndarray):
            agts_pos_societies (dict):
            nagts (int):
            t_nagts (int):
            depth (float):
            ntw_name (str):
            variant (str):
            interaction (bool):
        """
        agts_addrs = []
        agts_pos = np.empty([0, 3], dtype=np.int16)

        # I.2.1. Agent Position
        # Initial agents positions
        for s in agts_pos_societies:
            if s["id"] != "InactiveSociety":
                agts_pos = np.append(agts_pos, s["agts_pos"], axis=0)

        # TODO: Filling env_knl.vertices_to_agents and
        # env_knl.edges_to_agents with -1 before updating it

        for s in socs:
            if s["id"] != "InactiveSociety":
                for a in s['agents']:
                    # Check if tracing agents' idlenesses' estimates is
                    # performed only with MAPTrainerModelAgent agents
                    if self.trace_agents_estms and \
                            AgentTypes.id_to_class_name[a["type"]] == \
                            MAPTrainerModelAgent:
                        raise ValueError("Tracing agents' idlenesses' "
                                         "estimates is only available for "
                                         "MAPTrainerModelAgent agents")

                    # Here, all agents and Ananke share the same network object
                    #  in order to accelerate the initialisation
                    env_knl = EnvironmentKnowledge(ntw=self.network,
                                                   idls=idls.copy(),
                                                   speeds=self.speeds.copy(),
                                                   agts_pos=agts_pos.copy(),
                                                   ntw_name=ntw_name,
                                                   nagts=nagts,
                                                   t_nagts=t_nagts)
                    cnt = SimulatedConnection()

                    # Agents
                    # Initializing agents, retrieving their address to put them
                    #  into a list to send it to the archivist. In order to
                    #  keep idlenesses and agents position personal, passing
                    #  graph and agent perception by copy except for the
                    #  archivist

                    agt = AgentTypes. \
                        id_to_class_name[a["type"]](id_=a["id"],
                                                    original_id=
                                                    a["original_id"],
                                                    env_knl=env_knl,
                                                    connection=cnt,
                                                    agts_addrs=agts_addrs,
                                                    variant=variant,
                                                    depth=depth,
                                                    interaction=interaction)

                    agts_addrs.append(agt)

                # Agents' address is set for each agent after instantiating all
                #  agents in order to have retrieved all their addresses
                for agt in agts_addrs:
                    agt.set_agts_addrs(agts_addrs)

        return agts_addrs, agts_pos

    def start_archivist(self):
        r"""Tasks to perform before starting to archive."""

        if self.trace_agents:
            self.archivist_addr.prepare_idlss_agts(len(self.agts_addrs))

    def send_to_archivist(self, agts_pos: np.ndarray = None,
                          idls: np.ndarray = None):
        r"""
        Args:
            agts_pos (np.ndarray):
            idls (np.ndarray):
        """
        if agts_pos is None or idls is None:
            self.archivist_addr.log(self.t, self.agts_pos.tolist(),
                                    self.idls.tolist(),
                                    self.idls_agts.tolist(),
                                    self.agts_estm_idls.tolist())
        else:
            self.send(pickle.dumps({Data.Cycle: self.t,
                                    Data.Agts_pos: agts_pos.tolist(),
                                    Data.Idls: idls.tolist()}),
                      self.archivist_addr)

    def stop_archivist(self):
        self.archivist_addr.save_log()

    def stop_agts(self):
        for a in self.agts_addrs:
            a.stop()
