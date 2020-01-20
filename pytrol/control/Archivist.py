import json
import os
import pickle
import numpy as np

from pytrol.util import misc as misc
from pytrol.control.Communicating import Communicating
from pytrol.model.Data import Data
from pytrol.util.MetricComputer import MetricComputer
from pytrol.util.net.Connection import Connection


class Archivist(Communicating):
    r"""The archivist logs all the relevant data of the simulation: agent
    positions and idlenesses. It is a *communicating* insofar as it needs to
    communicate with agents and Ananke to acquire the data to log.

    Args:
        cnt: the connection used by the archivist to communicate
        log_path: the path of the file where to log the simulation
        duration: the duration of the simulation
        trace_agent: `True` if the archivist logs also the agents' individual
            idlenesses
        trace_agents_estms: `True` if the archivist logs also the estimated
            idlenesses of every agent, in the case of patrolling strategies
            using estimators
    """

    def __init__(self, cnt: Connection, log_path: str, duration: int,
                 trace_agent: bool, trace_agents_estms: bool):

        Communicating.__init__(self, cnt)

        self.duration = duration
        self.log_path = log_path
        self.trace_agents = trace_agent
        self.trace_agents_estms = trace_agents_estms

        self.t = 0
        self.agts_poss = []
        self.real_idlss = []
        self.idlss_agts = None
        self.idlss_estm_agts = None

        # Metric computer
        self.m_cptr = MetricComputer(duration)

    def run(self):
        print("Archivist: starting.")

        while not self.stop_working:
            while self.t < self.duration:
                self.receive()
                while len(self.messages) != 0:
                    m = self.messages.popleft()
                    data = pickle.loads(m)

                    self.log(data[Data.Cycle], data[Data.Agts_pos],
                             data[Data.Idls])
                self.t += 1

            self.save_log()
            self.stop_working = True

        print("Archivist stops to work.")

    def log(self, t: int, agts_pos: list, real_idls: list,
            idls_agts: list = None, agts_estm_idls: list = None):
        r"""Logs data of the simulation run for period `t` .

        When `t = self.duration` , the last position that corresponds in fact
        to the move of the last period is logged. Each agent will have
        therefore `T + 1` positions.

        Args:
            t (int):
            agts_pos (list):
            real_idls (list):
            idls_agts (list):
            agts_estm_idls (list):
        """

        self.t = t
        self.agts_poss += [agts_pos]

        # if t > 0:
        self.real_idlss += [real_idls]

        if self.trace_agents:
            for i, agt_seq in enumerate(self.idlss_agts):
                agt_seq.append(idls_agts[i])

        if self.trace_agents_estms:
            for i, agt_seq in enumerate(self.idlss_estm_agts):
                agt_seq.append(agts_estm_idls[i])

        self.m_cptr.get_intvls(np.array(self.real_idlss, dtype=np.int16))

    def save_log(self):
        # Trick to compute the last intervals: creation of a duplicate 2D list
        # of `self.real_idlss` to which we had a last list of zeros
        # corresponding to the reset idlenesses after the last time step,
        # this in order to compute the last intervals
        last_idlss = list(self.real_idlss)
        last_idlss += [[0] * len(self.real_idlss[0])]
        self.m_cptr.get_intvls(np.array(last_idlss, dtype=np.int16))

        # Computation of the simulation execution's metrics
        self.m_cptr.compute_metrics(np.array(self.real_idlss, dtype=np.int16))

        directory = os.path.dirname(self.log_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.log_path, 'w') as s:
            json.dump([[self.agts_poss, self.real_idlss],
                       self.m_cptr.metrics.tolist()], s)

        if self.trace_agents:
            with open(self.log_path.replace(".log", ".log.agts"), 'w') as s:
                json.dump(self.idlss_agts, s)

        if self.trace_agents_estms:
            with open(self.log_path.replace(".log", ".log.agts.est"),
                      'w') as s:
                json.dump(self.idlss_estm_agts, s)

        print("{}: Evaluation criteria:".format(misc.timestamp()))
        print(self.m_cptr.metrics, "\n")

    def prepare_idlss_agts(self, nagts: int):
        r"""
        Args:
            nagts (int):
        """
        self.idlss_agts = [[] for _ in range(nagts)] if self.trace_agents \
            else None

        self.idlss_estm_agts = [[] for _ in range(nagts)] if \
            self.trace_agents_estms else None
