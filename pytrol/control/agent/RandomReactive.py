# -*- coding: utf-8 -*-

import numpy as np
from pytrol.control.agent.Agent import Agent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.model.action.GoingToAction import GoingToAction


class RandomReactive(Agent):

    def __init__(self,
                 _id: int,
                 _original_id: str,
                 _env_knl: EnvironmentKnowledge,
                 _connection: Connection,
                 _agts_addrs: list,
                 interaction: bool = True):

        """
        Args:
            _id (int):
            _original_id (str):
            _env_knl (EnvironmentKnowledge):
            _connection (Connection):
            _agts_addrs (list):
            interaction (bool):
        """
        Agent.__init__(self, _id, _original_id, _env_knl, _connection,
                       _agts_addrs, interaction=interaction)

    def strategy_decide(self):
        super().strategy_decide()

        i = np.random.randint(len(self.env_knl.ntw.neighbours(
            self.pos)))
        self.PLAN.append(GoingToAction(self.env_knl.ntw.neighbours(
            self.pos)[i]))

        # i = np.random.randint(len(self.env_knl.ntw.graph[
        # self.pos[0]]))
        # self.PLAN.append(GoingToAction((i, -1, 0)))

    def process_message(self, m):
        """
        Args:
            m:
        """
        pass


