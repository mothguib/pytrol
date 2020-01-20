# -*- coding: utf-8 -*-

import numpy as np

from pytrol.control.agent.Coordinator import Coordinator
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


class RandomCoordinator(Coordinator):

    def __init__(self,
                 id: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 interaction: bool = True):

        """
        Args:
            id (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            interaction (bool):
        """
        Coordinator.__init__(self, id, original_id, env_knl,
                             connection, agts_addrs, interaction=interaction)

        self.d = False

    def strategy_decide(self):
        super().strategy_decide()

        # For each entry e in the deque of dictionaries to_send
        for e in self.to_send:
            i = np.random.randint(len(self.env_knl.ntw.neighbours(
                e["pos"])))
            e["next_pos"] = self.env_knl.ntw.neighbours(e["pos"])[i]

    def coordinator_act(self):
        print("Waking up")
        while len(self.to_send) > 0:
            # For each entry e in the list of dictionaries to_send
            e = self.to_send.popleft()
            self.send("next_position:" + str(e["next_pos"]), e["a_id"])