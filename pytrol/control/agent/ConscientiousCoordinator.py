# -*- coding: utf-8 -*-

import numpy as np
from pytrol.control.agent.Coordinator import Coordinator
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge

from pytrol.util.net.Connection import Connection


class ConscientiousCoordinator(Coordinator):
    r"""
    Args:
        id (int):
        original_id (str):
        env_knl (EnvironmentKnowledge):
        connection (Connection):
        agts_addrs (list):
        variant (str):
        depth (float):
        interaction (bool):
    """
    def __init__(self,
                 id: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 interaction: bool = True):

        Coordinator.__init__(self, id, original_id, env_knl, connection,
                             agts_addrs, variant, depth,
                             interaction=interaction)

        self.d = False

    def strategy_decide(self):
        super().strategy_decide()

        print("--###--")
        while len(self.to_send) > 0:
            # For each entry e in the list of dictionaries to_send
            e = self.to_send.popleft()

            # Updating of the requester agent
            print("pos of e -> ", e["pos"], " with the previous idleness "
                                            "of", e["a_id"], "-> ",
                  self.env_knl.idls[e["pos"][0]])
            self.env_knl.idls[e["pos"][0]] = 0
            print("Applied update -> idls[", e["pos"][0], "]",
                  self.env_knl.idls[e["pos"][0]])

            neighbours = self.env_knl.ntw.neighbours(e["pos"])
            sorted_neighbours = np.array(sorted(neighbours,
                                                key=lambda x:
                                                self.env_knl.idls[x[0]],
                                                reverse=True))
            worst_neighbours = [x for x in sorted_neighbours if
                                self.env_knl.idls[x[0]] ==
                                self.env_knl.idls[sorted_neighbours[0][0]]]
            i = np.random.randint(len(worst_neighbours))

            print("New pos is sent to e(", e, ")-> ", worst_neighbours[i],
                  " whose idleness is -> ", self.env_knl.idls[
                      worst_neighbours[i][0]])
            self.send("goal_position:" + str((worst_neighbours[i][0], -1,
                                              0)), e["a_id"])

        print("Current coordinator's idlenesses: ",
              self.env_knl.idls)
        print("--###--")
