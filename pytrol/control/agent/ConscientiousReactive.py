# -*- coding: utf-8 -*-

from pytrol.control.agent.Agent import Agent
from pytrol.model.action.GoingToAction import GoingToAction
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


class ConscientiousReactive(Agent):
    r"""Class Conscientious Reactive (CR)
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

        """
        Agent.__init__(self, id_, original_id, env_knl, connection,
                       agts_addrs, variant, depth, interaction=interaction)

    def strategy_decide(self):
        super().strategy_decide()

        my_neighbours = self.env_knl.ntw.neighbours(self.pos)
        # print("neighbrous = ", my_neighbours)

        # for n in my_neighbours:
        #    print(n," ",self.env_knl.idls[n[0]])

        best_neighbour = max(my_neighbours,
                             key=lambda x: self.env_knl.idls[x[0]])

        # print("best_neighbours = ", best_neighbour)

        self.PLAN.append(GoingToAction(best_neighbour))

    def process_message(self, m):
        """
        Args:
            m:
        """
        pass
