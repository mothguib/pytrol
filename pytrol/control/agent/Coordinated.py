# -*- coding: utf-8 -*-

from ast import literal_eval as make_tuple

from pytrol.control.agent.Agent import Agent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.model.action.GoingToAction import GoingToAction


class Coordinated(Agent):
    r"""
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

    def __init__(self, id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 interaction: bool = False):

        Agent.__init__(self, id_=id_, original_id=original_id,
                       env_knl=env_knl, connection=connection,
                       agts_addrs=agts_addrs, variant=variant,
                       depth=depth, interaction=interaction)

    def process_message(self, m):
        r"""
        Args:
            m:
        """
        super().process_message(m)

        if str(m).startswith("next_position"):
            # Split string
            ss = str(m).split(':')
            self.PLAN.append(GoingToAction(make_tuple(ss[1])))

            """
            # pour l'envoi de plusieurs plans
            for i in range(1, len(ss)):
                self.PLAN.append(GoingToAction(make_tuple(ss[i])))
            """

            return

    def strategy_decide(self):
        super().strategy_decide()

        self.send("goal_request:" + str(self.id) + ":" + str(
                  self.pos), 0)
