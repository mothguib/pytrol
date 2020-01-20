# -*- coding: utf-8 -*-

from ast import literal_eval as make_tuple
from collections import deque

import numpy as np

from pytrol.control.agent.Agent import Agent
from pytrol.model.action.Action import Action
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


class Coordinator(Agent):
    r"""
    Args:
        id_ (int):
        original_id (str):
        env_knl (EnvironmentKnowledge):
        connection (Connection):
        agts_addrs (list):
        variant (str):
        depth (float):
        situated (bool):
        interaction (bool):
    """
    def __init__(self, id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 situated: bool = True,
                 interaction: bool = False):

        Agent.__init__(self, id_, original_id, env_knl, connection,
                       agts_addrs, variant, depth, situated,
                       interaction=interaction)

        # Messages to send to the agents to provide them their next goal
        # position
        self.to_send = deque()

    def process_message(self, m):
        r"""
        Args:
            m:
        """
        super().process_message(m)

        if str(m).startswith("goal_request"):
            # Split string
            ss = str(m).split(':')

            # Current agent id
            agt_id = int(ss[1])
            # Current agent position
            agt_pos = make_tuple(ss[2])

            # Addition of the current requesting agent to the group of agents
            # to which to send the next position
            self.to_send.append({"a_id": agt_id, "pos": make_tuple(ss[2])})

            # Updating of agent position
            self.env_knl.agts_pos[agt_id] = np.asarray(agt_pos, dtype=np.int16)

            # Updating of individual idleness
            self.env_knl.reset_idl(agt_pos)

            self.d = True

    def act(self) -> Action:
        # In the current version, the coordinator agent is a special type of
        # agent. Its "action completed" boolean variable `ac` is always `true`
        # because it executes no actions but sending and receiving messages.
        # Therefore, it do not call the method `act` of its parent.

        self.coordinator_act()
        self.ac = True

        a = Action("none_action", -1)

        return a

    def coordinator_act(self):
        pass
