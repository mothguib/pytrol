# -*- coding: utf-8 -*-

import ast
from abc import ABC
from collections import deque

import numpy as np

from pytrol.control.Communicating import Communicating
from pytrol.model.action.Action import Action
from pytrol.model.action.Actions import Actions
from pytrol.model.action.MovingToAction import MovingToAction
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection

from pytrol.model.action.GoingToAction import GoingToAction


class Agent(ABC, Communicating):
    r"""
    `Agent` is an abstract class defining a template for any agent strategy.
    This template defines, in fact, the basic procedure that any agent must
    follow. This basic procedure, qualified as *main procedure of agent*,
    represents the life cycle of agents and consists of:

    - `Agent.prepare`: any preprocessing, if necessary, the
          agent needs to carry out to prepare the impending main procedure,

    - `Agent.perceive`: the agent perceives the position of the other
          ones, if required by its strategy; in the current version of PyTrol
          only the position of the agent itself is perceived, although other
          types of perception are left to the discretion of the user,

    - `Agent.communicate`: the agent communicates with other
          ones, if required by its strategy;

    - `Agent.analyse`: the agent checks and processes messages he has
          received,

    - `Agent.decide`: the agent decides; this method makes up the core
          of the strategy, given that any strategy is a decision-making
          procedure in the context of MAP,

    - `Agent.act`: the agent acts according to the decision made in the
          previous method.

    Each agent, namely each object instantiating the `Agent` class, is a
    *communicating* and therefore a thread; concretely, the `Agent` class
    extends the `Communicating` class. Any new strategy to add in PyTrol shall
    be implemented from the above methods, then added to the
    `pytrol.control.agent` package, and finally referenced in
    `pytrol.model.AgentTypes`

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

        Communicating.__init__(self, connection)

        self.id = id_

        self.original_id = original_id

        self.agts_addrs = agts_addrs

        self.env_knl = env_knl

        self.agts_addrs = agts_addrs

        # Current position
        self.pos = tuple(env_knl.agts_pos[self.id])

        self.goal_pos = (-1, -1, -1)

        self.depth = depth

        # If an agent is not situated, then its position is not taken into
        # account while incrementing idleness
        self.situated = situated

        # If the agent is nos positioned, then it is not situated
        if self.pos == (-1, -1, -1):
            self.situated = False

        # Plan
        self.PLAN = deque() # TODO: Sending_message actions must be put on
        # the top of deque PLAN and move actions (type 0 et 1) on the bottom.

        # Decision is made or not
        self.d = True

        self.variant = self.get_strat_params(variant)

        self.interaction = interaction

        self.interacted = False

    def set_agts_addrs(self, agts_addrs):
        r"""
        Args:
            agts_addrs:
        """
        self.agts_addrs = agts_addrs

    @staticmethod
    def get_strat_params(variant: str):
        """Returns the parameters setting the strategy's variant. By default
        returns the variable `variant` passed as argument

        Args:
            variant (str):
        """

        return variant

    def send(self, message, agt_id: int):
        r"""
        Args:
            message:
            agt_id (int):
        """
        super(Agent, self).send(message, self.agts_addrs[agt_id])

    # 3. Preparation of the upcoming main procedure
    def prepare(self):
        self.ac = False
        self.interacted = False

    # 4. Main procedure of agent

    # 4.1 Communicating
    # First communications if needed
    def communicate(self):
        if self.interaction and not self.interacted:
            self.interact()
            self.interacted = True

    # 4.2 Perceiving
    def perceive(self, agts_pos: np.ndarray):
        # Perceives the other agents around it
        r"""
        Args:
            agts_pos (np.ndarray):
        """
        self.env_knl.agts_pos = agts_pos

    # 4.3 Analysing
    # Analyses perceptions and received message to know its new state
    def analyse(self):
        self.receive()
        if len(self.messages) > 0:
            while len(self.messages) > 0:
                m = self.messages.popleft()
                self.process_message(m)

    # 4.3.1 Message processing
    def process_message(self, m):
        r"""
        Args:
            m:
        """
        if self.interaction:
            if str(m).startswith("shared_idlenesses"):
                # Transmitted idleness

                # TDP
                # print("#", self.env_knl.t, self.id, ":")
                # print(self.env_knl.idls)
                # print(str(m).split(':')[1])
                # print(np.array(ast.literal_eval(str(m).split(':')[3])))

                self.env_knl.shared_idls = \
                    np.minimum(self.env_knl.shared_idls, ast.literal_eval(
                        str(m).split(':')[3]))

                # `ast.literal_eval(str(m).split(':')[2]))` yields the list
                # following the second `:` in the received message

                # TDP
                # print(self.env_knl.idls)
                # print("# --------------------------------")

    # 4.4 Deciding
    def decide(self):
        # If it is the time to decide:
        if self.d:
            # Applying politic defined by the strategy:
            self.strategy_decide()
        self.d = False

    def strategy_decide(self):
        pass

    # 4.5 Acting
    def act(self) -> Action:
        a = Action("none_action", -1)
        if len(self.PLAN) != 0:
            while not self.ac:
                a = self.PLAN.popleft()

                if a.type == Actions.Going_to:
                    self.act_gt(a)

                elif a.type == Actions.Moving_to:
                    self.act_mt(a)
                    self.ac = True

                elif a.type == Actions.Waiting:
                    self.act_w(a)
                    self.ac = True

                '''
                elif a.type == Actions.Stopping_move:
                    self.act_st_mv(a)
            '''
        return a

    # 4.5.1: Act going to
    def act_gt(self, a: GoingToAction):
        r"""
        Args:
            a (GoingToAction):
        """

        # TDP
        # print(str(self.id), "is in", self.pos,
        #      " and planned to go to ", a.goal_position)

        # Retrieval of the path from.pos to goal_vertex

        path = self.env_knl.ntw.path(self.pos,
                                     a.goal_position)

        # TDP
        # print(str(self.id), " takes the path: ", path)

        self.goal_pos = a.goal_position

        # Making up of the Moving_to actions' plan to go to goal_vertex
        for i in range(len(path) - 1):
            self.PLAN.append(MovingToAction(path[i], path[i + 1]))

    # 4.5.2: Act moving to
    def act_mt(self, a: MovingToAction):
        # TDP
        # print(str(self.id), "is in", self.pos,
        #      " and planned to move to ", a.to)

        """
        Args:
            a (MovingToAction):
        """
        self.pos = a.to

        if self.goal_pos == a.to:
            # misc.vertices_equals(self.goal_pos, a.to)
            self.d = True

    # 4.5.3: Act waiting
    def act_w(self, a: Action):
        r"""
        Args:
            a (Action):
        """
        pass

    # # 4.5.4: Act stopping move
    def act_st_mv(self, a: Action):
        r"""
        Args:
            a (Action):
        """
        for i in range(len(self.PLAN)):
            # Deletion of the actions being Going/Moving_to
            # actions
            if self.PLAN[i].type < 2:
                del self.PLAN[i]

    # 5 Knowledge Processing
    def update_knowledge(self):
        self.env_knl.tick()

        # If the agent is on a vertex, situated and not crossing an edge
        if self.situated:
            self.env_knl.reset_idl(self.pos)

    # 6 Post-processings
    def post_process(self):
        self.interacted = False

    def stop(self):
        pass

    def interact(self):
        for i, p in enumerate(self.env_knl.agts_pos):
            if p[0] != -1:
                m = "shared_idlenesses:{}:{}:{}". \
                      format(self.env_knl.t, self.id,
                             self.env_knl.shared_idls.tolist())
                # TDP
                # print(m, i)
                self.send(m, i)
