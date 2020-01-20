# -*- coding: utf-8 -*-

import sys

import numpy as np
import torch
from abc import abstractmethod, ABC

from pytrol import Paths
import pytrol.util.argsparser as parser
from pytrol.control.agent.Agent import Agent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection



import maptrainer.pathformatter as pf
import maptrainer.model as model_pckg


# Statistical model agent
class StatModelAgent(Agent, ABC):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 model_type: str,
                 model_variant: str,
                 datasrc: str,
                 variant: str = '',
                 depth: float = 3.0,
                 interaction: bool = True):
        r"""
        Args:
            id_ (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            model_type (str): The type of the model used to make predictions
            model_variant (str): The variant of the model used to make
                predictions
            datasrc (str):
            variant (str): strategy variant
            depth (float):
            interaction (bool):
        """

        super().__init__(id_=id_, original_id=original_id, env_knl=env_knl,
                         connection=connection, agts_addrs=agts_addrs,
                         variant=variant, depth=depth, interaction=interaction)

        if datasrc is None:
            datasrc = "hcc_0.2"

        self.datasrc = datasrc

        self.model_type = model_type

        self.model_variant = model_variant

        self.model = self.load_model()

        # Model's idlenesses' estimate: idlenesses' estimate output by
        # the model
        self.model_estm_idls = np.zeros(len(self.env_knl.idls), dtype=np.int16)

    @abstractmethod
    def load_model(self):
        r"""
        Returns:
            the model as a callable
        """
        pass

    def prepare_input(self, input_):
        r"""Prepares input data if needed. By default, identity function.

        Args:
            input_:
        """

        return torch.from_numpy(input_) if type(input_) is np.ndarray \
            else torch.Tensor(input_)

    def run_model(self, input_) -> torch.Tensor:
        """
        Args:
            input_:
        """

        input_ = self.prepare_input(input_)

        return self.model(input_)

