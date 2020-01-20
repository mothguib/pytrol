# -*- coding: utf-8 -*-

import json
import torch

import numpy as np

import pytrol.util.argsparser as parser
from pytrol.control.agent.HPAgent import HPAgent
from pytrol.control.agent.StatModelAgent import StatModelAgent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.util import pathformatter as pf


# Heuristic Pathfinder Mean Predictor
class HPMEstimator(HPAgent, StatModelAgent):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 datasrc: str = None,
                 model_type: str = "Mean",
                 model_variant: str = "Standard",
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
            datasrc (str):
            model_type (str): The type of the model used to make predictions
            model_variant (str): The variant of the model used to make
                predictions
            variant (str):
            depth (float):
            interaction (bool):
        """

        HPAgent.__init__(self, id_=id_, original_id=original_id,
                         env_knl=env_knl,
                         connection=connection, agts_addrs=agts_addrs,
                         variant=variant, depth=depth, interaction=interaction)

        if datasrc is None:
            args = parser.parse_args()
            datasrc = args.datasrc

        StatModelAgent.__init__(self, id_=id_, original_id=original_id,
                                env_knl=env_knl, connection=connection,
                                agts_addrs=agts_addrs, model_type=model_type,
                                model_variant=model_variant, variant=variant,
                                depth=depth, interaction=interaction,
                                datasrc=datasrc)

        # Nodes' mean idlenesses:
        self.idls_means = self.load_idlenesses_means()

    def model(self, _input):
        r"""
        Args:
            _input:
        """
        return torch.Tensor(self.idls_means)

    def load_model(self):
        return self.model

    def estimate_idls(self) -> np.ndarray:
        r"""Predictor function: returns the model's estimation of
        idlenesses."""

        return np.minimum(self.run_model(self.env_knl.idls).numpy(),
                          self.env_knl.shared_idls)

    def load_idlenesses_means(self) -> list:
        args = parser.parse_args()

        # Current configuration's means' file path
        config_means_fp = pf.means_path(datasrc=self.datasrc,
                                        g=self.env_knl.ntw_name,
                                        n=self.env_knl.nagts,
                                        mean_dirpath=args.meanspath)

        with open(config_means_fp) as s:
            return json.load(s)

    def run_model(self, input_) -> torch.Tensor:
        r"""
        Args:
            input_:
        """

        input_ = self.prepare_input(input_)

        output = self.model(input_)

        self.model_estm_idls = output

        return self.model_estm_idls
