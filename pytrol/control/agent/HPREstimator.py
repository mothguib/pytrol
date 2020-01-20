# -*- coding: utf-8 -*-

import numpy as np
import torch

import pytrol.util.argsparser as parser
from pytrol.control.agent.HPAgent import HPAgent
from pytrol.control.agent.MAPTrainerModelAgent import MAPTrainerModelAgent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


# Heuristic Pathfinder ReLU Predictor
class HPREstimator(HPAgent, MAPTrainerModelAgent):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 datasrc: str = None,
                 variant: str = '',
                 gpu: bool = False,
                 depth: float = 3.0,
                 model_type: str = "MLP",
                 model_variant: str = "ReLU",
                 interaction: bool = True):
        r"""
        Args:
            id_ (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            datasrc (str):
            variant (str):
            gpu (bool):
            depth (float):
            model_type (str): The type of the model used to make predictions
            model_variant (str): The variant of the model used to make
                predictions
            interaction (bool):
        """

        HPAgent.__init__(self, id_=id_, original_id=original_id,
                         env_knl=env_knl,
                         connection=connection, agts_addrs=agts_addrs,
                         variant=variant, depth=depth, interaction=interaction)

        if datasrc is None:
            args = parser.parse_args()
            datasrc = args.datasrc

        MAPTrainerModelAgent.__init__(self, id_=id_, original_id=original_id,
                                      env_knl=env_knl, connection=connection,
                                      agts_addrs=agts_addrs, variant=variant,
                                      depth=depth, gpu=gpu,
                                      model_type=model_type,
                                      model_variant=model_variant,
                                      interaction=interaction, datasrc=datasrc)

    def run_model(self, input_) -> torch.Tensor:
        r"""
        Args:
            input_:
        """

        input_ = self.prepare_input(input_)

        output = self.model(input_)

        self.model_estm_idls = output

        return self.model_estm_idls

    def estimate_idls(self) -> np.ndarray:
        r"""Predictor function: returns the model's estimation of idlenesses"""

        estimated_idls = self.run_model(
            self.env_knl.idls).detach().cpu().numpy()

        # TODO: changing the model, meanwhile any negative idleness is
        #  frozen (set to 0)
        # Positive estimated idlenesses
        positive_estm_idls = np.maximum(estimated_idls,
                                        np.zeros(
                                            np.array(estimated_idls).shape)
                                        )

        # For each node the best idleness between the estimated,
        # the individual and the previous estimated incremented of 1 is
        # selected
        best_iidl_estm = \
            np.minimum(np.minimum(positive_estm_idls,
                                  self.env_knl.shared_idls),
                       np.array(self.prev_estimated_idls, dtype=np.int16) + 1)

        return best_iidl_estm
