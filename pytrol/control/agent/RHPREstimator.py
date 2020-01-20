# -*- coding: utf-8 -*-

import numpy as np

from pytrol.control.agent.HPREstimator import HPREstimator
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.util.randidlenest import draw_rand_idls


# Random Heuristic Pathfinder ReLU Predictor
class RHPREstimator(HPREstimator):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 gpu: bool = False,
                 depth: float = 3.0,
                 interaction: bool = True,
                 model_type: str = "MLP",
                 model_variant: str = "ReLU"):
        r"""
        Args:
            id_ (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            variant (str):
            gpu (bool):
            depth (float):
            interaction (bool):
            model_type (str): The type of the model used to make predictions
            model_variant (str): The variant of the model used to make
                predictions
        """

        HPREstimator.__init__(self, id_=id_, original_id=original_id,
                              env_knl=env_knl, gpu=gpu, model_type=model_type,
                              connection=connection, agts_addrs=agts_addrs,
                              variant=variant, depth=depth,
                              model_variant=model_variant,
                              interaction=interaction)

    def estimate_idls(self) -> np.ndarray:
        r"""Predictor function: return the model's estimation of idlenesses"""

        # For each node the best idleness between the estimated,
        # the individual and the previous estimated incremented of 1 is
        # selected
        best_iidl_estm = super().estimate_idls()

        eidls = draw_rand_idls(best_iidl_estm, self.env_knl.shared_idls)

        return eidls
