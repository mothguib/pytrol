# -*- coding: utf-8 -*-

import numpy as np

from pytrol.control.agent.HPMEstimator import HPMEstimator
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.util.randidlenest import draw_rand_idls


# Random Heuristic Pathfinder Mean Predictor
class RHPMEstimator(HPMEstimator):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
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
            variant (str):
            depth (float):
            interaction (bool):
        """

        HPMEstimator.__init__(self, id_=id_, original_id=original_id,
                              env_knl=env_knl, connection=connection,
                              agts_addrs=agts_addrs, variant=variant,
                              depth=depth, interaction=interaction)

    def estimate_idls(self) -> np.ndarray:
        r"""Predictor function: return the model's estimation of idlenesses"""

        # For each node the best idleness between the estimated,
        # the individual and the previous estimated incremented of 1 is
        # selected
        best_iidl_estm = super().estimate_idls()

        eidls = draw_rand_idls(best_iidl_estm, self.env_knl.shared_idls)

        return eidls
