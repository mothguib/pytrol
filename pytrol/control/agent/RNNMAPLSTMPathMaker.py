# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch

from pytrol import Paths
import pytrol.util.argsparser as parser
import pytrol.util.graphprocessor as gp
from pytrol.control.agent.MAPLSTMPathMaker import MAPLSTMPathMaker
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection




# Random Next Neighbour LSTMPathMaker
class RNNMAPLSTMPathMaker(MAPLSTMPathMaker):
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
                 datasrc: str = None):

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
            datasrc (str):
        """
        if datasrc is None:
            args = parser.parse_args()
            datasrc = args.datasrc

        super().__init__(id_=id_,
                         original_id=original_id,
                         env_knl=env_knl,
                         connection=connection,
                         agts_addrs=agts_addrs,
                         variant=variant,
                         depth=depth,
                         gpu=gpu,
                         interaction=interaction,
                         datasrc=datasrc)

    def best_nghb_vtx(self, log_probs: torch.Tensor) -> int:
        r"""
        Args:
            log_probs (torch.Tensor): vector of vertices' probabilities.

        Returns:
            the vertex id with the best probability
        """

        vtx = self.pos[0]

        # Neighbours' ids of the current vertex `vtx`
        scalar_ngbs = gp.build_neighbours(self.env_knl.ntw.graph)[vtx]

        # Tensor containing `scalar_ngbs`
        scalar_ngbs_t = self.cuda(torch.LongTensor(scalar_ngbs), self.gpu)

        # Neighbours' probabilities output by the model
        ngbs_probs = torch.exp(torch.take(log_probs, scalar_ngbs_t))
        ngbs_probs = ngbs_probs / torch.sum(ngbs_probs)

        next_vtx = np.random.choice(scalar_ngbs, p=ngbs_probs.detach().cpu(). \
                                    numpy())

        del scalar_ngbs_t, ngbs_probs

        return next_vtx
