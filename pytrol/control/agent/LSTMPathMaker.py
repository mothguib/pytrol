# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn

from pytrol import Paths
import pytrol.util.misc as misc
import pytrol.util.argsparser as parser
import pytrol.util.graphprocessor as gp
from pytrol.control.agent.MAPTrainerModelAgent import MAPTrainerModelAgent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection
from pytrol.model.action.GoingToAction import GoingToAction

# The MAPTrainer project


import maptrainer.pathformatter as pf
import maptrainer.model as model_pckg


# LSTM Path Maker
class LSTMPathMaker(MAPTrainerModelAgent):
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
                 datasrc: str = None,
                 **kwargs):

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
            **kwargs:
        """
        self.nlayers = self.load_nlayers(variant)

        self.nhid = self.load_nhid(variant)

        self.bptt = self.load_bptt(variant)

        # self.model_variant_2 = "SGD-pre"
        # self.model_variant_2 = "Adagrad-pre"
        self.model_variant_2 = ''  # TODO: passing this attribute as
        # argument of the constructor

        gpu = True  # TODO

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
                         model_type="RNN",
                         model_variant="LSTM",
                         datasrc=datasrc,
                         interaction=interaction)

        self.hidden = self.model.init_hidden(1)

    def load_nlayers(self, variant: str):
        r"""
        `rlpm-<nlayers>-<nhid>-<bptt>` :type variant: str :return:

        Args:
            variant (str): `variant` is in the form of
        """

        return int(variant.split('-')[0])

    def load_nhid(self, variant: str):
        r"""
        Args:
            variant (str): `variant` is in the form of
                `rlpm-<nlayers>-<nhid>-<bptt>`
        """

        return int(variant.split('-')[1])

    def load_bptt(self, variant: str):
        r"""
        Args:
            variant (str): `variant` is in the form of
                `rlpm-<nlayers>-<nhid>-<bptt>`
        """

        parts = variant.split('-')

        return - int(parts[-1]) if parts[-2] == '' \
            else - int(parts[-1])

    def load_model(self, **kwargs) -> nn.Module:
        args = parser.parse_args()

        # Directory path of the model
        dirpathmodel = pf. \
            generate_savefile_dirpath(type_="model",
                                      graph=self.env_knl.ntw_name,
                                      nlayers=self.nlayers,
                                      nhid=self.nhid,
                                      bptt=self.bptt,
                                      nagts=self.env_knl.nagts,
                                      model_type=self.model_type,
                                      model_variant=self.model_variant,
                                      suffix=self.model_variant_2,
                                      datasrc=self.datasrc,
                                      log_rep_dir=args.dirpath_models)

        # Path of the model's latest version
        # model_path = misc.get_latest_pytorch_model(dirpathmodel)
        model_path = pf. \
            entire_model_path(misc.get_latest_pytorch_model(dirpathmodel))

        with open(model_path, "rb") as s:
            if self.gpu:
                # Loading on GPU if trained with CUDA
                model = torch.load(s)
            else:
                # Loading on CPU if trained with CUDA
                model = torch.load(s,
                                   map_location=lambda storage, loc: storage)

        # TDP
        print("Path of the loaded model: {}".format(model_path))

        return model

    def strategy_decide(self):
        super().strategy_decide()

        self.PLAN.append(GoingToAction(self.select_goal_pos()))

    def select_goal_pos(self) -> tuple:
        output, self.hidden = self.run_model(self.pos[0])
        next_vtx = self.best_nghb_vtx(output[0][0])

        return next_vtx, -1, 0

    def prepare_input(self, input_):
        r"""Creates the one-hot vector corresponding to `input_` the vertex
        passed as argument.

        Args:
            input_:
        """

        vtx = input_
        nb_vts = len(self.env_knl.ntw.graph)

        # Use of the LSTM network
        input_ = torch.zeros(1, 1, nb_vts)
        input_ = input_.cuda() if self.gpu else input_
        input_[0][0][vtx] = 1
        input_ = input_

        return input_

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

        # Neighbours' probabilities output by the model
        ngbs_log_probs = \
            torch.take(log_probs,
                       torch.LongTensor(scalar_ngbs).cuda() if self.gpu
                       else torch.LongTensor(scalar_ngbs))

        max_log_probs, indices = torch.max(ngbs_log_probs, 0)  # plural for the
        # case where there are maximum probabilities equal
        i = indices.item()

        return scalar_ngbs[i]

    def stop(self):
        del self.model, self.hidden

    def run_model(self, input_) -> torch.Tensor:
        r"""
        Args:
            input_:
        """

        input_ = self.prepare_input(input_)

        return self.model(input_, self.hidden)
