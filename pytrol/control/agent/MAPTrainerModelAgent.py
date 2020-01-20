# -*- coding: utf-8 -*-

import sys
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


import pytrol.util.argsparser as parser
from pytrol.control.agent.StatModelAgent import StatModelAgent
from pytrol.model import Paths
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util import misc
from pytrol.util.net.Connection import Connection

# MAPTrainer project


import maptrainer.pathformatter as pf
import maptrainer.model as model_pckg


# Pytorch's model agent
class MAPTrainerModelAgent(StatModelAgent, ABC):

    def __init__(self,
                 id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 model_type: str,
                 model_variant: str,
                 datasrc: str,
                 interaction: bool = False,
                 variant: str = '',
                 gpu: bool = False,
                 depth: float = 3.0):
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
            interaction (bool):
            variant (str):
            gpu (bool):
            depth (float):
        """

        self.gpu = gpu

        # self.model_variant_2 = "Adagrad-pre"
        self.model_variant_2 = ''  # TODO: passing this attribute as argument
        # of the constructor

        self.variant_2 = "1-50--1"  # TODO: passing this attribute as argument
        # of the constructor

        self.nlayers = self.load_nlayers(self.variant_2)

        self.nhid = self.load_nhid(self.variant_2)

        self.bptt = self.load_bptt(self.variant_2)

        super().__init__(id_=id_, original_id=original_id,
                         env_knl=env_knl, connection=connection,
                         agts_addrs=agts_addrs, model_type=model_type,
                         model_variant=model_variant, variant=variant,
                         depth=depth, interaction=interaction,
                         datasrc=datasrc)

        self.model = self.cuda(self.model, self.gpu)

    def load_nlayers(self, variant: str):
        r"""
        Args:
            variant (str): `variant` is in the form of
                `<nlayers>-<nhid>-<bptt>`
        """

        return int(variant.split('-')[0])

    def load_nhid(self, variant: str):
        r"""
        Args:
            variant (str): `variant` is in the form of
                `<nlayers>-<nhid>-<bptt>`
        """

        return int(variant.split('-')[1])

    def load_bptt(self, variant: str):
        r"""
        Args:
            variant (str): `variant` is in the form of
                `rlpm-<nlayers>-<nhid>-<bptt>`
        """

        splt_variant = variant.split('-')

        return int(splt_variant[2]) if len(splt_variant) == 3 \
            else -int(splt_variant[3])

    def load_model(self, **kwargs) -> nn.Module:
        args = parser.parse_args()

        print(self.model_variant_2)
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

        print("Path of the loaded model: {}".format(model_path), '\n')

        return model

    def prepare_input(self, input_):
        r"""
        Args:
            input_:
        """

        # Use of the LSTM network
        input_ = torch.Tensor(input_)

        return self.cuda(input_, self.gpu)

    @staticmethod
    def cuda(o, gpu):
        r"""
        Args:
            o:
            gpu:
        """
        return o.cuda() if gpu else o
