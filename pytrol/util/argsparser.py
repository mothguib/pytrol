# -*- coding: utf-8 -*-

import argparse

from pytrol import Paths


def parse_args():
    parser = argparse.ArgumentParser(description="Pytrol: Python simulator "
                                                 "for the MAP problem")

    parser.add_argument("--map", type=str, default="islands",
                        help="the topology of network")
    parser.add_argument("--nagts", type=int, default=10,
                        help="number of agents")
    parser.add_argument("--strategy", type=str, default="hpcc",
                        help="the strategy of agents")
    parser.add_argument("--society", type=str, default="",
                        help="the society name of agents")
    parser.add_argument("--variant", type=str, default='0.5',
                        help="the variant of the strategy (its parameters, "
                             "etc.")
    parser.add_argument("--duration", type=int, default=20,
                        help="number of periods of the execution")
    parser.add_argument("--depth", type=int, default=14,
                        help="depth of perceptions of agents")
    parser.add_argument("--interaction", action="store_true",
                        help="Activating interaction between agents")

    parser.add_argument("--trace-agents", action="store_true",
                        help="Tracing of agents' idlenesses at each time step")
    parser.add_argument("--trace-agents-estimates", action="store_true",
                        help="Tracing of agents' idlenesses' estimates at "
                             "each time step. Only for MAPTrainerModelAgent "
                             "agents")

    parser.add_argument("--execid", type=int, default=0,
                        help="Id of the current execution")
    parser.add_argument("--nexecs", type=int, default=2,
                        help="number of simulation executions")
    parser.add_argument("--inf-exec-id", type=int, default=0,
                        help="Infemum execution id")
    parser.add_argument("--sup-exec-id", type=int, default=29,
                        help="Supremum execution id")

    parser.add_argument("--ninputs", type=int, default=50,
                        help="number of inputs for ANNs")
    parser.add_argument("--noutputs", type=int, default=50,
                        help="number of outputs for ANNs")
    parser.add_argument("--nlayers", type=int, default=1,
                        help="number of layers for ANNs")
    parser.add_argument("--nhid", type=int, default=50,
                        help="number of units for each hidden layer for ANNs")
    parser.add_argument("--bptt", type=int, default=-1,
                        help="Back-propagation through time")
    parser.add_argument("--gpu", action="store_true",
                        help="Loading the model on the GPU")
    parser.add_argument("--datasrc", type=str, default="hpcc_0.5",
                        help="MAP data source, e.g. HPCC 0.5")
    parser.add_argument("--meanspath", type=str, default=Paths.MEANS,
                        help="Directory path where means are stored for the "
                             "HPME strategies")

    parser.add_argument("--dirpath-logs", type=str, default=Paths.LOCALLOGS,
                        help="path of the directory where saving logs of "
                             "executions")
    parser.add_argument("--dirpath-data", type=str, default=Paths.DATA,
                        help="directory path of Pytrol's data")
    parser.add_argument("--dirpath-execs", type=str, default=Paths.EXECS,
                        help="directory path of executions files")
    parser.add_argument("--dirpath-models", type=str,
                        default=Paths.DIRPATHMODELS,
                        help="directory path of machine learning models")

    args = parser.parse_args()

    return args
