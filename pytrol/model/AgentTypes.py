# -*- coding: utf-8 -*-

from pytrol.control.agent.MAPLSTMPathMaker import MAPLSTMPathMaker
from pytrol.control.agent.RNNMAPLSTMPathMaker import RNNMAPLSTMPathMaker
from pytrol.control.agent.Coordinated import Coordinated
from pytrol.control.agent.HCCoordinator import HCCoordinator
from pytrol.control.agent.HPCCoordinator import HPCCoordinator
from pytrol.control.agent.ConscientiousCoordinator import \
    ConscientiousCoordinator
from pytrol.control.agent.ConscientiousReactive import ConscientiousReactive
from pytrol.control.agent.HPLEstimator import HPLEstimator
from pytrol.control.agent.HPREstimator import HPREstimator
from pytrol.control.agent.LSTMPathMaker import LSTMPathMaker
from pytrol.control.agent.HPMEstimator import HPMEstimator
from pytrol.control.agent.RHPLEstimator import RHPLEstimator
from pytrol.control.agent.RHPMEstimator import RHPMEstimator
from pytrol.control.agent.RHPREstimator import RHPREstimator
from pytrol.control.agent.RNNLSTMPathMaker import RNNLSTMPathMaker
from pytrol.control.agent.RandomCoordinator import RandomCoordinator
from pytrol.control.agent.RandomReactive import RandomReactive


class AgentTypes:
    NoType = -1

    # Random Reactive
    RR = 0

    # Conscientious Reactive
    CR = 1

    # Coordinated
    Cd = 2

    # Random Coordinator
    RCr = 3

    # Conscientious Coordinator
    CRr = 4

    # Heuristic Cognitive Coordinator
    HCCr = 5

    # Heuristic Pathfinder Cognitive Coordinator
    HPCCr = 6

    # LSTM Path Maker
    LPM = 7

    # Directed Random LSTM Path Maker
    RLPM = 8

    # Heuristic Pathfinder Linear Predictor
    HPLE = 9

    # Heuristic Pathfinder Mean Predictor
    HPME = 10

    # Heuristic Pathfinder ReLU Predictor
    HPRE = 11

    # Random Heuristic Pathfinder Mean Predictor
    RHPME = 12

    # Random Heuristic Pathfinder Linear Predictor
    RHPLE = 13

    # Random Heuristic Pathfinder ReLU Predictor
    RHPRE = 14

    # MAP LSTM-Path-Maker
    MAPLPM = 15

    # Random MAP LSTM-Path-Maker
    RMAPLPM = 16

    str_to_id = {"rr": RR,
                 "cr": CR,
                 "cd": Cd,
                 "rcr": RCr,
                 "crr": CRr,
                 "hccr": HCCr,
                 "hcc": Cd,  # Heuristic Cognitive CoordinatED
                 "hpccr": HPCCr,
                 "hpcc": Cd, # Heuristic Pathfinder Cognitive CoordinatED
                 "lpm": LPM,
                 "rlpm": RLPM,
                 "hple": HPLE,
                 "hpme": HPME,
                 "hpre": HPRE,
                 "rhpme": RHPME,
                 "rhple": RHPLE,
                 "rhpre": RHPRE,
                 "hplp": HPLE,
                 "hpmp": HPME,
                 "hprp": HPRE,
                 "rhpmp": RHPME,
                 "rhplp": RHPLE,
                 "rhprp": RHPRE,
                 "maplpm": MAPLPM,
                 "rmaplpm": RMAPLPM}

    id_to_class_name = [RandomReactive, ConscientiousReactive, Coordinated,
                        RandomCoordinator, ConscientiousCoordinator,
                        HCCoordinator, HPCCoordinator, LSTMPathMaker,
                        RNNLSTMPathMaker, HPLEstimator,
                        HPMEstimator, HPREstimator, RHPMEstimator,
                        RHPLEstimator, RHPREstimator, MAPLSTMPathMaker,
                        RNNMAPLSTMPathMaker]
