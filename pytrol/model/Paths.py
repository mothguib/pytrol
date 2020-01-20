# -*- coding: utf-8 -*-

import os

PRJCTROOT = os.path.abspath(os.path.dirname(__file__)) + "/../../"

PCKGROOT = os.path.abspath(os.path.dirname(__file__)) + "/../"

DATAROOT = os.environ['MAPDATA']

DATA = DATAROOT + "/Pytrol-Resources/"

MATERIALS = PCKGROOT + "materials/"

SIM_MATERIALS = MATERIALS + "sim_materials/"

MAPS = SIM_MATERIALS + "maps/"

JSON_MAPS = MAPS + "json/"

LOCALEXECS = SIM_MATERIALS + "execs/"

EXECS = DATA + "/execs"

MEANS = DATA + "/means/"

LOCALMEANS = PRJCTROOT + "/means/"

# TODO: Restore the paths built from current working directory
LOCALLOGS = PRJCTROOT + "/logs/logs/"

LOGS = DATA + "/logs/"

# PytrolSimEditor
PSE = PRJCTROOT + "/../" + "/PytrolSimEditor"

# The Pytorch Trainer Resources
MTRNR = DATAROOT + "/MAPTrainer-Resources/"

DIRPATHMODELS = MTRNR + "/models/"

MTRN_PRJCT = PRJCTROOT + "/../" + "/MAPTrainer/"
