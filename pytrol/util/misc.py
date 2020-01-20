# -*- coding: utf-8 -*-

import os
import time
import datetime

from pytrol import Paths
from pytrol.model.Maps import Maps
from pytrol.util.graphprocessor import *
from pytrol.util import pathformatter as pf


def _min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
    Args:
        a:
        b:
    """
    c = a <= b
    return a * c + (1 - c) * b


def indent(width: int):
    r"""
    Args:
        width:
    """
    r = ""
    for i in range(width):
        r += " "
    return r


def name_config_file(m: int, nbs_agts: list, sats: list = None, msn: str = '',
                     msnd: str = '', pfx: str = '', sfx: str = '',
                     odp: str = '') -> str:
    r"""
    Args:
        m:
        nbs_agts:
        sats:
        msn: meta-society name
        msnd: name of the meta-society directory in which the
        pfx:
        sfx: suffix
        odp: output root directory path of the configuration/execution
    """

    if pfx == '':
        # If sats is not empty
        if len(sats) > 0:
            # If the agents in the first society share the same strategy
            if len(sats[0]) == 1:
                pfx = sats[0][0]
            # Else if this is a coordinated strategy where except the
            # coordinator, all others agents share the same strategy
            else:
                c_str = True
                for i in range(2, len(sats[0])):
                    if sats[0][i] != sats[0][i - 1]:
                        c_str = False
                        break
                if c_str:
                    pfx = sats[0][1]
                    # TODO: Handle the case where there are others societies

    # Name of the meta-society directory. This is the acronym of the
    # second strategy (enabling to take into account he
    # coordinator-based strategies) in the first society of the
    # meta-society
    msn_id = sats[0][1] if len(sats[0]) > 1 else sats[0][0]

    if msnd == '':
        msnd = msn_id

    # If msn is non-defined, it is assigned a default meta-society name
    # from the pattern metasoc/soc_<metasoc id>_<nb of agts #1>_..._<nb of
    #                                                                  agts #n>
    if msn == '':
        msn = name_metasociety(nbs_agts, msn_id)
    if odp == '':
        odp = Paths.LOCALEXECS + '/'

    odp += '/' + Maps.id_to_name[m] + "/" + \
           msnd + "/" + \
           str(sum(nbs_agts)) + "/"

    return odp + \
           pfx + \
           ('-' if pfx != '' else '') + Maps.id_to_name[m] + \
           '-' + msn + \
           ('-' if len(nbs_agts) > 0 else '') + \
           '-'.join(str(n) for n in nbs_agts) + \
           '-' + sfx + \
           ".json"


def name_metasociety(nbs_agts: list, id: str = '', sfx: str = ''):
    r"""
    Args:
        nbs_agts: 1D list of numbers of agents
        id:
        sfx: suffix of the society name
    """
    id += '_'

    if sfx == '':
        sfx = '_'.join(str(n) for n in nbs_agts)
    return ("metasoc_" if len(nbs_agts) > 1 else "" if len(nbs_agts) == 0
    else "soc_") \
           + id \
           + sfx


# Retrieve the identical values in an iterable
def identical_values(itr):

    # The output
    r"""
    Args:
        itr:
    """
    o = []

    # Already added ids
    aai = []

    for i in range(len(itr)):
        if i not in aai:
            aai += [i]
            # Temporary output: the iterable's ids having the same current
            # value
            to = []
            for j in range(len(itr)):
                if itr[i] == itr[j]:
                    to += [j]
                    aai += [j]

            o += [to]

    return o


def get_memusage():
    import psutil

    # Memory used by the current process
    usedmem = psutil.Process(os.getpid()).memory_info().rss
    # Total physic memory available
    totalmem = psutil.virtual_memory().total

    return usedmem / totalmem * 10


def timestamp() -> str:
    r"""Returns the current time."""

    return '[' + datetime.datetime.fromtimestamp(time.time()) \
        .strftime('%Y-%m-%d %H:%M:%S') + ']'


def log(s: str):
    # Formatted string
    r"""
    Args:
        s (str):
    """
    f = "{}: {}".format(timestamp(), s)
    print(f)


def ends_with_pt(f: str):
    r"""Test whether a file path ends with ".pt".

    Args:
        f (str): the file path
    """
    return f.endswith(".pt")


def get_latest_pytorch_model(dirpath):
    r"""
    Args:
        dirpath: Path of directory where find the latest file
    """

    return \
        sorted(
            list(
                filter(os.path.isfile and ends_with_pt,
                       [dirpath + f for f in os.listdir(dirpath)])
            ),
            key=pf.timestamp_from_filename
        )[-1]


def lv_model_path(model_name: str, model_dirpath: str) -> str:
    r"""Getting the file path of model `model_name` 's latest version.

    Args:
        model_name:
        model_dirpath: directory path of the save files for the current
            model type and variant, default pattern is
            `models/<type>/<variant>/<map>` for basic models
            `models/<type>/<variant>/<pre>/<map>` for premodels

    Returns:
        The path of the last version of `model_name`.
    """

    # Path of the required model's directory whose the name is `model_name`
    model_dirpath = model_dirpath + '/' + model_name + '/'

    # Path of the latest version of the current model
    lvm_path = get_latest_pytorch_model(model_dirpath)

    return lvm_path


def print_idl(idls):
    r"""
    Args:
        idls: idleness vector to print
    """

    s = '['
    for i, idl in enumerate(idls):
        s += "{}: {}, ".format(i, idls[i])

    print("DBG: ", s[:-1] + ']')


def build_soc_name(strategy: str,
                   nagts: int):
    r"""
    Args:
        strategy:
        nagts:
    """
    return "soc_{}_{}".format(strategy, nagts)
