# -*- coding: utf-8 -*-

import os
import re

from pytrol.util import misc as misc


def build_execs_dirpath(graph: str,
                        strt: str,
                        nagts: int,
                        execs_rep: str):
    """Builds the directory path of executions for a given patrolling scenario
    `{graph, nagts, strt}` .

    Args:
        graph:
        strt:
        nagts:
        execs_rep: the repository where the runs are stored
    """

    return regularise_path("{}/{}/{}/{}/".
                           format(execs_rep, graph, strt, str(nagts)))


def build_exec_file_name(graph: str,
                         strt: str,
                         nagts: int,
                         exec_id: int,
                         soc_name: str = None):
    """Builds the execution file name of id `exec_id` for the given patrolling
    scenario `{graph, nagts, strt}` .

    Args:
        graph:
        strt:
        nagts:
        exec_id:
        soc_name:
    """

    if soc_name is None or soc_name == '':
        soc_name = misc.build_soc_name(strategy=strt, nagts=nagts)

    return regularise_path("{}-{}-{}-{}-{}.json".format(strt,
                                                        graph,
                                                        soc_name,
                                                        str(nagts),
                                                        str(exec_id)))


def build_exec_path(graph: str,
                    strt: str,
                    nagts: int,
                    exec_id: int,
                    execs_rep: str,
                    soc_name: str = None):
    """Builds the execution file path of id `exec_id` for the given patrolling
    scenario `{graph, nagts, strt}` .

    Args:
        graph:
        strt:
        nagts:
        exec_id:
        execs_rep: the repository where the tuns are stored
        soc_name:
    """

    return regularise_path(
        "{}/{}".format(build_execs_dirpath(graph=graph, strt=strt,
                                           nagts=nagts,
                                           execs_rep=execs_rep),
                       build_exec_file_name(graph=graph, strt=strt,
                                            nagts=nagts, exec_id=exec_id,
                                            soc_name=soc_name)))


def build_logs_dirpath(graph: str,
                       strt: str,
                       nagts: int,
                       logs_rep: str,
                       variant: str = None,
                       datasrc: str = None,
                       duration: int = 3000):
    """Builds the logs' directory path of the scenario/configuration {<graph>,
    <nagts>, <strt>_<variant>}.

    Args:
        graph:
        strt:
        nagts:
        variant:
        datasrc:
        duration:
        logs_rep: repository where logs are stored
    """

    if variant is None:
        variant = ''

    if datasrc is None:
        datasrc = ''

    strt = "{}_{}".format(strt, variant) if variant != '' \
        else "{}".format(strt)

    return regularise_path("{}/{}/{}/{}/{}/{}/".
                           format(logs_rep, graph, strt, datasrc,
                                  str(nagts), duration))


def build_log_file_name(graph: str,
                        strt: str,
                        nagts: int,
                        variant: str,
                        exec_id: int = -1,
                        duration: int = 3000,
                        ext: str = "log",
                        soc_name: str = None):
    """Build the log file name of a scenario execution id `exec_id` , if the
    latter is provided, otherwise the standard log file name is returned.

    Args:
        graph:
        strt:
        nagts:
        variant:
        exec_id:
        duration:
        ext:
        soc_name:
    """
    if soc_name is None or soc_name == '':
        soc_name = misc.build_soc_name(strategy=strt, nagts=nagts)

    return regularise_path("{}_{}-{}-{}-{}-{}-{}.{}.json".
                           format(strt, variant, graph, soc_name, str(nagts),
                                  duration, exec_id, ext) if exec_id != -1
                           else "{}_{}-{}-{}-{}-{}.{}.json".
                           format(strt, variant, graph, soc_name, str(nagts),
                                  duration, ext)
                           )


def build_log_path(graph: str,
                   strt: str,
                   nagts: int,
                   variant: str,
                   logs_rep: str,
                   datasrc: str = None,
                   exec_id: int = -1,
                   duration: int = 3000,
                   ext: str = "log",
                   soc_name: str = None):
    r"""Build a scenario's log file path of execution id `exec_id` .

    Args:
        graph:
        strt:
        nagts:
        variant:
        datasrc:
        exec_id:
        duration:
        logs_rep: repository where logs are stored
        ext:
        soc_name:
    """

    return regularise_path(
        "{}/{}".format(build_logs_dirpath(graph=graph, strt=strt,
                                          variant=variant, datasrc=datasrc,
                                          nagts=nagts, duration=duration,
                                          logs_rep=logs_rep),
                       build_log_file_name(graph=graph, strt=strt,
                                           variant=variant,
                                           nagts=nagts, exec_id=exec_id,
                                           duration=duration,
                                           soc_name=soc_name,
                                           ext=ext)))


def build_stat_path(graph: str,
                    strt: str,
                    nagts: int,
                    statpath: str,
                    nrm: bool,
                    datasrc: str = None,
                    duration: int = 3000):
    r"""Builds the statistic file path for the current scenario.

    Args:
        graph:
        strt:
        nagts:
        statpath:
        nrm:
        datasrc:
        duration:
    """

    if datasrc is None:
        datasrc = ''

    ext = "norm.stat" if nrm else "stat"

    return regularise_path("{}/{}/{}-{}-{}-{}.{}".
                           format(statpath, datasrc, graph, strt, int(nagts),
                                  int(duration), ext))


def regularise_path(path: str):
    """
    Args:
        path (str):
    """
    return re.sub(r"\/$", '', re.sub(r"\/+", "/", path))


def filepath_to_exec_params(p: str) -> (str, str, str, str, int, int, int):
    r"""Returns the parameters of an execution from its file path.

    Args:
        p: execution's path
    """

    # The execution's file name
    fn = regularise_path(p).split('/')[-1]

    # The split execution's file name
    sfn = fn.split('-')

    # The split strategy name
    sstrt = sfn[0].split('_')
    strt = sstrt[0]
    variant = ''.join(sstrt[1:]) if len(sstrt) > 1 else ''

    # The map
    graph = sfn[1]

    # The society name
    soc_name = sfn[2]

    # The number of agents
    nagts = int(sfn[3])

    # The duration
    duration = int(sfn[4])

    # The execution id
    execid = int(sfn[5].split('.')[0])

    return strt, variant, graph, soc_name, nagts, duration, execid


def dirpath_to_exec_params(p: str, soc_name: str = None) -> (str, str, str,
                                                             str, int, int):
    r"""Returns the parameters of a scenario from its directory path.

    Args:
        p: scenario's directory path
        soc_name:
    """
    # The split path
    sp = regularise_path(p).split('/')

    duration = int(sp[-1])
    nagts = int(sp[-2])
    # The split strategy name
    sstrt = sp[-3].split('_')
    strt = sstrt[0]
    variant = sstrt[1]
    graph = sp[-4]

    if soc_name is None:
        soc_name = misc.build_soc_name(strategy=strt, nagts=nagts)

    return strt, variant, graph, soc_name, nagts, duration


def timestamp_from_filename(fn: str):
    """Return the timestamp of the file `fn` .

    Args:
        fn (str):
    """

    return int(regularise_path(fn).split('/')[-1].split('.')[1])


def execid_from_filename(fn: str):
    r"""Returns the execution id of the file `fn` .

    Args:
        fn:
    """

    return int(regularise_path(fn).split('/')[-1].
               split('-')[-1].
               split('.')[0])


def means_path(g: str,
               n: int,
               datasrc: str,
               mean_dirpath: str):
    r"""

    Args:
        g:
        n: the number of agents
        datasrc:
        mean_dirpath:
    """

    fp = "{}/{}/{}-{}-{}.means.json".format(mean_dirpath, datasrc, datasrc,
                                            g, n)

    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))

    return fp

