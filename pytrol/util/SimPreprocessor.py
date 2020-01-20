import json
import os
import random
import numpy as np
import sys
import untangle

from pytrol import SEED
from pytrol.model import Paths

import pytrol.util.misc as misc
from pytrol.model.Maps import Maps
from pytrol.model.AgentTypes import AgentTypes

from maptor.control.XMLReader import XMLReader
from maptor.control.MapLoader import MapLoader


class SimPreprocessor:
    r"""Various tools and methods to prepare patrolling configurations,
    scenarios or runs. In the context of MAP, we distinguish the notions of
    *patrolling configuration* and *patrolling scenario*. Here, a *patrolling
    configuration*, *MAP configuration* or simply *configuration*,
    noted `K`, is any unordered pair `K = {G, N_a}`. For example, `{Grid, 15}`
    is a patrolling configuration of 15 agents on the topology Grid. Then,
    a *patrolling scenario*, *MAP scenario* or simply *scenario*, is any
    triplet :math:`{\Pi, G, N_a\}` such as :math:`\Pi` is a MAP strategy.
    In fact, a MAP scenario is the pair of a MAP strategy and a MAP
    configuration. For example, `{HPCC, Grid, 15}` is a MAP scenario of 15
    HPCC agents on the topology Grid.

    Likewise, a *mission*, an *execution* or a *run*, is an execution of a
    MAP scenario with specific initial conditions. These initial conditions
    correspond to a positioning of agents on nodes of the graph to patrol.
    For non-deterministic strategies the seeds of random sorting functions are
    also part of the initial conditions. For the present time, initial
    positions on edge are not considered.

    A *society of agents* is a set of run agents, and *metasociety of
    agents* is a set of societies, useful for runs where agents can change
    of society, etc."""

    @classmethod
    def load_run(cls, file_path: str):
        if file_path.endswith("xml"):
            return cls.config_xml_to_iterables(file_path)
        elif file_path.endswith("json"):
            with open(file_path) as file:
                return cls.config_dict_to_iterables(json.load(file))
        else:
            raise ValueError("Only JSON and XML files can be loaded.")

    @classmethod
    def simplified_generate_runs(cls,
                                 strts: list,
                                 nb_execs: int = 30,
                                 odp: str = '',
                                 netids: list = None,
                                 pop_sizes: list = None):
        """
        Simplified run generation: generates patrollings instances according to
        the provided strategies `strts`, the number of executions
        `nb_execs`, the output directory path `odp`, the network ids,
        i.e. the networks to patrol, `netids`, and the number of agents
        `pop_sizes`. For the same number of agents and network, each run
        has the same starting position distribution of agents over the
        vertices. This enables comparing different strategies for the same
        networks with the same initial conditions. This method is a
        simplified version of `generate_runs`.

        Args:
            strts: 2D list of strategies. The 1st dimension represents the
            different strategies for which the run will be generated,
            whereas the 2nd one represents the *sub-strategies*,
            when necessary, in particular for the coordinated strategies.
            For a coordinated strategy there are 2 sub-strategies on the
            second dimension.
            nb_execs: number of runs,
            odp: output directory path
            netids: list of networks ids to patrol
            pop_sizes: number of agents
        """

        if pop_sizes is None:
            pop_sizes = [1, 5, 10, 15, 25]

        mats = []
        cdtrss = []
        nbs_agtss = []

        for t in strts:
            for p in pop_sizes:
                mats += [[t * p] if len(t) == 1
                         else [[t[0]] + [t[1]] * p]]

                nbs_agtss += [[p]]

                cdtrss += [[True]] if len(t) == 2 else [[False]]

        cls.generate_runs(nbs_agtss, mats, cdtrss,
                          nb_execs=nb_execs, odp=odp, netids=netids)

    @classmethod
    def generate_runs(cls,
                      nbs_agtss: list = None,
                      mats: list = None,
                      cdtrs: list = None,
                      mns: list = None,
                      sns: list = None,
                      nb_execs: int = 10,
                      odp: str = '',
                      netids: list = None,
                      pseudorand: bool = True):
        r"""Generates patrolling runs/missions.

        Args:
            nbs_agtss: 2D list standing for the number of agents for each
                society within each metasociety. Each element along the 1st
                dimension corresponds to a metasociety and each element along
                the 2nd dimension corresponds to a society.
            mats: 3D list of agent strategies
            cdtrs: boolean 2D list representing whether there is a
                coordinator for each society (2nd dimension) and that for each
                metasociety (1st dimension)
            mns: 1D list of the meta-societies' names
            sns: 2D list of the societies' names (2nd dimension) for each
            nb_execs: number of executions, i.e. runs/missions, to generate
            odp: output directory path, i.e. the path of the folder where
                the execution files are generated
            netids: 1D list of network ids for which generating runs
            pseudorand: if `True`, use seed `SEED` for random generation

        """

        if nbs_agtss is None:
            nbs_agtss = [[]]
        if mats is None:
            mats = [[[]]]
        if cdtrs is None:
            cdtrs = [[False] * len(m) for m in mats]
        if mns is None:
            mns = []
            for i in range(len(nbs_agtss)):
                # Each i represents the id of the current list of numbers
                # of agents numbers i.e. the current meta-society

                # TODO: changing the naming pattern for suffix of this script:
                # types of agents instead of their number
                mns += [misc.name_metasociety(nbs_agtss[i], str(i))]
        if sns is None:
            sns = []
            for i in range(len(nbs_agtss)):
                sn = []
                for j in range(len(nbs_agtss[i])):
                    sfx = str(j) + "_" + str(nbs_agtss[i][j])
                    sn += [misc.name_metasociety([nbs_agtss[i][j]], sfx)]
                sns += [sn]
        if netids is None:
            maps = list(range(1, 7))
        else:
            maps = [Maps.name_to_id[m] for m in netids]

        if len(nbs_agtss) != len(mats):
            raise ValueError(
                "The number of meta-societies in the mats list must "
                "be equal to the bunches of numbers of agents numbers in "
                "nbs_agtss list")

        # `sps_ids` for `same population size ids`: a list that gathers the ids
        #  of `mats` having the same coupling of population sizes,
        # each coupling of population sizes corresponding to a coupling of
        # societies compsing a meta-society
        sps_ids = misc.identical_values(nbs_agtss)

        if pseudorand:
            random.seed(SEED)

        for m in maps:
            for e in range(nb_execs):
                for ids in sps_ids:
                    # Number of agents in all societies for the current
                    # meta-society
                    na = sum(nbs_agtss[ids[0]]) + 1  # " +1 ": addition of a
                    # further vertex for the case where the strategy has a
                    # coordinator

                    graph = cls.load_map(m)

                    # Vertices to the dict format randomly selected on which
                    # agents will stand
                    vs = random.sample([v["id"] for v in graph["vertices"]],
                                       na)

                    for i in ids:
                        # Generation of a blank configuration structured for
                        #  the mats whose indices are in the ids' list
                        config, op = cls.generate_scenario(m,
                                                           nbs_agtss[i],
                                                           mats[i],
                                                           cdtrs[i],
                                                           mns[i],
                                                           sns[i],
                                                           odp=odp)

                        # List of dict vertices
                        exc = cls. \
                            inject_agts_positions_in_config(config=config,
                                                            vs=vs)

                        op = op.replace("blank", str(e))

                        with open(op, 'w') as s:
                            # s as stream
                            json.dump(exc, s)

                        print(op, " generated.")

    @classmethod
    def generate_scenario(cls, m: int, nbs_agts: list = None,
                          sats: list = None, cdtrs: list = None,
                          mn: str = '', sn: list = None,
                          op: str = '', sp: str = None, odp: str = '') \
            -> (dict, str):
        r"""Generates a patrolling scenario file, i.e. a blank run file in
        which agents have no positions.

        Args:
            m: the map id. See `model.Maps`
            sp: societies's file path, i.e. the path of the file containing
                the societies; if not provided the societies are generated
                automatically according to the other arguments
            nbs_agts: list of agent numbers, one for each society
            mn: the metasociety name
            sn: list of the societies' names
            sats: 2D list of agent strategies for each society: each element
                on the 1st dimension corresponds to a society
            cdtrs: boolean 1D list representing whether there is a
                coordinator for each society
            op: scenario's output path
            odp: scenario's output directory path

        Returns:
            The path of the generated scenario file.
        """

        # TODO: raising an error if sats is an empty 2D list

        if nbs_agts is None:
            nbs_agts = []
        if odp == '':
            odp = Paths.LOCALEXECS + '/'
        if op == '':
            op = misc.name_config_file(m, nbs_agts, sats, sfx="blank",
                                       odp=odp)

        config = {"environment": {"graph": cls.load_map(m)}}

        if sp is not None:
            # TODO: take into account the case where a config file of the
            # society is provided
            pass
        else:
            config["environment"]["meta-society"] = \
                {"id": mn,
                 "societies": cls.generate_societies(nbs_agts, sats, cdtrs,
                                                     sn, op)
                 }

        directory = os.path.dirname(op)
        if not os.path.exists(directory):
            os.makedirs(directory)

        return config, op

    @staticmethod
    def load_map(m: int):
        with open(Maps.id_to_path[m]) as s:
            return json.load(s)

    @staticmethod
    def generate_societies(nbs_agts: list = None, sats: list = None,
                           cdtrs: list = None, sn: list = None, op: str = ""):
        if nbs_agts is None:
            nbs_agts = []
        if sats is None:
            sats = [[]]
        if cdtrs is None:
            cdtrs = [False] * len(nbs_agts)
        if sn is None:
            sn = []
            for i in range(len(nbs_agts)):
                sn += [misc.name_metasociety([nbs_agts[i]], str(i))]

        if len(nbs_agts) != len(sn):
            raise ValueError("The number of society names in the sn list "
                             "must be equal to the number of agents numbers "
                             "in the nagts list")
        if len(nbs_agts) != len(sats):
            raise ValueError("The number of societies in the sats list must "
                             "be equal to the number of numbers of agents in "
                             "nagts list")
        for i in range(len(nbs_agts)):
            if not cdtrs[i] and len(sats[i]) != nbs_agts[i] or \
                    cdtrs[i] and len(sats[i]) != nbs_agts[i] + 1:
                raise ValueError("Each number of types in the sats 2D list "
                                 "must be equals to the corresponding number "
                                 "of agents in the nagts list")
            if len(sats[i]) == 1:
                sats[i] = sats[i] * nbs_agts[i]

        socs = []
        for i in range(len(nbs_agts)):
            soc = {"id": sn[i], "label": sn[i], "is_closed": "true"}
            agts = []

            if cdtrs[i]:
                agts += [{"id": "coordinator", "type": sats[i][0],
                          "vertex_id": ""}]
            else:
                agts += [{"id": "0", "type": sats[i][0],
                          "vertex_id": ""}]

            for j in range(1, nbs_agts[i] + (1 if cdtrs[i] else 0)):
                agts += [{"id": str(j), "type": sats[i][j], "vertex_id": ""}]

            soc["agents"] = agts
            socs += [soc]

        if op != "":
            pass

        return socs

    @classmethod
    def inject_agts_positions_in_config(cls, config: dict, vs: list) -> dict:

        new_config = {**config}

        # Next vertice from the vs list to insert
        nv = 0
        for s in new_config["environment"]["meta-society"]["societies"]:
            cls.inject_agts_positions_in_soc(s, vs[nv:nv + len(s["agents"])])
            # `vs[nv:nv + len(s["agents"])]`: only the number of vertices
            # corresponding to the number of agents in that society is passed.
            #  If the strategy is not a coordinated strategy then the
            # further vertex is not passed to the method.

            nv = len(s["agents"])

        return new_config

    @staticmethod
    def inject_agts_positions_in_soc(soc: dict, vs: list):
        # Taken into account of the case where the strategy has a coordinator
        if soc["agents"][0]["id"] == "coordinator":
            soc["agents"][0]["vertex_id"] = vs[-1]
            for i in range(1, len(soc["agents"])):
                soc["agents"][i]["vertex_id"] = vs[i - 1]
        else:
            for i in range(len(soc["agents"])):
                soc["agents"][i]["vertex_id"] = vs[i]

    @staticmethod
    def generate_config():
        pass

    @classmethod
    def convert_config_xml_to_json(cls, xml_path: str, json_path: str):
        with open(json_path, 'w') as s:
            # s as stream
            json.dump(cls.config_xml_to_dict(xml_path)[0], s)

    @classmethod
    def convert_map_xml_to_json(cls, xml_path: str, json_path: str):
        with open(json_path, 'w') as s:
            # s as stream
            json.dump(XMLReader.map_xml_to_dict(xml_path), s)

    @classmethod
    def convert_socs_xml_to_json(cls, xml_path: str, json_path: str):
        with open(json_path, 'w') as s:
            # s as stream
            json.dump(XMLReader.socs_xml_to_dicts(xml_path)[0], s)

    @classmethod
    def config_xml_to_dict(cls, path: str) -> (dict, dict):
        agt_type = path.split("/")
        agt_type = agt_type[len(agt_type) - 1].split("-")[0]
        doc = untangle.parse(path)
        return cls.config_untangle_to_dict(doc, agt_type)

    @classmethod
    def config_xml_to_iterables(cls, path: str) -> tuple:
        return cls.config_dict_to_iterables(cls.config_xml_to_dict(path))

    @classmethod
    def config_untangle_to_dict(cls, doc: untangle.Element, agt_type: str) \
            -> (dict, dict):
        socs_dict = XMLReader.socs_untangle_to_dicts(doc.environment,
                                                     agt_type)[0]

        return {"environment": {"graph": XMLReader.map_untangle_to_dict(
            doc.environment),
            "meta-society": {"id": "_",
                             "societies": socs_dict
                             }
        }
        }

    @classmethod
    def config_dict_to_iterables(cls, d: dict) -> (np.ndarray, np.ndarray,
                                                   dict, dict, np.ndarray,
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray, list, list,
                                                   list, dict):
        r"""
        Args:
            d: dictionary of the whole configuration
        """

        graph, fl_edges_lgts, vertices, edges, edges_to_vertices, locations, \
        edge_activations, idls = \
            MapLoader.map_dict_to_iterables(d["environment"]["graph"])

        socs, speeds_societies, agts_pos_societies, socs_to_ids = \
            cls.socs_dict_to_iterables(d)

        return graph, fl_edges_lgts, vertices, edges, edges_to_vertices, \
               locations, edge_activations, idls, socs, speeds_societies, \
               agts_pos_societies, socs_to_ids

    @classmethod
    def socs_dict_to_iterables(cls, d: dict) -> (list, list, list, dict):
        r"""

        Args:
            d: dictionary of the whole configuration
        """
        vertices = MapLoader.map_dict_to_iterables(d["environment"][
                                                       "graph"])[2]

        socs_d = d["environment"]["meta-society"]["societies"]

        # `socs` stands for the list of societies as dictionaries containing
        #  the society ids and associated agents
        socs = []
        # List of dictionaries standing for the speed of agents for each
        # society
        speeds_societies = []
        # List of dictionaries standing for the position of agents for each
        # society
        agts_pos_societies = []
        # `socs_to_ids`: the dictionary mapping soc ids to a dictionary of
        # agent ids in that society
        socs_to_ids = {}

        counter_s = 0
        # `soc_d` stands for the dictionary of the current society retrieved
        #  from `socs_d`, itself standing for the dictionary of all societies
        #  retrieved from the argument `d` being the dictionary of an
        # execution setting
        for soc_d in socs_d:
            # `s`: the dictionary representing the current society
            s = {"id": soc_d["id"], "agents": []}
            socs_to_ids[soc_d["id"]] = {"int_id": counter_s, "agt_ids": {}}
            speeds = np.empty([0], dtype=np.float16)
            agts_pos = np.empty([0, 3], dtype=np.int16)

            # Testing if the position of agents in the society is defined
            pos_def = True
            if len(soc_d["agents"]) != 0:
                if soc_d["agents"][0]["vertex_id"] == "" \
                        or soc_d["agents"][0]["vertex_id"] == "a" \
                        or soc_d["agents"][0]["vertex_id"] == "c":
                    pos_def = False

            counter_a = 0
            for agt_dic in soc_d["agents"]:  # `agt_dic` stands for a
                #  dictionary containing the features of an agent

                a = {"id": counter_a, "original_id": agt_dic["id"],
                     "type": AgentTypes.str_to_id[agt_dic["type"]],
                     "vertex_id": vertices[agt_dic["vertex_id"]] if
                     pos_def else -1}

                s["agents"].append(a)
                agts_pos = np.append(agts_pos, [[a["vertex_id"], -1, 0]],
                                     axis=0)
                speeds = np.append(speeds, [1], axis=0)

                socs_to_ids[soc_d["id"]]["agt_ids"][agt_dic["id"]] = \
                    counter_a
                counter_a += 1

            speeds_societies.append({"id": s["id"], "speeds": speeds})
            agts_pos_societies.append({"id": s["id"], "agts_pos": agts_pos})
            socs.append(s)

            counter_s += 1

        # `swap_coord` puts the coordinator to the first position
        socs, socs_to_ids = XMLReader.swap_coord(socs, socs_to_ids)

        return socs, speeds_societies, agts_pos_societies, socs_to_ids

    @classmethod
    def convert_xmls_to_jsons(cls, xmls_path: str, jsons_path: str):
        for root, dirs, files in os.walk(xmls_path):
            for name in files:
                cls.convert_map_xml_to_json(xmls_path + name,
                                            (jsons_path + name).replace(
                                                "xml", "json"))
