# -*- coding: utf-8 -*-

from pytrol.control.agent.Coordinator import Coordinator
from pytrol.control.agent.HPAgent import HPAgent
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


class HPCCoordinator(Coordinator, HPAgent):
    def __init__(self, id_: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 situated: bool = False,
                 interaction: bool = False):

        r"""
        Args:
            id_ (int):
            original_id (str):
            env_knl (EnvironmentKnowledge):
            connection (Connection):
            agts_addrs (list):
            variant (str):
            depth (float):
            situated (bool):
            interaction (bool):
        """
        if variant is None or variant == '':
            variant = "0.5"

        interaction = False  # A HPCCoordinator agent never applies interaction
        # procedures because it already interacts by design

        Coordinator.__init__(self, id_=id_, original_id=original_id,
                             env_knl=env_knl, connection=connection,
                             agts_addrs=agts_addrs, variant=variant,
                             depth=depth, situated=situated,
                             interaction=interaction)

        HPAgent.__init__(self, id_=id_, original_id=original_id,
                         env_knl=env_knl, connection=connection,
                         agts_addrs=agts_addrs,
                         variant=variant, depth=depth, interaction=interaction)

    def strategy_decide(self):
        super().strategy_decide()

        self.prev_estimated_idls = self.estimated_idls

        self.estimated_idls = self.estimate_idls()

        # DBG
        # print("DBG: HPCCordinator's idlenesses:", self.estimated_idls)
        for e in self.to_send:
            # DBG
            # print("DBG: strategy_decide: message of a coordinated:", e)

            # TODO: converting the representation of node position from the 3D
            # into the scalar

            # DBG
            # print("DBG: strategy_decide: self.hr_agt_goal_pos[e['a_id']]: ",
            #       self.hr_agt_goal_pos[e["a_id"]])

            if self.hr_agt_goal_pos[e["a_id"]] == (-1, -1, 0) \
                    or self.hr_agt_goal_pos[e["a_id"]] == e["pos"]:
                # The heuristic method is applied in `process_idls`
                prcssd_idls = self.process_idls(self.estimated_idls, e["pos"])

                self.hr_agt_goal_pos[e["a_id"]] = self.select_goal_pos(
                    prcssd_idls, self.hr_agt_goal_pos, e["pos"])

            # The Pathfinder path is figured out at each decision step i.e. at
            # each node due to idlenesses evolving continuously.
            pathfinder_path = self.pathfinder(idls=self.estimated_idls,
                                              edges_lgts=self. \
                                              env_knl.ntw.edges_lgts,
                                              agt_pos=e["pos"],
                                              goal_pos=self. \
                                              hr_agt_goal_pos[e["a_id"]],
                                              r=self.r,
                                              graph=self.env_knl.ntw.graph)

            next_pos = self.pathfinder_next_pos(pathfinder_path)

            # DBG
            # print("DBG: ", next_pos)

            e["next_pos"] = next_pos

    def coordinator_act(self):
        if len(self.to_send) > 0:
            while len(self.to_send) > 0:
                # For each entry e in the list of dictionaries to_send
                e = self.to_send.popleft()
                self.send("next_position:{}".format(str(e["next_pos"])),
                          e["a_id"])
