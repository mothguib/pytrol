# Master 2 ANDROIDE
# CoCoMa Projet 2017 - 2018
# Thomas Lenoir
# Malcolm Auffray

import numpy as np

from pytrol.control.agent.Coordinator import Coordinator
from pytrol.model.knowledge.EnvironmentKnowledge import EnvironmentKnowledge
from pytrol.util.net.Connection import Connection


class HCCoordinatorOld(Coordinator):
    r"""
    Args:
        id (int):
        original_id (str):
        env_knl (EnvironmentKnowledge):
        connection (Connection):
        agts_addrs (list):
        variant (str):
        depth (float):
        interaction (bool):
    """
    def __init__(self,
                 id: int,
                 original_id: str,
                 env_knl: EnvironmentKnowledge,
                 connection: Connection,
                 agts_addrs: list,
                 variant: str = '',
                 depth: float = 3.0,
                 interaction: bool = True):

        Coordinator.__init__(self, id, original_id, env_knl, connection,
                             agts_addrs, variant, depth,
                             interaction=interaction)

        self.min_max_dist = self.get_min_max_dist()
        self.d = True

        self.r = 0.5
        self.tick = -1
        #dictionnary to save the data about each agent :

        # id => current position, goal position, index_path, path
        # 
        self.knowledge_agents = {}

        self.real_idlness = [0 for _ in range(len(self.env_knl.idls))] 
    

    #this function is called when this agent receive a message.
    def strategy_decide(self):
        super().strategy_decide()

        #while there some messages
        while len(self.to_send) > 0:
            # For each entry e in the list of dictionaries to_sen
            e = self.to_send.popleft()
            id_agnt = e["a_id"]
            pos_agnt = e["pos"]


            # test if the agnt is in its goal or if he doesn't have a goal
            if id_agnt not in self.knowledge_agents \
                or self.equal_triplet(self.knowledge_agents[id_agnt]["goal"],pos_agnt):
                # in this case calculte new goal node : 

                # Update the idleness when the agent arrives in its goal.
                self.real_idlness[pos_agnt[0]] = 0

                # get the min and the max idleness.
                i_min = min(self.real_idlness)
                i_max = max(self.real_idlness)

                # if first time to receive a message from this agent
                # initialize the data.
                if id_agnt not in self.knowledge_agents:
                    self.knowledge_agents[id_agnt] = {}
                    self.knowledge_agents[id_agnt]["pos"] = pos_agnt
                    self.knowledge_agents[id_agnt]["goal"] = None
                    self.knowledge_agents[id_agnt]["index_path"] = 0
                    self.knowledge_agents[id_agnt]["path"] = None


                # calcule the heuristics for each nodes
                heuristics = []
                for i, idleness in enumerate(self.real_idlness):
                    # get the distance between his position and the node i.
                    dist = self.env_knl.ntw.v_dists[pos_agnt[0]][i]
                    heuristics.append(self.heuristic_decision_makin(idleness, i_max,i_min,dist))


                # get the node with the min heuristics.
                min_h = min(heuristics)
                #print("min_h = ",min_h," ",i_max," ",i_min," "," idle =",idleness)
                argmins = []
                for i in range(len(heuristics)):
                    if heuristics[i] == min_h:
                        argmins.append(i)

                goal = (argmins[np.random.randint(0,len(argmins))],-1,0)

                # save and send the goal
                self.knowledge_agents[id_agnt]["goal"] = goal
                self.knowledge_agents[id_agnt]["index_path"] = 0

                self.knowledge_agents[id_agnt]["path"] = self.env_knl.ntw.path(pos_agnt, goal)
                self.send("goal_position:"+ str(goal), id_agnt)
                #print(id_agnt," goal =  ",goal)



    ## applied at each turn to update the idleness and the position of agents
    def prepare(self):

        # if there is a difference between the tick of the coordinator and
        # the current tick in the simulation
        # update the idleness of each nodes.
        #print("tick = ",self.tick," t = ",self.env_knl.t)
        if self.tick != self.env_knl.tick:
            for i in range(len(self.env_knl.idls)):
                self.real_idlness[i] += (self.env_knl.t - self.tick)
            self.tick = self.env_knl.t
            self.real_idlness[self.pos[0]]=0

            ## check if an agent is positionning in a Node which is not his goal
            ## in this case the idleness must be put 0
            #print("--------------------------------------------")

            for (key,agnt) in self.knowledge_agents.items():
                #path = self.env_knl.ntw.path(agnt["pos"], agnt["goal"])
                agnt["index_path"]+= 1

                if agnt["index_path"] < len(agnt["path"]):
                    agnt["pos"] = agnt["path"][agnt["index_path"]]
                    if agnt["pos"][1] == -1 : #check if on a node => edge = -1
                        self.real_idlness[agnt["pos"][0]] = 0


                #print(key," = " ,agnt["pos"]," goal = ",agnt["goal"], agnt["index_path"])
                #print("path = ",agnt["path"])
        
        # receive messages: 
        #print("idleness =",self.real_idlness)
       




    def equal_triplet(self,t1,t2):
        """
        Args:
            t1:
            t2:
        """
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]


    # method calls at the begining to calculate the min and max distance in the graph.
    def get_min_max_dist(self):
        max_d = -1
        min_d = -1
        # print(len(self.env_knl.ntw.vertices))
        # print(len(self.env_knl.ntw.v_dists))

        for i in range(len(self.env_knl.ntw.v_dists)):
            for j in range(len(self.env_knl.ntw.v_dists)):
                if i != j:
                    length_u_v = self.env_knl.ntw.v_dists[i][j]
                    if length_u_v > max_d or max_d == -1:
                        max_d = length_u_v
                    elif length_u_v < min_d or min_d == -1:
                        min_d = length_u_v
        return (min_d, max_d)

    ## return a coeff in [0,1] indicating the lenght of the distance.
    ## The more the will be close to 1 the more the distance d0 is big.
    def time_to_go_normalized(self, d0):
        """
        Args:
            d0:
        """
        return (d0 - self.min_max_dist[0]) / (self.min_max_dist[1] - self.min_max_dist[0])

    ## v0 is an id of a node
    def idleness_normalized(self, idle, max_i, min_i):
        """
        Args:
            idle:
            max_i:
            min_i:
        """
        if min_i - max_i == 0:
            return 0
        return (idle - max_i) / (min_i - max_i)

    ## r * I(v,t) * (1-r) * d(v0  , v)
    ## where v is a node, v0 is agent's node
    def heuristic_decision_makin(self, idle, idle_max, idle_min, dist):
        """
        Args:
            idle:
            idle_max:
            idle_min:
            dist:
        """
        return self.r * self.idleness_normalized(idle, idle_max, idle_min) + (1 - self.r) * self.time_to_go_normalized(
            dist)

 