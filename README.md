PyTrol
------

*PyTrol* is a discrete-time simulator dedicated to MAP, designed as
a Python framework with the aim of performing MAP simulation.
Originally, this simulator has been developed for the purpose of
generating merely and rapidly a significant amount of MAP data, for the
needs of this work. It has thereafter been evolved to become an
easy-to-extend framework totally dedicated to MAP: the user can write
custom building blocks to express and experiment new ideas for MAP
research; they can also develop state-of-the-art models. It is
distributed over multiple threads, and can be used sequentially or
parallelly. Its temporal model is discrete and each time step, or
period, is not finished as long as all the agents have not acted.

To simulate a MAP mission, that is to say an instance of a given MAP
scenario {$\Pi$, $G$, $N_a$}, a JSON file containing the setting of the
mission to simulate shall be provided to the simulator. More precisely,
the JSON file contains the description of the graph $G$, the society of
agents, as well as their initial positions on the graph. The duration of
the mission $T \in \mathbb{N}^*$ ought also to be set. For each mission,
simulation traces are recorded in JSON log files, in which at each time
step the position and the individual idlenesses of each agent, as well
as the true idlenesses of nodes are logged. These files are then
processed to compute statistics, or even to be used as data for
learning, for example.

In this framework, edges are discretised. They are sampled over time, that is to say
divided into units that agents travel in one period.

In its current version, PyTrol relies on $5$ variables:

-   the *position of agents* in the graph,

-   the *completed actions*: a boolean variable which indicates whether
    all agents have completed their action,

-   the *communication step*: a boolean variable which indicates whether
    the communication step has begun, outside this step agents cannot
    communicate,

-   the *decision step*: a boolean variable which indicates whether the
    decision step has begun, outside this step agents cannot decide,

-   the *interaction mode*: a boolean variable which indicates whether
    the agents can interact, i.e. whether the interaction scheme can be
    used.

### Main components

PyTrol comes in the form of a python package called `pytrol`, which is
itself decomposed into three main subpackages, as follows:

-   `control`: represents the controller, namely the component of PyTrol
    which executes agents and controls all operations necessary to play
    the simulation out,

-   `model`: represents the data model of MAP, namely all concepts and
    objects which determine the structure of MAP necessary to simulate a
    MAP execution,

-   `util`: utilities, that is to say annex tools, procedures and
    algorithms being generic enough to be used in other projects
    independent from PyTrol.

### `pytrol.control.Communicating`

A key structure in PyTrol is the
`pytrol.control.Communicating.Communicating` class. This class, which
extends `threading.Thread`, provides all of the abstract methods
necessary to communicate, and thereby allows creating independent
threads able to communicate. Any `Communicating` object will be referred
to as *communicating*. Any possible way of communication is practicable,
letting the user provide an object whose the class extends the
`utils.net.Connection.Connection` abstract class. Thus, the type of
needed connection is left to the discretion of the user. By default, the
concrete class `utils.net.SimulatedConnection.SimulatedConnection` is
used, enabling *communicatings* to communicate by reference, i.e. memory
address.

### `pytrol.control.agent`

Before continuing, it is worth recalling that any agent strategy is, in
fact, an algorithm. In the context of this work a *multiagent
strategy* is merely defined as a set of $N_a$ single-agent strategies.

`pytrol.control.agent` contains agent strategy implementations. Any new
implemented strategy shall extend the `Agent` class located in the
`pytrol.control.agent. Agent` module, and be added in this package.
`Agent` is an abstract class defining a template for any agent strategy.
This template defines, in fact, the basic procedure that any agent must
follow. This basic procedure, qualified as *main procedure of agent*,
represents the life cycle of agents and consists of:

-   `Agent.prepare`: any preprocessing, if necessary, the agent needs to
    carry out to prepare the impending main procedure,

-   `Agent.perceive`: the agent perceives the position of the other
    ones, if required by its strategy; in the strategies studied in this
    dissertation only the position of the agent itself is perceived,
    although other types of perception are left to the discretion of the
    user,

-   `Agent.communicate`: the agent communicates with other ones, if
    required by its strategy;

-   `Agent.analyse`: the agent checks and processes messages he has
    received,

-   `Agent.decide`: the agent decides; this method constitutes the core
    of the strategy, given that any strategy is a decision-making
    procedure in the context of MAP,

-   `Agent.act`: the agent acts according to the decision made in the
    previous method.

Each agent, namely each object instantiating the `Agent` class, is a
*communicating* and therefore a thread; concretely the `Agent` class
extends the `Communicating` class. Any new strategy to add in PyTrol
shall be implemented from the above methods, then added to the
`pytrol.control.agent` package, and finally referenced in
`pytrol.model.AgentTypes`. A set of strategies are already implemented
in PyTrol:

-   CR in `pytrol.control.agent.CR`,

-   HPCC in `pytrol.control.agent.HPCCoordinator` and `pytrol.control.agent.Coordinated`,

-   HCC in `pytrol.control.agent.HCCoordinator`,

-   strategies based on machine learning that extend the abstract class
    `pytrol.control. agent.MAPTrainerModelAgent`.

In the implementation of HPCC studied in this dissertation and coded in
PyTrol, agents request a new goal node each time they arrive at a node;
for each agent the Heuristic and Pathfinder algorithms are therefore
executed by the coordinator each time they arrive at a node. With regard
to the implementation of HCC, the Warshall's algorithm is executed at
the simulation's startup to compute once and for all the shortest
distances and paths between the nodes.

### `pytrol.control.Ananke`

The `pytrol.control.Ananke.Ananke`[^1] class is the core of PyTrol, i.e.
the structure which concretely handles the simulation running. It is
also a *communicating*.

The life cycle of `Ananke` starts with the mission's initialisation
which takes place in `Ananke.__init__`, where it loads the graph, all
information relative to the current mission, as well as the agents.

Then, in `Ananke.run` the main simulation loop over the time steps is
executed. There is as many iterations in this loop as the duration $T$
set for the current run. This loop stands for the running of simulation:
at each period, the strategy of agents simulated herein is deployed.
More precisely, at each iteration Ananke executes the main procedure of
the strategy by calling, for every agent, the methods described above
which constitutes their life cycle.

### `pytrol.control.Archivist`

The `pytrol.control.Archivist.Archivist` class gives rise to a
communicating object which logs the running of the simulation, that is
as stated above, the positions, individual idlenesses of each agent, and
true idlenesses. Complementary MAP elements or events to log might be
added.

[^1]: *Ananke* is an ancient Greek goddess who was the personification
    of inevitability, compulsion and necessity, and in other terms, of
    what must happen.
