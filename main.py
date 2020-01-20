# -*- coding: utf-8 -*-

import time
from mem_top import mem_top

import pytrol.util.misc as misc
import pytrol.util.argsparser as parser
from pytrol.control.Ananke import Ananke
from pytrol.control.Archivist import Archivist
from pytrol.util import pathformatter
from pytrol.util.net.SimulatedConnection import SimulatedConnection

# Using `mem_top`
memtop = False


def main(graph: str,
         strategy: str,
         variant: str,
         datasrc: str,
         nagts: int,
         duration: int,
         soc_name: str,
         exec_id: int,
         dirpath_execs: str,
         dirpath_logs: str,
         depth: int,
         trace_agents: bool,
         trace_agents_estimates: bool,
         interaction: bool):

    # If the agents' estimates are traced, then their individual idlenesses
    # are also traced
    trace_agents = True if trace_agents_estimates else trace_agents

    variant_sfx = '_' + variant if variant != '' else ''

    print('{}: # {}, {}{}, {} agents, #{}\n' \
          .format(misc.timestamp(), graph, strategy, variant_sfx, nagts,
                  exec_id))

    if interaction:
        print("interaction mode: depth: {}".format(depth))

    exec_path = pathformatter.build_exec_path(graph=graph, strt=strategy,
                                              exec_id=exec_id, nagts=nagts,
                                              soc_name=soc_name,
                                              execs_rep=dirpath_execs)

    log_path = pathformatter.build_log_path(graph=graph, strt=strategy,
                                            duration=duration, exec_id=exec_id,
                                            soc_name=soc_name, datasrc=datasrc,
                                            nagts=nagts, logs_rep=dirpath_logs,
                                            variant=variant)

    # Archivist's connection
    ar_cnt = SimulatedConnection()

    # Archivist
    archivist = Archivist(ar_cnt, log_path, duration, trace_agents,
                          trace_agents_estimates)

    # archivist.start()

    start = time.time()

    # Ananke's connection
    an_cnt = SimulatedConnection()

    ananke = Ananke(an_cnt, exec_path, archivist, duration, depth, graph,
                    nagts, variant, trace_agents, trace_agents_estimates,
                    interaction)
    ananke.start()
    ananke.join()

    end = time.time()
    print(misc.timestamp(), ": Time: ", (end - start), '\n')

    print(misc.timestamp(), ": `main.py`: memory usage: ",
          misc.get_memusage(), '% \n')

    if memtop:
        print(misc.timestamp(), ": `main.py`: Showing of top  "
                                "suspects for memory leaks in your "
                                "Python program with `mem_top`:")
        print("{}:{}".format(misc.timestamp(),
                             mem_top(limit=10, width=100, sep='\n',
                                     refs_format='{''num}\t{type} {obj}',
                                     bytes_format='{num}\t {obj}',
                                     types_format='{num}\t {obj}',
                                     verbose_types=None,
                                     verbose_file_name="logs/mem_top.txt")),
              '\n'
              )

    print("{}: ----------------------------------\n" \
          .format(misc.timestamp(), misc.get_memusage()))


# Executed only if run as a script
if __name__ == '__main__':
    args = parser.parse_args()

    # main(dirpath_execs=Paths.LOCALEXECS, duration=10, exec_id=1,
    #     strategy="rhple")
    main(graph=args.map,
         strategy=args.strategy,
         variant=args.variant,
         datasrc=args.datasrc,
         nagts=args.nagts,
         duration=args.duration,
         soc_name=args.society,
         exec_id=args.execid,
         dirpath_execs=args.dirpath_execs,
         dirpath_logs=args.dirpath_logs,
         depth=args.depth,
         trace_agents=args.trace_agents,
         trace_agents_estimates=args.trace_agents_estimates,
         interaction=args.interaction)
