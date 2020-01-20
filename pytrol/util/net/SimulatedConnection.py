# -*- coding: utf8 -*-

from pytrol.util.net.Connection import Connection

import pytrol.util.misc as misc


class SimulatedConnection(Connection):
    r"""Connection working by exchanging directly messages in memory on the
    same machine.
    """

    def __init__(self):
        Connection.__init__(self)

        '''
        # The simulated input stream received from others agents
        self.SIMULATED_INPUT = []
        '''

    def send(self, message: str, address: object):
        r"""Sends a message to memory address, i.e. reference, `address` .
        `address` ought to own a `SimulatedConnection` connection.

        Args:
            message (str): the message string
            address (object): the reference of the object recipient
        """
        address.cnt.BUFFER.append(message)

    '''
    def receive(self):
        for m in self.SIMULATED_INPUT:
            self.BUFFER.append(m)
        del self.SIMULATED_INPUT[:]
    '''

    def to_string(self, indent_1: int, indent_2: int):

        r"""
        Args:
            indent_1 (int):
            indent_2 (int):
        """
        indent_str_1 = misc.indent(indent_1)
        indent_str_2 = misc.indent(indent_2)

        super(self).to_string(indent_1, indent_2)
        print(indent_str_1 + "SimulatedConnection:\n",
              indent_str_2 + "Number of SIMULATED_INPUT elements:", "\n")

