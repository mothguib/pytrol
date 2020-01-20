from abc import ABC, abstractmethod

import pytrol.util.misc as misc


class Connection(ABC):

    def __init__(self):
        # The buffer where the connection writes the received messages.
        self.BUFFER = []

    def get_buffer_and_flush(self) -> list:
        r"""Returns the buffer, then flushes it"""

        answer = self.BUFFER[:]
        del self.BUFFER[:]

        return answer

    @abstractmethod
    def send(self, message: str, address):
        r"""Sends a message to address `address` . `address` ought to own the
        same `Connection` type as the current sender.

        Args:
            message (str): the message string
            address: the address of the object recipient
        """
        pass

    '''
    @abstractmethod
    def receive(self):
        pass
    '''

    def to_string(self, indent_1: int, indent_2 : int):
        r"""
        Args:
            indent_1 (int):
            indent_2 (int):
        """
        indent_str_1 = misc.indent(indent_1)
        indent_str_2 = misc.indent(indent_2)
        print(indent_str_1 + "Connection:\n",
              indent_str_2 + "Number of BUFFER elements:", len(self.BUFFER),
              "\n")

