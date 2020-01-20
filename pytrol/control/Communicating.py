# coding -*- coding: utf-8 -*-

from collections import deque
from threading import Thread

from pytrol.util.net.Connection import Connection


class Communicating(Thread):
    def __init__(self, cnt_: Connection):
        r"""This class, which extends `threading.Thread` and is thereby a
        thread, provides all of the abstract methods necessary to
        communicate, and thus allows creating independent threads able to
        communicate. Any `Communicating` object will be referred to as
        *communicating*. Any possible way of communication is practicable,
        letting the user provide a connection, i.e. an object whose the
        class extends the `utils.net.Connection.Connection` abstract class.
        Thus, the type of needed connection is left to the discretion of the
        user. By default, the concrete class
        `utils.net.SimulatedConnection.SimulatedConnection` is used,
        enabling *communicatings* to communicate by reference., i.e. by
        memory address.

        Args:
            cnt_ (Connection):
        """

        Thread.__init__(self)

        # Network and messages
        self.cnt = cnt_

        self.messages = deque()

        # Communicating cycle completed
        self.ac = False

        # Communication mode: if `True`, ready to communicate and it cannot
        # terminate its current cycle
        self.c = False

        self.stop_working = False

    #  Messages
    def send(self, message, address):
        r"""Sends a message to address `address` . `address` ought to own a the
        same a connection of the same `Connection` type than the current sender
        .

        Args:
            message: the message string
            address: the address of the object recipient
        """
        self.cnt.send(message, address)

    def receive(self):
        r"""Empties the buffer and adds the messages to the mailbox
        `self.messages` .
        """
        buffer = self.cnt.get_buffer_and_flush()
        for m in buffer:
            self.messages.append(m)

    '''
    def __str__(self):

        indent_str_1 = misc.indent(0)
        indent_str_2 = misc.indent(2)

        s = indent_str_1 + self.__class__.__name__ + ":\n" \
            + indent_str_2 + "Action completed : " + str(self.ac) + "\n\n" \
            + indent_str_2 + "Stop working : " + str(self.stop_working) + \
            "\n\n"

        return s
    '''