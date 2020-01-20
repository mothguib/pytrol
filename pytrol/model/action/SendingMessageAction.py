# -*- coding: utf-8 -*-

from pytrol.model.action.Action import Action

from pytrol.model.action.Actions import Actions


class SendingMessageAction(Action):
    def __init__(self, _message: str, _agt_id: int):
        r"""
        Args:
            _message (str):
            _agt_id (int):
        """
        Action.__init__(self, "sending_message", Actions.Sending_message)
        self.message = _message
        self.agt_id = _agt_id




