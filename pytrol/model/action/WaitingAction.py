# -*- coding: utf-8 -*-

from pytrol.model.action.Action import Action

from pytrol.model.action.Actions import Actions


class WaitingAction(Action):
    def __init__(self, _pos: tuple):
        r"""
        Args:
            _pos (tuple):
        """
        Action.__init__(self, "waiting", Actions.Waiting)
        self.pos = _pos




