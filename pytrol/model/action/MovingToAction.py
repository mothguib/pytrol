# -*- coding: utf-8 -*-

from pytrol.model.action.Action import Action

from pytrol.model.action.Actions import Actions


class MovingToAction(Action):
    def __init__(self, _frm: tuple, _to: tuple):
        r"""
        Args:
            _frm (tuple): from
            _to (tuple): iterable
        """
        Action.__init__(self, 'moving_to', Actions.Moving_to)
        self.frm = _frm
        self.to = _to
