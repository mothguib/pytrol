# -*- coding: utf-8 -*-

from pytrol.model.action.Action import Action

from pytrol.model.action.Actions import Actions


class GoingToAction(Action):
    def __init__(self, _goal_position: tuple):
        r"""
        Args:
            _goal_position (tuple):
        """
        Action.__init__(self, 'going_to', Actions.Going_to)
        self.goal_position = _goal_position
