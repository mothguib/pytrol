# -*- coding: utf-8 -*-

from pytrol.model import Paths


class Maps:
    City_Traffic = 0

    Islands = 1

    A = 2

    B = 3

    Circle = 4

    Corridor = 5

    Grid = 6

    id_to_name = ["city_traffic", "islands", "map_a", "map_b", "circle",
                  "corridor", "grid"]

    name_to_id = {"city_traffic": 0,
                  "islands": 1,
                  "map_a": 2,
                  "map_b": 3,
                  "circle": 4,
                  "corridor": 5,
                  "grid": 6}

    id_to_path = [Paths.JSON_MAPS + n + ".json" for n in id_to_name]
