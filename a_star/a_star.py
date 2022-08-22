from collections import deque
import geopy
import numpy as np
import pandas as pd
import pathos.pools as p
from loguru import logger
from typing import List, Iterable, Any, NamedTuple, Tuple, Callable, Dict, TypeVar
import itertools
import json
import math
import os
import uuid
import argparse
from typing import Dict

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.timetable import RaptorTimetable
from pyraptor.model.output import Journey, AlgorithmOutput
from pyraptor.model.algos.raptor import (
    RaptorAlgorithm,
    reconstruct_journey,
    best_stop_at_target_station,
)
from pyraptor.util import str2sec


class Graph:
    def __init__(self, adjac_list, heuristic):
        self.adjacency_list = adjac_list
        self.heuristic = heuristic

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def a_star_algorithm(self, start, stop):
        # In this open_lst is a list of nodes which have been visited, but who's
        # neighbours haven't all been always inspected,
        # It starts off with the start node
        # And closed_lst is a list of nodes which have been visited
        # and who's neighbors have been always inspected
        open_lst = {start}
        closed_lst = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        curr_dist = {start: 0}

        # parents contain an adjacency map of all nodes
        parents = {start: start}

        while len(open_lst) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_lst:
                if n is None or curr_dist[v] + self.heuristic[v] < curr_dist[n] + self.heuristic[n]:
                    n = v

            if n is None:
                print('Path does not exist!')
                return None

            # if the current node is the stop
            # then we start again from start
            if n == stop:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all the neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node is not present in both open_lst and closed_lst
                # add it to open_lst and note n as it's parents
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    parents[m] = n
                    curr_dist[m] = curr_dist[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parents data and curr_dist data
                # and if the node was in the closed_lst, move it to open_lst
                else:
                    if curr_dist[m] > curr_dist[n] + weight:
                        curr_dist[m] = curr_dist[n] + weight
                        parents[m] = n

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)

            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)

        print('Path does not exist!')
        return None
