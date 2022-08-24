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
from pyraptor.util import sec2str


class Graph:
    def __init__(self, adjac_list, heuristic, timetable: RaptorTimetable, dep_time):
        self.adjacency_list = adjac_list
        self.heuristic = heuristic
        self.timetable = timetable
        self.departure = dep_time

    def is_int(self, v):
        return isinstance(v, int)

    def both_int(self, v1, v2):
        return self.is_int(v1) and self.is_int(v2)

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

        # curr_dist contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        curr_time = {start: self.departure}

        # parents contain an adjacency map of all nodes
        parents = {start: start}

        durations = {start: 0}
        arrival_times = {start: self.departure}

        to_delete = 0
        while len(open_lst) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function, earliest arrival time
            for v in open_lst:
                if n is None \
                        or curr_time[v] + self.heuristic[v] < curr_time[n] + self.heuristic[n]:
                    n = v

            if n is None:  # to delete, it should impossible to reached
                print('Path does not exist!')
                return None

            # if the current node is the stop print journey
            if n == stop:
                path_found = []
                duration = 0
                times = []

                while parents[n] != n:
                    path_found.append(self.timetable.stops.get_stop(n).name)
                    duration = duration + durations[n]
                    times.append(arrival_times[n])
                    n = parents[n]

                path_found.append(self.timetable.stops.get_stop(start).name)
                path_found.reverse()
                duration = duration + durations[start]
                times.append(arrival_times[start])
                times.reverse()

                print('Path found: {}'.format(path_found))
                print('duration: ', sec2str(duration))
                print('arrival time: ', times)

                return path_found

            # for all the neighbors of the current node do
            for step in self.get_neighbors(n):

                # if n == "A_CADORNA FN M1":
                #     print("found") #quando è che torna indietro a controllare? tempo di arrivo sicuro brutto
                # if n == "A_PAGANO" and step.stop_to.name == "BUONARROTI" and step.departure_time == 44155:
                #     print("time found") # problema che non trova da qua bisceglie qt8

                if not self.is_int(step.departure_time) \
                        or arrival_times[n] <= step.departure_time:

                    # if the current node is not present in both open_lst and closed_lst
                    # add it to open_lst and note n as it's parents
                    if step.stop_to.id not in open_lst and step.stop_to.id not in closed_lst:
                        open_lst.add(step.stop_to.id)

                        if self.is_int(step.arrive_time):  # this cover all hours and waiting time
                            arrival_times[step.stop_to.id] = step.arrive_time
                            curr_time[step.stop_to.id] = step.arrive_time  # old: curr_time[n] + step.duration
                        else:
                            arrival_times[step.stop_to.id] = curr_time[n] + step.duration
                            curr_time[step.stop_to.id] = curr_time[n] + step.duration
                        parents[step.stop_to.id] = n
                        durations[step.stop_to.id] = step.duration  # waiting time is not added

                    # otherwise, check if it's quicker to first visit n, than step
                    # and if it is, update parents data and curr_dist data
                    # and if the node was in the closed_lst, move it to open_lst
                    else:
                        if (self.both_int(curr_time[step.stop_to.id], curr_time[n])
                                and curr_time[step.stop_to.id] > curr_time[n] + step.duration):

                            if self.is_int(step.arrive_time):
                                arrival_times[step.stop_to.id] = step.arrive_time
                                curr_time[step.stop_to.id] = step.arrive_time
                            else:
                                arrival_times[step.stop_to.id] = curr_time[n] + step.duration
                                curr_time[step.stop_to.id] = curr_time[n] + step.duration
                            parents[step.stop_to.id] = n
                            durations[step.stop_to.id] = step.duration
                            # todo consider to make a method instead of these 2 blocks of the same code

                            if step.stop_to.id in closed_lst:
                                open_lst.add(step.stop_to.id)
                                closed_lst.remove(step.stop_to.id)

            # if to_delete == 138:
            #     print()
            print(n, to_delete)
            to_delete = to_delete+1
            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)

        print('Path does not exist!')
        return None

    # todo salvare la sequenza di step fatti
    # todo try to not use "x" ad dep_time and arr_time in transfers
