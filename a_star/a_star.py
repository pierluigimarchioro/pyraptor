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


# This is heuristic function which is having equal values for all nodes
def h(n):  # todo lo creo a mano perche mi serve # devo fargli leggere l'euristica che creo io
    heu = {
        'A': 1,
        'B': 1,
        'C': 1,
        'D': 1
    }

    return heu[n]


class Step(object):
    """
    The Stop object. Can be interpreted as a node in the structure.
    Includes positional values and weighted values as colors.
    Can be visited and not visited (bool).
    Can be the end if the node is the goal tile (bool).
    """

    # usando gli id delle fermate, e facile ricavare le info usando la timetable
    def __init__(self, duration, arrive_time, route_id, transport_type):
        """
        Initializes the stop node
        :param duration: duration to reach a stop
        :param arrive_time: arrive time to a stop
        :param route_id: route id of this step
        :param transport_type: transport type of this route (route_type : Indicates the type of transportation used on a route)
        """

        self.duration = duration
        self.arrive_time = arrive_time
        self.route_id = route_id
        self.transport_type = transport_type

    def set_duration(self, duration):
        """set a new duration value"""
        self.duration = duration

    def set_arrive_time(self, arrive_time):
        """set a new arrive time"""
        self.arrive_time = arrive_time

    def set_route_id(self, route_id):
        """set a new route id"""
        self.route_id = route_id

    def set_transport_type(self, transport_type):
        """set a new transport type"""
        self.transport_type = transport_type



# todo leggere il gtfs e ricavarmi tutte le fermate in set di Stop
# filtrare i servizi attivi in quella data
# cosi posso usarle per calcolare l'euristica
# + estrarre l'ordine delle route e vari pesi (distanza)
# ora non ricordo come trattare gli orari e che per uno stop passano piu route
# rileggere il pdf
def process_gtfs(self):  # -> TBDclass:
    """
    Creates and saves a GTFS containing only the stops included in the journey provided to this instance,
    starting from the GTFS originally used to calculate the Raptor timetable and the aforementioned journey.
    Returns an object containing the coordinates of the first and the last stop of the journey.

    :return: object containing the coordinates of the first and the last stop of the journey
    """

    # todo non ricorda cosa deve fare


# def manhattan_heuristic(stop1, stop2) -> float:
#     """
#     Manhattan distance between two stops
#     :param stop1: Stop
#     :param stop2: Stop
#     :return: float distance
#     """
#     (x1, y1) = (stop1.lat, stop1.long)
#     (x2, y2) = (stop2.lat, stop2.long)
#     return abs(x1 - x2) + abs(y1 - y2)
#
#
# def euclidean_heuristic(stop1, stop2) -> float:
#     """
#     Euclidean distance between two stops
#     :param stop1: Stop
#     :param stop2: Stop
#     :return: float distance
#     """
#     return np.linalg.norm(stop1 - stop2)


def get_heuristic(destination, timetable: RaptorTimetable) -> dict[str, int]:
    """
    Time = distance/speed [hour]
    we use the average public transport speed in Italy [km/h]
    reference:
    https://www.statista.com/statistics/828636/public-transport-trip-speed-by-location-in-italy/
    we won't use Manhattan distance or Euclidean distance since we want the fastest travel not the shortest

    :param destination: destination station
    :param timetable: timetable
    :return: assign an heuristic value to every stops
    """
    heuristic = {}
    avg_speed = (14 + ((47 + 56) / 2)) / 2

    # a differenza di raptor io passo id non il nome
    # todo mi conviene calcolarlo per tutte le destinazioni ?
    for st in timetable.stops:
        heuristic[st.id] = (st.distance_from(timetable.stops.get_stop(destination)) / avg_speed)*3600
    return heuristic


# adjac_lis = {
#     'A': [('B', Step object), ('C', Step object), ('D', Step object)],
#     'B': [('D', Step object)],
#     'C': [('D', Step object)]
# }
# TODO contiene i nodi vicini non tutta la route
def get_adj_list(timetable: RaptorTimetable) -> dict[str, list[tuple[str, int]]]:
    adjacency_list = {}

    # todo
    # per ogni fermata
    #   trovo tutte le fermate che riesce a raggiungere tramite un trip
    #   mi salvo la fermata vicina + duration, arrive_time, route_id, transport_type
    #   in qualche modo me li ricavo, transport type sta in route.txt, arrive_time e route_id ci sono gia stop_times.txt
    #   duration me la calcolo, come?
    #   duration deve essere calcolato anche considerando possibili momenti di attesa alla fermata per prendere il mezzo

    return adjacency_list


# todo da adattare l'algoritmo all'uso di Step invece che un peso diretto
class Graph:
    def __init__(self, adjac_list):
        self.adjacency_list = adjac_list

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
                if n is None or curr_dist[v] + h(v) < curr_dist[n] + h(n):
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


if __name__ == "__main__":
    adjacency_list = {
        'A': [('B', 1), ('C', 3), ('D', 7)],
        'B': [('D', 5)],
        'C': [('D', 12)]
    }
    graph1 = Graph(adjacency_list)
    graph1.a_star_algorithm('A', 'D')

# if __name__ == "__main__":
#     point1 = np.array((45.45506133419171,9.113051698074148, 0))
#     point2 = np.array((45.486488320854384,9.136206849656572, 0))
#     print(euclidean_heuristic(point1,point2))
#
#     point1 = np.array((45.46432595988547,9.18700965522402, 0))
#     point2 = np.array((45.48443970103568,9.202612807879362, 0))
#     print(euclidean_heuristic(point1,point2))

# Todo usare metodi da preprocessing e ritornare un output scritto
# Todo visualizzazione in folium



# come gestire piu route che passano per la stessa fermata --> penso risolto
# raptor normale non tiene conto del numero di cambi che fa? --> ottimizza solo il tempo, puo fare anche 12344 cambi
# capire come gestire il tempo (se deve aspettare ad una stazione devo tenerne conto)
# transfers ci sta il tempo di percorrenza in secondi --> devo dire all'algoritmo di considerare i transfers ( sono tutti a piedi) (in caso di eccezione metto un valore di default)
# euristica e peso non stessa scala --> velocita media usando la distanza in linea d'aria cosi ho tutto in secondi


# todo forse mi conviene salvare la lista delle fermate
# dato che l'euristica viene calcolata da tutti i punti verso la destinazione

# todo salvarmi l'ordine delle route
# capire se ci mette tanto a leggere il gtfs

# todo le route sono diverse per mezzo di trasporto
# quindi se cambia route do un peso in piu, ma quanto?
# uso il tempo come peso, bisogna calcolare i tempi "buchi" (quando cambia route)