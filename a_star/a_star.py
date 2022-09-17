from __future__ import annotations

import joblib
import os

from pyraptor.model.timetable import RaptorTimetable, TimetableInfo
from pyraptor.util import str2sec, sec2str, mkdir_if_not_exists
from pathlib import Path
from loguru import logger


class Graph:
    def __init__(self, adjac_list, heuristic, timetable: RaptorTimetable, dep_time):
        self.adjacency_list = adjac_list
        self.heuristic = heuristic
        self.timetable = timetable
        self.departure = dep_time

    def is_int(self, v):
        return isinstance(v, int)

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
                times = []
                all_durations= []
                tot_duration = 0

                while parents[n] != n:
                    path_found.append(self.timetable.stops.get_stop(n).name)
                    tot_duration = tot_duration + durations[n]
                    times.append(curr_time[n])
                    all_durations.append(durations[n])
                    n = parents[n]

                path_found.append(self.timetable.stops.get_stop(start).name)
                path_found.reverse()
                tot_duration = tot_duration + durations[start]
                times.append(curr_time[start])
                times.reverse()
                all_durations.append(durations[n])
                all_durations.reverse()

                print('Path found:')
                for s, t, d in zip(path_found, times, all_durations):
                    print('Stop: {} - Arrival time: {} - Duration: {}'.format(s, sec2str(t), sec2str(d)))
                print('total duration: ', sec2str(tot_duration))

                return path_found

            # for all the neighbors of the current node do
            for step in self.get_neighbors(n):
                # if n == "QT8" and step.stop_to.name == "qt8 m1" and step.departure_time == 44155:
                #     print("time found")

                if not self.is_int(step.departure_time) \
                        or curr_time[n] <= step.departure_time:

                    # if the current node is not present in both open_lst and closed_lst
                    # add it to open_lst and note n as it's parents
                    if step.stop_to.id not in open_lst and step.stop_to.id not in closed_lst:
                        open_lst.add(step.stop_to.id)

                        if self.is_int(step.arrive_time):  # this cover all hours and waiting time
                            curr_time[step.stop_to.id] = step.arrive_time  # old: curr_time[n] + step.duration
                            durations[step.stop_to.id] = step.duration + (step.departure_time - curr_time[n])
                        else:
                            curr_time[step.stop_to.id] = curr_time[n] + step.duration
                            durations[step.stop_to.id] = step.duration
                        parents[step.stop_to.id] = n

                    # otherwise, check if it's quicker to first visit n, than step
                    # and if it is, update parents data and curr_dist data
                    # and if the node was in the closed_lst, move it to open_lst
                    else:
                        if (self.is_int(step.arrive_time) and curr_time[step.stop_to.id] > step.arrive_time) \
                                or (not self.is_int(step.arrive_time) and curr_time[step.stop_to.id] > curr_time[n] + step.duration):

                            if self.is_int(step.arrive_time):
                                curr_time[step.stop_to.id] = step.arrive_time
                                durations[step.stop_to.id] = step.duration + (step.departure_time - curr_time[n])
                            else:
                                curr_time[step.stop_to.id] = curr_time[n] + step.duration
                                durations[step.stop_to.id] = step.duration
                            parents[step.stop_to.id] = n
                            # todo consider to make a method instead of these 2 blocks of the same code

                            if step.stop_to.id in closed_lst:
                                open_lst.add(step.stop_to.id)
                                closed_lst.remove(step.stop_to.id)

            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)

        print('Path does not exist!')
        return None

    # todo salvare la sequenza di step fatti
    # todo note down when route or means of transport changes
    # todo save stop order to give as input to folium for visualization


class AstarOutput(TimetableInfo):
    """
    Class that represents the data output of a Raptor algorithm execution.
    Contains the best journey found by the algorithm, the departure date and time of said journey
    and the path to the directory of the GTFS feed originally used to build the timetable
    provided to the algorithm.
    """

    _DEFAULT_FILENAME = "algo-output"

    # journeys: Iterable[Journey] = None # TODO fare un iterable dell'oggetto che contiene le fermate? o trovare tutti gli archi
    """Best journey found by the algorithm"""

    departure_time: str = None
    """string in the format %H:%M:%S"""

    # not adapted to a_star yet because it's not used anyway
    @staticmethod
    def read_from_file(filepath: str | bytes | os.PathLike) -> AstarOutput:
        """
        Returns the AlgorithmOutput instance read from the provided folder
        :param filepath: path to an AlgorithmOutput .pcl file
        :return: AlgorithmOutput instance
        """

        def load_joblib() -> AstarOutput:
            logger.debug(f"Loading '{filepath}'")
            with open(Path(filepath), "rb") as handle:
                return joblib.load(handle)

        if not os.path.exists(filepath):
            raise IOError(
                "PyRaptor AlgorithmOutput not found. Run `python pyraptor/query_raptor`"
                " first to generate an algorithm output .pcl file."
            )

        logger.debug("Using cached datastructures")

        algo_output: AstarOutput = load_joblib()

        return algo_output

    @staticmethod
    def save(output_dir: str | bytes | os.PathLike,
             algo_output: AstarOutput,
             filename: str = _DEFAULT_FILENAME):
        """
        Write the algorithm output to the provided directory

        :param output_dir: path to the directory to write the serialized output file
        :param algo_output: instance to serialize
        :param filename: name of the serialized output file
        """

        logger.info(f"Writing PyRaptor output to {output_dir}")
        mkdir_if_not_exists(output_dir)

        with open(Path(output_dir, f"{filename}.pcl"), "wb") as handle:
            joblib.dump(algo_output, handle)
