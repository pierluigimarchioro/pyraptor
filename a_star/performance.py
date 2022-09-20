"""
This script provides an interface to test A STAR performances
It computes journeys with different origin and destination stops and departure time
"""
from __future__ import annotations

import argparse
import json
import os
from os import path
from typing import Dict, List, Mapping
from timeit import default_timer as timer

from loguru import logger

import a_star
from preprocessing import Step
from preprocessing import get_heuristic
from preprocessing import read_adjacency

from pyraptor.model.timetable import RaptorTimetable
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME
from pyraptor.util import mkdir_if_not_exists, str2sec

OUT_FILE = "performance_out_astar.json"  # output file with performance info
TMP_DIR = "tmp"  # temporary directory to save wmc_config file

# journeys
JOURNEYS = 'journeys'
JOURNEYS_ORIGIN = 'from'
JOURNEYS_DESTINATION = 'to'
JOURNEYS_TIME = 'at'

CONFIG_KEYS: set[str] = {JOURNEYS}


class JourneyConfig:
    """ Following class represent a journey describing from stop, arrival stop and departure time"""

    def __init__(self, d: Mapping[str, str]):
        """
        Saves configuration file information
        :param d: 'journeys' dictionary from configuration file
        """
        self.origin: str = d[JOURNEYS_ORIGIN]
        self.destination: str = d[JOURNEYS_DESTINATION]
        self.departure_time: str = d[JOURNEYS_TIME]

    def __str__(self) -> str:
        return f"[FROM {self.origin} TO {self.destination} AT {self.departure_time}]"


def main(input_: str, adjacent: str, output: str, config: str):
    logger.debug("Timetable directory         : {}", input_)
    logger.debug("Adjacency list directory    : {}", adjacent)
    logger.debug("Output directory            : {}", output)
    logger.debug("Configuration file          : {}", config)

    tmp_dir = path.join(output, TMP_DIR)
    mkdir_if_not_exists(output)
    mkdir_if_not_exists(tmp_dir)

    d: Dict = _json_to_dict(config)

    # Reading timetable
    timetable: RaptorTimetable = read_timetable(input_, TIMETABLE_FILENAME)

    # Reading adjacency list
    adjlst_start_time = timer()
    adjacency_list = read_adjacency(adjacent)
    adjlst_end_time = timer()

    # Journeys
    journeys: List[JourneyConfig] = []
    for journey in d[JOURNEYS]:
        journeys.append(JourneyConfig(d=journey))

    # Init dict
    k = "Adjacency list time: {}".format(adjlst_end_time-adjlst_start_time)
    out_dict: Dict[str, Dict[str, List[float]]] = {k: {str(j): [] for j in journeys}}

    for journey in journeys:
        # find paths
        path_start_time = timer()
        heuristic = get_heuristic(journey.destination, timetable)
        graph = a_star.Graph(adjacency_list, heuristic, timetable, str2sec(journey.departure_time))
        destination_journeys = graph.a_star_algorithm(journey.origin, journey.destination)
        path_end_time = timer()
        time = path_end_time - path_start_time

        out_dict[k][str(journey)].append(time)

    _dict_to_json(out_dict, path.join(output, OUT_FILE))

    os.rmdir(tmp_dir)


def _json_to_dict(file: str) -> Dict:
    """
    Convert a json to a dictionary
    :param file: path to json file
    :return: data as a dictionary
    """

    return json.load(open(file))


def _dict_to_json(data: Dict, path_: str):
    """
    Store a dictionary locally as a .json file
    :param data: path to json file
    .param path_: path to store file
    :return: data as a dictionary
    """
    json.dump(data, open(path_, 'w'))


def _parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output/milan",
        help="Timetable directory",
    )
    parser.add_argument(
        "-a",
        "--adjacent",
        type=str,
        default="data/output/milan/a_star",
        help="Adjacency list directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="performance/out",
        help="Output directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="performance/in/performance_config_astar.json",
        help="Configuration file",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = _parse_arguments()
    main(
        args.input,
        args.adjacent,
        args.output,
        args.config
    )
