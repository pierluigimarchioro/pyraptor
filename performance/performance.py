"""
This script provides an interface to test RAPTOR performances
It computes journeys with different:
    - origin and destination stops and time departure
    - raptor settings
"""
import argparse
import json
import os
from os import path
from typing import Dict, List, Mapping

from loguru import logger

from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import RaptorTimetable
from pyraptor.query import query_raptor
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME, SHARED_MOB_TIMETABLE_FILENAME
from pyraptor.util import mkdir_if_not_exists

# i/o
OUT_FILE = "performance_out.json"  # output file with performance info
TMP_DIR = "tmp"  # temporary directory to save wmc_config file
WMC_CONFIG = "wmc_config.json"  # temporary weight configuration file name

# MACROS for configuration file names

# rounds
ROUNDS = 'rounds'
ROUND_START = 'min'
ROUND_END = 'max'

# journeys
JOURNEYS = 'journeys'
JOURNEYS_ORIGIN = 'from'
JOURNEYS_DESTINATION = 'to'
JOURNEYS_TIME = 'at'

# settings
SETTINGS = 'settings'
SETTINGS_VARIANT = 'variant'
SETTINGS_ENABLE_SM = 'enable_sm'
SETTINGS_SM = 'sm'
SETTINGS_WEIGHTS = 'weights'

CONFIG_KEYS: set[str] = {ROUNDS, JOURNEYS, SETTINGS}


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


class SettingHandler:
    """ Handle raptor query setting in terms of RAPTOR variant and shared mobility enabling"""

    def __init__(self, d: Dict, out_dir: str, tmp_dir: str):
        """

        :param d:
        :param out_dir:
        """
        self.d = d
        self.out_dir = out_dir  # save output
        self.tmp_cfg_file = path.join(tmp_dir, WMC_CONFIG)  # temporary config file
        self.variant = d[SETTINGS_VARIANT]
        self.enable_sm = d[SETTINGS_ENABLE_SM]
        self.handle_variant()

    def handle_variant(self):
        """ If it uses weighted-multi-criteria variant it stores
            configuration file to temporary directory """
        if self.variant == 'wmc':
            weights = self.d[SETTINGS_WEIGHTS]
            _dict_to_json(weights, self.tmp_cfg_file)

    def run(self, timetable: RaptorTimetable, timetable_sm: RaptorTimetableSM,
            journey_config: JourneyConfig, rounds: int) -> float:
        """
        R
        :param timetable: basic timetable
        :param timetable_sm: timetable containing shared-mobility info
        :param journey_config: object with departure and arrival stop and arrival time
        :param rounds: number of rounds
        :return: execution time
        """
        elapsed_time = query_raptor(
            variant=self.variant,
            timetable=timetable_sm if self.enable_sm else timetable,  # type of timetable based on shared-mob enabling
            output_folder=self.out_dir,
            origin_station=journey_config.origin,
            destination_station=journey_config.destination,
            departure_time=journey_config.departure_time,
            rounds=rounds,
            criteria_config=self.tmp_cfg_file,
            enable_sm=self.enable_sm,
            preferred_vehicle=self.d[SETTINGS_SM]['preferred_vehicle'] if self.enable_sm else None,
            enable_car=self.d[SETTINGS_SM]['enable_car'] if self.enable_sm else None
        )

        return elapsed_time


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


def main(input_: str, output: str, config: str):
    logger.debug("Timetable directory         : {}", input_)
    logger.debug("Output directory            : {}", output)
    logger.debug("Configuration file          : {}", config)

    tmp_dir = path.join(output, TMP_DIR)
    mkdir_if_not_exists(output)
    mkdir_if_not_exists(tmp_dir)

    d: Dict = _json_to_dict(config)

    # Reading both timetables
    timetable: RaptorTimetable = read_timetable(input_, TIMETABLE_FILENAME)
    timetable_sm: RaptorTimetableSM = read_timetable(input_, SHARED_MOB_TIMETABLE_FILENAME)

    # Rounds
    round_start, round_end = d[ROUNDS]['min'], d[ROUNDS]['max']

    # Journeys
    journeys: List[JourneyConfig] = []
    for journey in d[JOURNEYS]:
        journeys.append(JourneyConfig(d=journey))

    # Init dict
    # number of round : raptor variant : shared-mobility enabling : (start_stop, end_stop, dep_time): list of elapsed times
    out_dict: Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        k: {v: {sm: {str(j): [] for j in journeys}
                for sm in ['shared_mobility', 'normal']
                }
            for v in ['basic', 'wmc']
            }
        for k in range(round_start, round_end + 1)
    }

    # TODO since raptor computes from 1 to K rounds: can we skip external for loop and access all bags
    for k in range(round_start, round_end + 1):  # rounds
        for setting in d[SETTINGS]:  # algos
            setting_handler = SettingHandler(setting, output, tmp_dir)
            for journey in journeys:
                time = setting_handler.run(timetable, timetable_sm, journey, k)
                out_dict[k][setting_handler.variant]['shared_mobility' if setting_handler.enable_sm else 'normal'][
                    str(journey)].append(time)

    _dict_to_json({'rounds': out_dict}, path.join(output, OUT_FILE))

    os.remove(path.join(output, "algo-output.pcl"))
    os.remove(path.join(tmp_dir, WMC_CONFIG))
    os.rmdir(tmp_dir)


def _parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Timetable directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Output directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="performance/performance_config.json.sample",
        help="Configuration file",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = _parse_arguments()

    main(
        args.input,
        args.output,
        args.config
    )
