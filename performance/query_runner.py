"""
This script provides an interface to test RAPTOR performances
It computes journeys with different:
    - origin and destination stops and time departure
    - raptor settings
"""
from __future__ import annotations

import argparse
import json
import os.path
import random as rnd
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Dict, List, Type, Tuple

import pandas as pd
from loguru import logger

from pyraptor.model.criteria import Criterion, ArrivalTimeCriterion, TransfersCriterion, DistanceCriterion, \
    EmissionsCriterion, CriterionConfiguration, CriteriaProvider
from pyraptor.query import query_raptor
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import RaptorTimetable, Stop
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME, SHARED_MOB_TIMETABLE_FILENAME
from pyraptor.util import mkdir_if_not_exists

OUT_FILENAME = "runner_out.csv"  # output file with performance info


@dataclass
class Query:
    """
    Class that represents a query, with info about origin stop, destination stop
    and departure time.
    """

    origin: str
    destination: str
    dep_time: str

    def __str__(self) -> str:
        return f"[FROM {self.origin} TO {self.destination} AT {self.dep_time}]"


def main(input_: str, output_dir: str, config: str):
    logger.debug("Timetable directory         : {}", input_)
    logger.debug("Output directory            : {}", output_dir)
    logger.debug("Configuration file          : {}", config)

    mkdir_if_not_exists(output_dir)

    logger.debug("Loading runner configuration...")
    runner_config: Dict = _json_to_dict(config)

    # Reading both timetables
    logger.debug("Reading timetables from the provided input directory...")
    timetable: RaptorTimetable = read_timetable(input_, TIMETABLE_FILENAME)
    timetable_sm: RaptorTimetableSM = read_timetable(input_, SHARED_MOB_TIMETABLE_FILENAME)

    # Rounds
    max_rounds: int = runner_config["max_rounds"]
    queries_settings: Mapping = runner_config["queries_settings"]
    raptor_configs: Sequence[Mapping] = runner_config["raptor_configs"]

    # Get the queries to execute
    # timetable_sm is passed since it's a superset of the base timetable
    queries = get_queries(queries_settings=queries_settings, timetable=timetable_sm)

    # Records with the following fields:
    # query,query_time,generalized_cost,config_name,dataset,fwd_deps
    runner_results: List[Mapping] = []

    logger.debug("Executing generated queries...")
    all_stop_names = [s.name for s in timetable.stops]
    all_stop_names_sm = [s.name for s in timetable_sm.stops]
    for q in queries:
        for raptor_config_obj in raptor_configs:
            sm_enabled = raptor_config_obj["enable_sm"]
            timetable = timetable_sm if sm_enabled else timetable
            timetable_stop_names = all_stop_names_sm if sm_enabled else all_stop_names

            # This might happen if query contains sm stops but sm is not enabled,
            # causing the "base" timetable to be selected
            if q.origin not in timetable_stop_names or q.destination not in timetable_stop_names:
                logger.warning("Skipping query because it contains stops not in the timetable\n"
                               f"Stops: [{q.origin}, {q.destination}]")
                continue

            query_time, journey_cost = run_raptor_config(
                raptor_config=raptor_config_obj,
                timetable=timetable,
                query=q,
                max_rounds=max_rounds
            )

            result = {
                "query": str(q),
                "query_time": query_time,
                "generalized_cost": journey_cost,
                "fwd_deps_enabled": raptor_config_obj["fwd_deps_heuristic"],
                "config_name": raptor_config_obj["configuration_id"],
                "dataset": "ATM+Trenord"  # TODO change if multiple GTFS are used
            }
            runner_results.append(result)

    output_file = os.path.join(output_dir, OUT_FILENAME)
    logger.debug(f"Runner execution terminated. Saving output to {output_file}")

    results_df = pd.DataFrame.from_records(data=runner_results)
    results_df.to_csv(output_file)


def _json_to_dict(file: str) -> Dict:
    """
    Convert a json to a dictionary
    :param file: path to json file
    :return: data as a dictionary
    """

    return json.load(open(file))


def get_queries(queries_settings: Mapping, timetable: RaptorTimetable) -> Sequence[Query]:
    """
    Returns a sequence of queries based on the provided settings.

    :param queries_settings: value of the "queries_settings" field of the runner configuration
    :param timetable: timetable to retrieve random stops from, if random queries are enabled
    :return: sequence of queries
    """

    queries: List[Query] = []

    # If True, generate random queries
    # if False, consider the queries in the field "queries"
    random_queries = queries_settings["random"]
    if random_queries:
        logger.debug(f"Generating random queries...")

        # Range of valid distances [Km]
        min_distance = queries_settings["min_distance"]
        max_distance = queries_settings["max_distance"]

        # Range of possible query hours
        min_hour = queries_settings["min_hour"]
        max_hour = queries_settings["max_hour"]

        # Total number of random queries to generate
        total_number = queries_settings["number"]

        all_stops = timetable.stops
        for i in range(total_number):
            logger.debug(f"Generating random query #{i} of {total_number}")

            # Get two random stops whose distance from each other is in the valid range
            origin_stop: Stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]
            dest_stop: Stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]

            while (origin_stop == dest_stop
                   or not (min_distance <= Stop.stop_distance(origin_stop, dest_stop) <= max_distance)):
                dest_stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]

            # Generate random query time in the provided range
            query_hour = rnd.randint(min_hour, max_hour)
            query_time = f"{query_hour}:00:00"

            query = Query(origin=origin_stop.name, destination=dest_stop.name, dep_time=query_time)
            queries.append(query)
            logger.debug(f"Query generated: {query}")
    else:
        logger.debug(f"Generating specified queries...")
        for q_obj in queries_settings["queries"]:
            query = Query(origin=q_obj["from"], destination=q_obj["to"], dep_time=q_obj["at"])
            queries.append(query)

            logger.debug(f"Query generated: {query}")

    return queries


def run_raptor_config(
        raptor_config: Mapping,
        timetable: RaptorTimetable,
        query: Query,
        max_rounds: int
) -> Tuple[float, float]:
    """
    R
    :param raptor_config: RAPTOR run configuration, i.e. a "raptor_configs" object
        in the runner_config.json file
    :param timetable: RAPTOR timetable
    :param query: object with departure and arrival stop and arrival time
    :param max_rounds: number of rounds
    :return: query execution time and total generalized cost of the resulting journey
    """

    variant = raptor_config["variant"]
    enable_sm = raptor_config["enable_sm"]

    criteria_provider = None
    if variant == 'gc':
        criteria_config_raw: Mapping[str, Mapping[str, float]] = raptor_config["criteria_config"]

        # These are the names defined in the runner_config.json file
        criteria_classes: Mapping[str, Type[Criterion]] = {
            "arrival_time": ArrivalTimeCriterion,
            "transfers": TransfersCriterion,
            "distance": DistanceCriterion,
            "co2": EmissionsCriterion
        }

        criteria_config: Dict[Type[Criterion], CriterionConfiguration] = {}
        for c_name, c_params in criteria_config_raw.items():
            criteria_config[criteria_classes[c_name]] = CriterionConfiguration(
                weight=c_params["weight"],
                upper_bound=c_params["upper_bound"]
            )

        criteria_provider = CriteriaProvider(criteria_config=criteria_config)

    query_time, algo_output = query_raptor(
        variant=variant,
        timetable=timetable,
        origin_station=query.origin,
        destination_station=query.destination,
        departure_time=query.dep_time,
        rounds=max_rounds,
        enable_fwd_deps=raptor_config["fwd_deps_heuristic"],
        criteria_provider=criteria_provider,
        enable_sm=enable_sm,
        preferred_vehicle=raptor_config["sm"]["preferred_vehicle"] if enable_sm else None,
        enable_car=raptor_config["sm"]["enable_car"] if enable_sm else None
    )

    if (algo_output.journeys is None
            or len(algo_output.journeys) == 0):
        journey_cost = -1
    else:
        # TODO as of now, algorithms output just one journey.
        #  How to handle in case multiple ones are produced? Average?
        journey_cost = algo_output.journeys[0].total_cost()

    return query_time, journey_cost


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
        default="data/output/performance",
        help="Output directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="performance/runner_config.json",
        help="Runner configuration file",
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
