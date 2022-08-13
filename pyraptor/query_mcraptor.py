"""Run query with RAPTOR algorithm"""
import argparse
from typing import List, Dict, Any
from copy import copy
from time import perf_counter

from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.timetable import Timetable, Station, Stop
from pyraptor.model.output import Journey
from pyraptor.util import str2sec


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Input directory",
    )
    parser.add_argument(
        "-or",
        "--origin",
        type=str,
        default="Hertogenbosch ('s)",
        help="Origin station of the journey",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="Rotterdam Centraal",
        help="Destination station of the journey",
    )
    parser.add_argument(
        "-t", "--time", type=str, default="08:35:00", help="Departure time (hh:mm:ss)"
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds to execute the RAPTOR algorithm",
    )
    parser.add_argument(
        "-wmc",
        "--weighted_mc",
        type=bool,
        default=False,
        help="If True, the Weighted McRaptor algorithm is executed",
    )
    parser.add_argument(
        "-cfg",
        "--mc_config",
        type=str,
        default="data/input/mc.json",
        help="Path to the criteria configuration file. "
             "This argument is ignored if argument -wmc is set to False.",
    )
    arguments = parser.parse_args()
    return arguments


def main(
        input_folder,
        origin_station,
        destination_station,
        departure_time,
        rounds,
        is_weighted_mc,
        criteria_config
):
    """Run RAPTOR algorithm"""

    logger.debug("Input directory     : {}", input_folder)
    logger.debug("Origin station      : {}", origin_station)
    logger.debug("Destination station : {}", destination_station)
    logger.debug("Departure time      : {}", departure_time)
    logger.debug("Rounds              : {}", str(rounds))
    logger.debug("Is WeightedMcRaptor : {}", is_weighted_mc)
    logger.debug("Criteria Config     : {}", criteria_config)

    timetable = read_timetable(input_folder)

    logger.info(f"Calculating network from : {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations
    journeys_to_destinations = run_mcraptor(
        timetable,
        origin_station,
        dep_secs,
        rounds,
        is_weighted_mc,
        criteria_config
    )

    # Output journey
    journeys = journeys_to_destinations[destination_station]
    if len(journeys) != 0:
        for jrny in journeys:
            jrny.print(dep_secs=dep_secs)
    else:
        logger.debug(f"No journeys found to {destination_station}")


def run_mcraptor(
        timetable: Timetable,
        origin_station: str,
        dep_secs: int,
        rounds: int,
        is_weighted_mc: bool,
        criteria_file_path: str
) -> Dict[Station, List[Journey]]:
    """
    Perform the McRaptor algorithm.

    :param timetable: timetable
    :param origin_station: name of origin station
    :param dep_secs: time of departure in seconds
    :param rounds: number of iterations to perform
    :param is_weighted_mc: if True, the weighted version of McRaptor is executed
    :param criteria_file_path: path to the criteria configuration file
    """

    # Run Round-Based Algorithm for an origin station
    from_stops = timetable.stations.get(origin_station).stops

    # Select McRaptor variant
    if is_weighted_mc:
        from pyraptor.model.algos.weighted_mcraptor import (
            WeightedMcRaptorAlgorithm,
            reconstruct_journeys,
            best_legs_to_destination_station
        )
        raptor = WeightedMcRaptorAlgorithm(timetable, criteria_file_path)
    else:
        from pyraptor.model.algos.mcraptor import (
            McRaptorAlgorithm,
            reconstruct_journeys,
            best_legs_to_destination_station,
        )
        raptor = McRaptorAlgorithm(timetable)

    bag_round_stop, actual_rounds = raptor.run(from_stops, dep_secs, rounds)
    last_round_bag = copy(bag_round_stop[rounds])

    # Calculate journeys to all destinations
    logger.info("Calculating journeys to all destinations")
    s = perf_counter()

    destination_stops: Dict[Any, List[Stop]] = {
        st.name: timetable.stations.get_stops(st.name) for st in timetable.stations
    }
    destination_stops.pop(origin_station, None)

    # TODO here journeys are constructed just with the first and last stop
    #   of a leg (i.e. just beginning and end stop of each different trip)
    #   maybe include intermediate stop to help with debug and visualization
    journeys_to_destinations = dict()
    for destination_station_name, to_stops in destination_stops.items():
        destination_legs = best_legs_to_destination_station(to_stops, last_round_bag)

        if len(destination_legs) == 0:
            logger.debug(f"Destination '{destination_station_name}' unreachable with given parameters."
                         f"Station stops: {to_stops}")
            continue

        journeys = reconstruct_journeys(
            from_stops, destination_legs, bag_round_stop, k=rounds
        )
        journeys_to_destinations[destination_station_name] = journeys

    logger.info(f"Journey calculation time: {perf_counter() - s}")

    return journeys_to_destinations


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds,
        is_weighted_mc=args.weighted_mc,
        criteria_config=args.mc_config
    )
