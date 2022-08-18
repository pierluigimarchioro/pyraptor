"""
Script that allows to execute queries with all the RAPTOR algorithm variants.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from copy import copy
from time import perf_counter
from typing import Dict, Any, List

from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.algos.raptor_sm import RaptorAlgorithmSharedMobility
from pyraptor.model.shared_mobility import SharedMobilityFeed
from pyraptor.model.timetable import RaptorTimetable, Stop, TransportType
from pyraptor.model.output import Journey, AlgorithmOutput
from pyraptor.model.algos.raptor import (
    RaptorAlgorithm,
    reconstruct_journey,
    best_stop_at_target_station,
)
from pyraptor.util import str2sec

# TODO
#   - make McRAPTOR return journeys with intermediate legs too
#   - one single script file for all algorithms
#   - make demo call script method and not the whole script, so we have control
#       over the generation of the timetable.
#       Also make timetable a parameter of the run_algorithm method for each variant
#   - make algorithms inherit from base algo
#   - make algo output accept list of journeys

# TODO refactoring notes:
"""
- keep and move here just one copy of the functions reconstruct_journey(s), etc.
    The only necessary copy might be the one that treats bags of labels, in which case the bags
    used in base RAPTOR need to be adapted (i.e. convert from Dict[str, Label] to Dict[Stop, Bag]
- now each algorithm should output a list of journeys for each stop
"""


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
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Output directory",
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
        "-t",
        "--time",
        type=str,
        default="08:35:00",
        help="Departure time (hh:mm:ss)"
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds to execute the RAPTOR algorithm",
    )
    parser.add_argument(
        "-var",
        "--variant",
        type=bool,
        default=False,
        help="""
        Variant of the RAPTOR algorithm to execute. Possible values:\n
            - `base`: base RAPTOR
            - `mc`: More Criteria RAPTOR
            - `wmc`: Weighted More Criteria RAPTOR
        """,
    )
    parser.add_argument(
        "-cfg",
        "--mc_config",
        type=str,
        default="data/input/mc.json",
        help="Path to the criteria configuration file. "
             "This argument is ignored if the algorithm variant is not Weighted More Criteria",
    )
    parser.add_argument(
        "-sm",
        "--shared_mob",
        type=bool,
        default=True,
        help="Enable use of shared mobility data (default True)",
    )
    parser.add_argument(
        "-f",
        "--feeds",
        type=str,
        default="data/input/gbfs.json",
        help="Path to .json key specifying list of feeds and langs"
             "Ignored if argument -sm is set to False"
    )
    parser.add_argument(
        "-p",
        "--preferred",
        type=str,
        default="regular",
        help="Preferred type of vehicle (regular | electric | car)"
             "Ignored if argument -sm is set to False"
    )
    parser.add_argument(
        "-c",
        "--car",
        type=bool,
        default=False,
        help="Enable car-sharing transfers"
             "Ignored if argument -sm is set to False"
    )

    arguments = parser.parse_args()
    return arguments


def main(
    input_folder: str | bytes | os.PathLike,
    output_folder: str | bytes | os.PathLike,
    origin_station: str,
    destination_station: str,
    departure_time: str,
    rounds: int,
    variant: str,
    criteria_config: str | bytes | os.PathLike,
    feeds: str | bytes | os.PathLike,
    preferred: str,
    car: bool
):
    """Run RAPTOR algorithm"""

    logger.debug("Input directory       : {}", input_folder)
    logger.debug("Output directory      : {}", output_folder)
    logger.debug("Origin station        : {}", origin_station)
    logger.debug("Destination station   : {}", destination_station)
    logger.debug("Departure time        : {}", departure_time)
    logger.debug("Rounds                : {}", str(rounds))
    logger.debug("Algorithm Variant     : {}", variant)
    logger.debug("Criteria Config       : {}", criteria_config)
    logger.debug("Input shared-mob      : {}", feeds)
    logger.debug("Preferred vehicle     : {}", preferred)
    logger.debug("Enable car            : {}", car)

    # Input check TODO move check in other position ?
    if origin_station == destination_station:
        raise ValueError(f"{origin_station} is both origin and destination")

    timetable = read_timetable(input_folder)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations

    journeys_to_destinations = run_raptor(
        timetable,
        origin_station,
        dep_secs,
        rounds,
    )

    # Print journey to destination
    destination_journey = journeys_to_destinations[destination_station]
    destination_journey.print()

    algo_output = AlgorithmOutput(
        journey=destination_journey,
        date=timetable.date,
        departure_time=departure_time,
        original_gtfs_dir=timetable.original_gtfs_dir
    )
    AlgorithmOutput.save(output_dir=output_folder,
                         algo_output=algo_output)


def run_raptor(
    timetable: RaptorTimetable,
    origin_station: str,
    destination_station: str,
    dep_secs: int,
    rounds: int,
) -> Dict[str, Sequence[Journey]]:
    """
    Executes the base RAPTOR algorithm.

    :param timetable: timetable
    :param origin_station: name of origin station
    :param destination_station: name of the destination station
    :param dep_secs: time of departure in seconds
    :param rounds: number of iterations to perform
    """

    # Get stops for origin and all destinations
    try:
        from_stops = timetable.stations.get(origin_station).stops
        destination_stops = {
            st.name: timetable.stations.get_stops(st.name) for st in timetable.stations
        }
        destination_stops.pop(origin_station, None)

        # Run Round-Based Algorithm
        raptor = RaptorAlgorithm(timetable)
        bag_round_stop = raptor.run(from_stops, dep_secs, rounds)
        best_labels = bag_round_stop[rounds]

        # Determine the best journey to all possible destination stations
        journey_to_destinations = dict()
        for destination_station_name, to_stops in destination_stops.items():
            dest_stop = best_stop_at_target_station(to_stops, best_labels)

            if dest_stop is not None:
                journey = reconstruct_journey(dest_stop, best_labels)
                journey_to_destinations[destination_station_name] = journey

        return journey_to_destinations
    except Exception as ex:
        logger.error(ex)


def run_mcraptor(
        timetable: RaptorTimetable,
        origin_station: str,
        dep_secs: int,
        rounds: int,
        is_weighted_mc: bool,
        criteria_file_path: str
) -> Dict[str, Sequence[Journey]]:
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


def run_raptor_sm(
    timetable: RaptorTimetable,
    feeds: List[SharedMobilityFeed],
    origin_station: str,
    dep_secs: int,
    rounds: int,
    preferred_vehicle: TransportType,
    use_car: bool
) -> Dict[str, Sequence[Journey]]:
    """
    Run the Shared Mobility Raptor algorithm.

    :param timetable: timetable
    :param feeds: share mobility feeds to include in the timetable
    :param origin_station: Name of origin station
    :param dep_secs: Time of departure in seconds
    :param rounds: Number of iterations to perform
    :param preferred_vehicle: type of preferred vehicle
    :param use_car: car-sharing transfer enabled
    """

    # Get stops for origin and all destinations
    from_stops = timetable.stations.get(origin_station).stops
    destination_stops = {
        st.name: timetable.stations.get_stops(st.name) for st in timetable.stations
    }
    destination_stops.pop(origin_station, None)

    # Run Round-Based Algorithm
    raptor = RaptorAlgorithmSharedMobility(timetable, feeds, preferred_vehicle, use_car)
    bag_round_stop = raptor.run(from_stops, dep_secs, rounds)
    best_labels = bag_round_stop[rounds]

    # Determine the best journey to all possible destination stations
    journey_to_destinations = dict()
    for destination_station_name, to_stops in destination_stops.items():
        dest_stop = best_stop_at_target_station(to_stops, best_labels)
        if dest_stop != 0:
            journey = reconstruct_journey(dest_stop, best_labels)
            journey_to_destinations[destination_station_name] = [journey]

    return journey_to_destinations


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds
    )
