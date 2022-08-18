"""
Script that allows to execute queries with all the RAPTOR algorithm variants.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence, Mapping, Iterable
from enum import Enum
from time import perf_counter
from typing import Dict, Any, List, Tuple, Callable

from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.algos.raptor_sm import RaptorAlgorithmSharedMobility
from pyraptor.model.algos.weighted_mcraptor import WeightedMcRaptorAlgorithm
from pyraptor.model.criteria import Bag, MultiCriteriaLabel, pareto_set, BaseRaptorLabel
from pyraptor.model.shared_mobility import SharedMobilityFeed
from pyraptor.model.timetable import RaptorTimetable, Stop, TransportType
from pyraptor.model.output import Journey, AlgorithmOutput, Leg
from pyraptor.util import str2sec

# TODO
#   - make McRAPTOR return journeys with intermediate legs too
#   - one single script file for all algorithms
#   - make demo call script method and not the whole script, so we have control
#       over the generation of the timetable.
#       Also make timetable a parameter of the run_algorithm method for each variant
#   - use criteria.Label class in RaptorSMAlgorithm
#   - make algorithms inherit from base algo class and from base SM algo class
#   - implement shared mob in WMC and rename RaptorSM to just RAPTOR
#   - make algo output accept list of journeys
#   - delete old RAPTOR variants (i.e. base without SM and Mc)


class RaptorVariants(Enum):
    """
    Enumeration that represents all the RAPTOR algorithm
    variants for which querying is supported
    """

    Base = "base"
    WeightedMc = "wmc"


def _parse_arguments():
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
        timetable: RaptorTimetable,
        output_folder: str | bytes | os.PathLike,
        origin_station: str,
        destination_station: str,
        departure_time: str,
        rounds: int,
        variant: str,
        criteria_config: str | bytes | os.PathLike,
        sm_feeds_path: str | bytes | os.PathLike,
        preferred_vehicle: str,
        enable_car: bool
):
    """
    :param timetable: timetable
    :param output_folder: folder where the algorithm output is saved
    :param variant: variant of the algorithm to run
    :param origin_station: name of the station to depart from
    :param destination_station: name of the station to arrive at
    :param departure_time: departure time in the format %H:%M:%S
    :param rounds: number of iterations to perform
    :param criteria_config: path to the criteria configuration file
    :param sm_feeds_path: path to the shared mob configuration file
    :param preferred_vehicle: type of preferred vehicle
    :param enable_car: car-sharing transfer enabled
    """

    logger.debug("Output directory      : {}", output_folder)
    logger.debug("Origin station        : {}", origin_station)
    logger.debug("Destination station   : {}", destination_station)
    logger.debug("Departure time        : {}", departure_time)
    logger.debug("Rounds                : {}", str(rounds))
    logger.debug("Algorithm Variant     : {}", variant)
    logger.debug("Criteria Config       : {}", criteria_config)
    logger.debug("Input shared-mob      : {}", sm_feeds_path)
    logger.debug("Preferred vehicle     : {}", preferred_vehicle)
    logger.debug("Enable car            : {}", enable_car)

    # Input check
    if origin_station == destination_station:
        raise ValueError(f"{origin_station} is both origin and destination")

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    origin_stops = timetable.stations.get(origin_station).stops
    destination_stops = {
        st.name: timetable.stations.get_stops(st.name) for st in timetable.stations
    }
    destination_stops.pop(origin_station, None)

    preferred_transport_type, sm_feeds = _process_shared_mob_args(
        preferred_vehicle=preferred_vehicle,
        enable_car=enable_car,
        feeds_path=sm_feeds_path
    )

    best_labels = _handle_raptor_variant(
        variant=RaptorVariants(variant),
        timetable=timetable,
        origin_stops=origin_stops,
        dep_secs=dep_secs,
        criteria_file_path=criteria_config,
        feeds=sm_feeds,
        preferred_vehicle=preferred_transport_type,
        enable_car=enable_car,
        rounds=rounds
    )

    journeys_to_all_destinations = _get_journeys_to_destinations(
        origin_stops=origin_stops,
        destination_stops=destination_stops,
        best_labels=best_labels
    )

    # Print all the journeys to the specified destination
    destination_journeys = journeys_to_all_destinations[destination_station]
    for j in destination_journeys:
        j.print()

    algo_output = AlgorithmOutput(
        # TODO change AlgorithmOutput class
        journey=destination_journeys[0],
        date=timetable.date,
        departure_time=departure_time,
        original_gtfs_dir=timetable.original_gtfs_dir
    )
    AlgorithmOutput.save(
        output_dir=output_folder,
        algo_output=algo_output
    )


def _process_shared_mob_args(
        preferred_vehicle: str,
        enable_car: bool,
        feeds_path: str
) -> Tuple[TransportType, Iterable[SharedMobilityFeed]]:
    # Reading shared mobility feed
    feed_infos: List[Dict] = json.load(open(feeds_path))['feeds']
    feeds: List[SharedMobilityFeed] = ([SharedMobilityFeed(feed_info['url'], feed_info['lang'])
                                        for feed_info in feed_infos])
    logger.debug(f"{', '.join([feed.system_id for feed in feeds])} feeds retrieved successfully")

    # Preferred bike type
    if preferred_vehicle == 'car':
        preferred_vehicle_type = TransportType.Car
    elif preferred_vehicle == 'regular':
        preferred_vehicle_type = TransportType.Bike
    elif preferred_vehicle == 'electric':
        preferred_vehicle_type = TransportType.ElectricBike
    else:
        raise ValueError(f"Unhandled vehicle for value `{preferred_vehicle}`")

    # Car check
    if preferred_vehicle == TransportType.Car and not enable_car:
        raise Exception("Preferred vehicle is car, but car-sharing transfers are disabled")

    return preferred_vehicle_type, feeds


def _handle_raptor_variant(
        variant: RaptorVariants,
        timetable: RaptorTimetable,
        origin_stops: Iterable[Stop],
        dep_secs: int,
        rounds: int,
        criteria_file_path: str,
        feeds: Iterable[SharedMobilityFeed],
        preferred_vehicle: TransportType,
        enable_car: bool
) -> Mapping[Stop, Bag]:
    """
    Executes the specified variant of the Raptor algorithm and returns a
    mapping that pairs each stop with its bag of best labels.

    :param variant: variant of the algorithm to run
    :param timetable: timetable
    :param origin_stops: collection of stops to depart from
    :param dep_secs: time of departure in seconds
    :param rounds: number of iterations to perform
    :param criteria_file_path: path to the criteria configuration file
    :param feeds: share mobility feeds to include in the timetable
    :param preferred_vehicle: type of preferred vehicle
    :param enable_car: car-sharing transfer enabled
    :return: mapping that pairs each stop with its bag of best labels
    """

    def run_weighted_mc_raptor() -> Mapping[Stop, Bag]:
        # TODO implement shared mob in WeightedMcRaptor too
        raptor = WeightedMcRaptorAlgorithm(
            timetable=timetable,
            criteria_file_path=criteria_file_path
        )
        bag_round_stop, actual_rounds = raptor.run(origin_stops, dep_secs, rounds)

        return bag_round_stop[actual_rounds]

    def run_base_raptor() -> Mapping[Stop, Bag]:
        raptor = RaptorAlgorithmSharedMobility(
            timetable,
            shared_mobility_feeds=feeds,
            preferred_vehicle=preferred_vehicle,
            use_car=enable_car
        )
        bag_round_stop = raptor.run(origin_stops, dep_secs, rounds)
        best_labels = bag_round_stop[rounds]

        # Convert best labels from Dict[Stop, Label] to Dict[Stop, Bag]
        best_bags: Dict[Stop, Bag] = {}
        for stop, label in best_labels.items():
            # TODO remove once RaptorSM uses BaseRaptorLabel instead of old Label classes
            base_label = BaseRaptorLabel(
                boarding_stop=label.boarding_stop,
                trip=label.trip,
                earliest_arrival_time=label.earliest_arrival_time
            )
            mc_label = MultiCriteriaLabel.from_base_raptor_label(base_label)
            best_bags[stop] = Bag(labels=[mc_label])

        return best_bags

    variant_switch: Dict[RaptorVariants, Callable[[], Mapping[Stop, Bag]]] = {
        RaptorVariants.Base: run_base_raptor,
        RaptorVariants.WeightedMc: run_weighted_mc_raptor
    }

    return variant_switch[variant]()


def _get_journeys_to_destinations(
        origin_stops: Iterable[Stop],
        destination_stops: Dict[Any, Iterable[Stop]],
        best_labels: Mapping[Stop, Bag]
) -> Mapping[Any, Sequence[Journey]]:
    # Calculate journeys to all destinations
    logger.info("Calculating journeys to all destinations")
    s = perf_counter()

    # TODO here journeys are constructed just with the first and last stop
    #   of a leg (i.e. just beginning and end stop of each different trip)
    #   maybe include intermediate stop to help with debug and visualization
    journeys_to_destinations = {}
    for destination_station_name, to_stops in destination_stops.items():
        destination_legs = _best_legs_to_destination_station(origin_stops, best_labels)

        if len(destination_legs) == 0:
            logger.debug(f"Destination '{destination_station_name}' unreachable with given parameters."
                         f"Station stops: {to_stops}")
            continue

        journeys = _reconstruct_journeys(
            origin_stops, destination_legs, best_labels
        )
        journeys_to_destinations[destination_station_name] = journeys

    logger.info(f"Journey calculation time: {perf_counter() - s}")

    return journeys_to_destinations


def _best_legs_to_destination_station(
        to_stops: Iterable[Stop], last_round_bag: Mapping[Stop, Bag]
) -> Sequence[Leg]:
    """
    Find the last legs to destination station that are reached by non-dominated labels.
    """

    # Find all labels to target_stops
    best_labels = [
        (stop, label) for stop in to_stops for label in last_round_bag[stop].labels
    ]

    # TODO Use merge function on Bag
    # Pareto optimal labels
    pareto_optimal_labels = pareto_set([label for (_, label) in best_labels])
    pareto_optimal_labels: List[Tuple[Stop, MultiCriteriaLabel]] = [
        (stop, label) for (stop, label) in best_labels if label in pareto_optimal_labels
    ]

    # Label to leg, i.e. add to_stop
    legs = [
        Leg(
            from_stop=label.boarding_stop,
            to_stop=to_stop,
            trip=label.trip,
            criteria=label.criteria
        )
        for to_stop, label in pareto_optimal_labels
    ]
    return legs


def _reconstruct_journeys(
        from_stops: Iterable[Stop],
        destination_legs: Iterable[Leg],
        best_labels: Mapping[Stop, Bag]
) -> List[Journey]:
    """
    Construct Journeys for destinations from bags by recursively
    looping from destination to origin.
    """

    def loop(best_labels: Mapping[Stop, Bag], journeys: Iterable[Journey]):
        """Create full journey by prepending legs recursively"""

        for jrny in journeys:
            current_leg = jrny[0]

            # End of journey if we are at origin stop or journey is not feasible
            if current_leg.trip is None or current_leg.from_stop in from_stops:
                jrny = jrny.remove_empty_legs()

                # Journey is valid if leg k ends before the start of leg k+1
                if jrny.is_valid() is True:
                    yield jrny
                continue

            # Loop trough each new leg. These are the legs that come before the current and that lead to from_stop
            labels_to_from_stop = best_labels[current_leg.from_stop].labels
            for new_label in labels_to_from_stop:
                new_leg = Leg(
                    from_stop=new_label.boarding_stop,
                    to_stop=current_leg.from_stop,
                    trip=new_label.trip,
                    criteria=new_label.criteria
                )
                # Only prepend new_leg if compatible before current leg, e.g. earlier arrival time, etc.
                if new_leg.is_compatible_before(current_leg):
                    new_jrny = jrny.prepend_leg(new_leg)
                    for i in loop(best_labels, [new_jrny]):
                        yield i

    journeys = [Journey(legs=[leg]) for leg in destination_legs]
    journeys = [jrny for jrny in loop(best_labels, journeys)]

    return journeys


def _load_timetable(input_folder: str) -> RaptorTimetable:
    logger.debug("Input directory       : {}", input_folder)

    return read_timetable(input_folder)


if __name__ == "__main__":
    args = _parse_arguments()

    timetable = _load_timetable(args.input)

    main(
        variant=args.variant,
        timetable=timetable,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds,
        criteria_config=args.mc_config,
        sm_feeds_path=args.feeds,
        preferred_vehicle=args.preferred,
        enable_car=args.car
    )
