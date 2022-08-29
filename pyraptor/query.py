"""
Script that allows to execute queries with all the RAPTOR algorithm variants.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Iterable
from enum import Enum
from typing import Dict, Callable

from loguru import logger

from pyraptor.model.algos.base import SharedMobilityConfig
from pyraptor.model.algos.raptor import RaptorAlgorithm
from pyraptor.model.algos.weighted_mcraptor import WeightedMcRaptorAlgorithm
from pyraptor.model.criteria import Bag, MultiCriteriaLabel
from pyraptor.model.output import AlgorithmOutput, get_journeys_to_destinations
from pyraptor.model.timetable import RaptorTimetable, Stop, TransportType
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME, SHARED_MOB_TIMETABLE_FILENAME
from pyraptor.util import str2sec


class RaptorVariants(Enum):
    """
    Enumeration that represents all the RAPTOR algorithm
    variants for which querying is supported
    """

    Basic = "basic"
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
        type=str,
        default=RaptorVariants.Basic.value,
        help="""
        Variant of the RAPTOR algorithm to execute. Possible values:\n
            - `basic`: base RAPTOR
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
        "--enable_sm",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable use of shared mobility data (default False)",
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable car-sharing transfers"
             "Ignored if argument -sm is set to False"
    )

    arguments = parser.parse_args()
    return arguments


def query_raptor(
        timetable: RaptorTimetable,
        output_folder: str | bytes | os.PathLike,
        origin_station: str,
        destination_station: str,
        departure_time: str,
        rounds: int,
        variant: str,
        criteria_config: str | bytes | os.PathLike = None,
        enable_sm: bool = False,
        preferred_vehicle: str = None,
        enable_car: bool = None
):
    """
    Queries the RAPTOR algorithm with the provided parameters and saves its output in the
    specified output folder.

    :param timetable: timetable
    :param output_folder: folder where the algorithm output is saved
    :param variant: variant of the algorithm to run
    :param origin_station: name of the station to depart from
    :param destination_station: name of the station to arrive at
    :param departure_time: departure time in the format %H:%M:%S
    :param rounds: number of iterations to perform
    :param criteria_config: path to the criteria configuration file.
        Ignored if variant is not multi-criteria.
    :param enable_sm: if True, shared mobility data is included in the itinerary computation.
        If False, provided shared mob data is ignored
    :param preferred_vehicle: type of preferred vehicle
    :param enable_car: car-sharing transfer enabled
    """

    logger.debug("Output directory         : {}", output_folder)
    logger.debug("Origin station           : {}", origin_station)
    logger.debug("Destination station      : {}", destination_station)
    logger.debug("Departure time           : {}", departure_time)
    logger.debug("Rounds                   : {}", str(rounds))
    logger.debug("Algorithm Variant        : {}", variant)
    logger.debug("Criteria Config          : {}", criteria_config)
    logger.debug("Enable use of shared-mob : {}", enable_sm)
    logger.debug("Preferred vehicle        : {}", preferred_vehicle)
    logger.debug("Enable car               : {}", enable_car)

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
        if st.name == destination_station  # only true destination stops
    }
    destination_stops.pop(origin_station, None)

    preferred_transport_type = _process_shared_mob_args(
        enable_sm=enable_sm,
        preferred_vehicle=preferred_vehicle,
        enable_car=enable_car
    )

    best_labels = _handle_raptor_variant(
        variant=RaptorVariants(variant),
        timetable=timetable,
        origin_stops=origin_stops,
        dep_secs=dep_secs,
        criteria_file_path=criteria_config,
        enable_sm=enable_sm,
        preferred_vehicle=preferred_transport_type,
        enable_car=enable_car,
        rounds=rounds
    )

    journeys_to_all_destinations = get_journeys_to_destinations(
        origin_stops=origin_stops,
        destination_stops=destination_stops,
        best_labels=best_labels
    )

    # Print all the journeys to the specified destination
    destination_journeys = journeys_to_all_destinations[destination_station]

    if len(destination_journeys) == 0:
        logger.warning(f"No journeys found for destination `{destination_station}`")
    else:
        for j in destination_journeys:
            j.print()

    algo_output = AlgorithmOutput(
        journeys=destination_journeys,
        date=timetable.date,
        departure_time=departure_time,
        original_gtfs_dir=timetable.original_gtfs_dir
    )
    AlgorithmOutput.save(
        output_dir=output_folder,
        algo_output=algo_output
    )


def _process_shared_mob_args(
        enable_sm: bool,
        preferred_vehicle: str,
        enable_car: bool
) -> TransportType | None:
    if enable_sm:

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

        return preferred_vehicle_type
    else:
        return None


def _handle_raptor_variant(
        variant: RaptorVariants,
        timetable: RaptorTimetable,
        origin_stops: Iterable[Stop],
        dep_secs: int,
        rounds: int,
        criteria_file_path: str,
        enable_sm: bool,
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
    :param criteria_file_path: path to the criteria configuration file.
        Ignored if variant is not multi-criteria
    :param enable_sm: if True, shared mobility data is included in the itinerary computation.
        If False, provided shared mob data is ignored
    :param preferred_vehicle: type of preferred vehicle
    :param enable_car: car-sharing transfer enabled
    :return: mapping that pairs each stop with its bag of best labels
    """

    sm_config = SharedMobilityConfig(
        preferred_vehicle=preferred_vehicle,
        enable_car=enable_car
    )

    def run_weighted_mc_raptor() -> Mapping[Stop, Bag]:
        raptor = WeightedMcRaptorAlgorithm(
            timetable=timetable,
            enable_sm=enable_sm,
            sm_config=sm_config,
            criteria_file_path=criteria_file_path
        )
        bag_round_stop = raptor.run(origin_stops, dep_secs, rounds)

        return bag_round_stop[rounds]

    def run_base_raptor() -> Mapping[Stop, Bag]:
        raptor = RaptorAlgorithm(
            timetable=timetable,
            enable_sm=enable_sm,
            sm_config=sm_config
        )
        bag_round_stop = raptor.run(origin_stops, dep_secs, rounds)
        best_labels = bag_round_stop[rounds]

        # Convert best labels from Dict[Stop, Label] to Dict[Stop, Bag]
        best_bags: Dict[Stop, Bag] = {}
        for stop, label in best_labels.items():
            mc_label = MultiCriteriaLabel.from_base_raptor_label(label)
            best_bags[stop] = Bag(labels=[mc_label])

        return best_bags

    variant_switch: Dict[RaptorVariants, Callable[[], Mapping[Stop, Bag]]] = {
        RaptorVariants.Basic: run_base_raptor,
        RaptorVariants.WeightedMc: run_weighted_mc_raptor
    }

    return variant_switch[variant]()


def _load_timetable(input_folder: str, enable_sm: bool) -> RaptorTimetable:
    logger.debug("Loading timetable...")
    logger.debug("Input directory         : {}", input_folder)
    logger.debug("Enable Shared Mob       : {}", enable_sm)

    if enable_sm:
        timetable_name = SHARED_MOB_TIMETABLE_FILENAME
    else:
        timetable_name = TIMETABLE_FILENAME

    return read_timetable(input_folder=input_folder, timetable_name=timetable_name)


if __name__ == "__main__":
    args = _parse_arguments()

    timetable = _load_timetable(args.input, args.enable_sm)

    query_raptor(
        variant=args.variant,
        timetable=timetable,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds,
        criteria_config=args.mc_config,
        enable_sm=args.enable_sm,
        preferred_vehicle=args.preferred,
        enable_car=args.car
    )
