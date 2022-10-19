"""
Script that allows to execute queries with all the RAPTOR algorithm variants.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Iterable
from enum import Enum
from typing import Dict, Callable, Tuple
from timeit import default_timer as timer

from loguru import logger

from pyraptor.model.algos.base import SharedMobilityConfig
from pyraptor.model.algos.et_raptor import EarliestArrivalTimeRaptor
from pyraptor.model.algos.gc_raptor import GeneralizedCostRaptor
from pyraptor.model.criteria import MultiCriteriaLabel, FileCriteriaProvider, CriteriaFactory, ParetoBag, Bag
from pyraptor.model.output import AlgorithmOutput, get_journeys_to_destinations, Journey, Leg
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import RaptorTimetable, Stop, TransportType
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME, SHARED_MOB_TIMETABLE_FILENAME
from pyraptor.util import str2sec, sec2minutes


class RaptorVariants(Enum):
    """
    Enumeration that represents all the RAPTOR algorithm
    variants for which querying is supported
    """

    EarliestArrivalTime = "et"
    GeneralizedCost = "gc"

    def __str__(self):
        return self.value


def query_raptor(
        timetable: RaptorTimetable,
        origin_station: str,
        destination_station: str,
        departure_time: str,
        rounds: int,
        variant: str,
        enable_fwd_deps: bool = True,
        criteria_provider: CriteriaFactory = None,
        enable_sm: bool = False,
        preferred_vehicle: str = None,
        enable_car: bool = None
) -> Tuple[float, AlgorithmOutput]:
    """
    Queries the RAPTOR algorithm with the provided parameters.
    Returns the query execution time and the algorithm output

    :param timetable: timetable
    :param variant: variant of the algorithm to run
    :param origin_station: name of the station to depart from
    :param destination_station: name of the station to arrive at
    :param departure_time: departure time in the format %H:%M:%S
    :param rounds: number of iterations to perform
    :param enable_fwd_deps: if True, the Forward Dependencies Heuristic is enabled
    :param criteria_provider: provider of parameterized criteria for multi-criteria variants
    :param enable_sm: if True, shared mobility data is included in the itinerary computation.
        If False, provided shared mob data is ignored
    :param preferred_vehicle: type of preferred vehicle
    :param enable_car: car-sharing transfer enabled

    :return: query execution time and the algorithm output
    """

    logger.debug("Origin station           : {}", origin_station)
    logger.debug("Destination station      : {}", destination_station)
    logger.debug("Departure time           : {}", departure_time)
    logger.debug("Rounds                   : {}", str(rounds))
    logger.debug("Algorithm Variant        : {}", variant)
    logger.debug("Criteria Provider        : {}", criteria_provider)
    logger.debug("Enable use of shared-mob : {}", enable_sm)
    logger.debug("Preferred vehicle        : {}", preferred_vehicle)
    logger.debug("Enable car               : {}", enable_car)

    logger.debug("Criteria Configuration:")
    for c in criteria_provider.create_criteria():
        logger.debug(repr(c))

    start_time = timer()

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

    best_bags = _execute_raptor_variant(
        variant=RaptorVariants(variant),
        timetable=timetable,
        origin_stops=origin_stops,
        dep_secs=dep_secs,
        criteria_provider=criteria_provider,
        enable_fwd_deps=enable_fwd_deps,
        enable_sm=enable_sm,
        preferred_vehicle=preferred_transport_type,
        enable_car=enable_car,
        rounds=rounds
    )

    end_time = timer()
    query_time = end_time - start_time

    journeys_to_all_destinations = get_journeys_to_destinations(
        origin_stops=origin_stops,
        destination_stops=destination_stops,
        best_bags=best_bags
    )

    if destination_station not in journeys_to_all_destinations:
        logger.warning(f"Destination `{destination_station}` is unreachable")

        destination_journeys = []
    else:
        # Print all the journeys to the specified destination
        destination_journeys = journeys_to_all_destinations[destination_station]

    if len(destination_journeys) == 0:
        logger.warning(f"No valid journeys found for destination `{destination_station}`")

        # TODO this is a "hack" to get the cost of the destination label
        #   in the query runner a Journey object is only used for the total_cost() property
        dest_stop = next(filter(lambda s: s.name == destination_station, timetable.stops), None)

        if len(best_bags[dest_stop].labels) != 0:
            dest_label = best_bags[dest_stop].labels[0]
            destination_journeys.append(
                Journey(
                    legs=[
                        Leg(
                            from_stop=dest_label.boarding_stop,
                            to_stop=dest_stop,
                            trip=dest_label.trip,
                            criteria=dest_label.criteria
                        )
                    ]
                ))
        else:
            logger.warning("Destination is TRULY unreachable")
    else:
        for j in destination_journeys:
            j.print()

    algo_output = AlgorithmOutput(
        journeys=destination_journeys,
        date=timetable.date,
        departure_time=departure_time,
        original_gtfs_dir=timetable.original_gtfs_dir
    )

    return query_time, algo_output


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
            raise ValueError("Preferred vehicle is car, but car-sharing transfers are disabled")

        return preferred_vehicle_type
    else:
        return None


def _execute_raptor_variant(
        variant: RaptorVariants,
        timetable: RaptorTimetable | RaptorTimetableSM,
        origin_stops: Iterable[Stop],
        dep_secs: int,
        rounds: int,
        enable_fwd_deps: bool,
        criteria_provider: CriteriaFactory,
        enable_sm: bool,
        preferred_vehicle: TransportType,
        enable_car: bool
) -> Mapping[Stop, Bag[MultiCriteriaLabel]]:
    """
    Executes the specified variant of the Raptor algorithm and returns a
    mapping that pairs each stop with its bag of best labels.
    """

    sm_config = SharedMobilityConfig(
        preferred_vehicle=preferred_vehicle,
        enable_car=enable_car
    )

    def run_gc_raptor() -> Mapping[Stop, Bag[MultiCriteriaLabel]]:
        raptor = GeneralizedCostRaptor(
            timetable=timetable,
            enable_fwd_deps_heuristic=enable_fwd_deps,
            enable_sm=enable_sm,
            sm_config=sm_config,
            criteria_provider=criteria_provider
        )
        return raptor.run(origin_stops, dep_secs, rounds)

    def run_base_raptor() -> Mapping[Stop, Bag[MultiCriteriaLabel]]:
        raptor = EarliestArrivalTimeRaptor(
            timetable=timetable,
            enable_fwd_deps_heuristic=enable_fwd_deps,
            enable_sm=enable_sm,
            sm_config=sm_config
        )
        results = raptor.run(origin_stops, dep_secs, rounds)

        # Convert best labels from Dict[Stop, Label] to Dict[Stop, Bag]
        best_bags: Dict[Stop, ParetoBag] = {}
        for stop, bag in results.items():
            mc_labels = []
            for label in bag.labels:
                mc_lbl = MultiCriteriaLabel.from_et_label(label)  # TODO find cleaner way to get label

                mc_labels.append(mc_lbl)

            best_bags[stop] = ParetoBag(labels=mc_labels)

        return best_bags

    variant_switch: Dict[RaptorVariants, Callable[[], Mapping[Stop, Bag[MultiCriteriaLabel]]]] = {
        RaptorVariants.EarliestArrivalTime: run_base_raptor,
        RaptorVariants.GeneralizedCost: run_gc_raptor
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
        default=RaptorVariants.EarliestArrivalTime.value,
        choices=[str(raptor_var) for raptor_var in RaptorVariants],
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


def _main():
    args = _parse_arguments()

    cached_timetable = _load_timetable(input_folder=args.input, enable_sm=args.enable_sm)
    file_criteria_provider = FileCriteriaProvider(criteria_config_path=args.mc_config)

    elapsed_time, algo_output = query_raptor(
        variant=args.variant,
        timetable=cached_timetable,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds,
        criteria_provider=file_criteria_provider,
        enable_sm=args.enable_sm,
        preferred_vehicle=args.preferred,
        enable_car=args.car
    )

    logger.debug("Output directory         : {}", args.output)
    AlgorithmOutput.save(
        output_dir=args.output,
        algo_output=algo_output
    )

    logger.info(f"Elapsed time: {elapsed_time} sec ({sec2minutes(elapsed_time)})")


if __name__ == "__main__":
    _main()
