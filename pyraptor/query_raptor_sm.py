"""Run query with RAPTOR algorithm"""
import argparse
import json
from typing import Dict, List

from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.algos.raptor_sm import (
    RaptorAlgorithmSharedMobility,
    reconstruct_journey,
    best_stop_at_target_station,
)
from pyraptor.model.timetable import Timetable, TransportType
from pyraptor.model.output import Journey, AlgorithmOutput
from pyraptor.model.shared_mobility import SharedMobilityFeed
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
        "-f",
        "--feeds",
        type=str,
        default="data/input/gbfs.json",
        help="Path to .json key specifying list of feeds and langs"
    )
    parser.add_argument(
        "-or",
        "--origin",
        type=str,
        default="stazione centrale p.za duca d'aosta m2 m3",
        help="Origin station of the journey",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="duomo m1 m3",
        help="Destination station of the journey",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="07:00:00",
        help="Departure time (hh:mm:ss)"
    )
    parser.add_argument(
        "-p",
        "--preferred",
        type=str,
        default="regular",
        help="Preferred type of vehicle (regular | electric | car)"
    )
    parser.add_argument(
        "-c",
        "--car",
        type=bool,
        default=True,
        help="Enable car-sharing transfers"
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds to execute the RAPTOR algorithm",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Output directory",
    )

    arguments = parser.parse_args()
    return arguments


def main(
    input_folder: str,
    feeds: str,
    origin_station: str,
    destination_station: str,
    departure_time: str,
    preferred: str,
    car: bool,
    rounds: int,
    output_folder: str
):
    """Run RAPTOR-sm algorithm"""

    logger.debug("Input directory     : {}", input_folder)
    logger.debug("Input shared-mob    : {}", feeds)
    logger.debug("Origin station      : {}", origin_station)
    logger.debug("Destination station : {}", destination_station)
    logger.debug("Departure time      : {}", departure_time)
    logger.debug("Preferred vehicle   : {}", preferred)
    logger.debug("Use car             : {}", car)
    logger.debug("Rounds              : {}", str(rounds))
    logger.debug("Output folder       : {}", output_folder)

    timetable = read_timetable(input_folder)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Reading shared mobility feed
    feed_infos: List[Dict] = json.load(open(feeds))['feeds']
    feeds: List[SharedMobilityFeed] = [SharedMobilityFeed(feed_info['url'], feed_info['lang']) for feed_info in feed_infos]
    logger.debug(f"{[feed.system_id for feed in feeds]} feeds got successfully")

    # Preferred bike type
    if preferred == 'car':
        value = 9001  # Car
    elif preferred == 'regular':
        value = 9002  # Bike
    elif preferred == 'electric':
        value = 9003  # ElectricBike
    else:
        raise ValueError(f"No {preferred} vehicle")

    preferred_vehicle = TransportType(value)

    # Car check
    if preferred_vehicle == TransportType.Car and not car:
        raise Exception("Preferred vehicle is car, but car-sharing transfers are disabled")

    # Find route between two stations
    journey_to_destinations = run_raptor(
        timetable,
        feeds,
        origin_station,
        dep_secs,
        rounds,
        preferred_vehicle,
        car
    )

    # Print journey to destination
    destination_journey = journey_to_destinations[destination_station]
    destination_journey.print()

    # Save the algorithm output
    algo_output = AlgorithmOutput(
        journey=destination_journey,
        date=timetable.date,
        departure_time=departure_time,
        original_gtfs_dir=timetable.original_gtfs_dir
    )
    AlgorithmOutput.save_to_dir(output_dir=output_folder,
                                algo_output=algo_output)


def run_raptor(
    timetable: Timetable,
    feeds: List[SharedMobilityFeed],
    origin_station: str,
    dep_secs: int,
    rounds: int,
    preferred_vehicle: TransportType,
    use_car: bool
) -> Dict[str, Journey]:
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
            journey_to_destinations[destination_station_name] = journey

    return journey_to_destinations



if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        feeds=args.feeds,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        preferred=args.preferred,
        car=args.car,
        rounds=args.rounds,
        output_folder=args.output
    )