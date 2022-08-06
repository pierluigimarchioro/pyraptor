"""Run query with RAPTOR algorithm"""
import argparse
from typing import Dict

from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.structures import Journey, Station, Timetable, AlgorithmOutput
from pyraptor.model.raptor import (
    RaptorAlgorithm,
    reconstruct_journey,
    best_stop_at_target_station,
)
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
    origin_station: str,
    destination_station: str,
    departure_time: str,
    rounds: int,
    output_folder: str
):
    """Run RAPTOR algorithm"""

    logger.debug("Input directory     : {}", input_folder)
    logger.debug("Origin station      : {}", origin_station)
    logger.debug("Destination station : {}", destination_station)
    logger.debug("Departure time      : {}", departure_time)
    logger.debug("Rounds              : {}", str(rounds))

    timetable = read_timetable(input_folder)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time seconds
    dep_secs = str2sec(departure_time)
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations
    journey_to_destinations = run_raptor(
        timetable,
        origin_station,
        dep_secs,
        rounds,
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
    origin_station: str,
    dep_secs: int,
    rounds: int,
) -> Dict[str, Journey]:
    """
    Run the Raptor algorithm.

    :param timetable: timetable
    :param origin_station: Name of origin station
    :param dep_secs: Time of departure in seconds
    :param rounds: Number of iterations to perform
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

            if dest_stop != 0:
                journey = reconstruct_journey(dest_stop, best_labels)
                journey_to_destinations[destination_station_name] = journey

        return journey_to_destinations
    except Exception as ex:
        logger.error(ex)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time,
        rounds=args.rounds,
        output_folder=args.output
    )
