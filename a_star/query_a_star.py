from __future__ import annotations
from loguru import logger
import argparse
import a_star

from a_star import AstarOutput
from preprocessing import Step
from preprocessing import get_heuristic
from preprocessing import read_adjacency

from timeit import default_timer as timer

from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import TIMETABLE_FILENAME
from pyraptor.util import str2sec, sec2str

_DEFAULT_FILENAME = "algo-output"


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output/milan",
        help="Input directory",
    )
    parser.add_argument(
        "-or",
        "--origin",
        type=str,
        default="A_CENTRALE FS",
        help="Origin station of the journey",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="A_DUOMO M1",
        help="Destination station of the journey",
    )
    parser.add_argument(
        "-t", "--time", type=str, default="08:35:00", help="Departure time (hh:mm:ss)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output/milan/a_star",
        help="Output directory",
    )

    arguments = parser.parse_args()
    return arguments


def main(
        input_folder: str,
        output_folder: str,
        origin_station: str,
        destination_station: str,
        departure_time: str
):
    """Run A Star algorithm"""

    logger.debug("Input directory       : {}", input_folder)
    logger.debug("Output directory      : {}", output_folder)
    logger.debug("Origin station        : {}", origin_station)
    logger.debug("Destination station   : {}", destination_station)
    logger.debug("Departure time        : {}", departure_time)

    start_time = timer()

    # Input check
    if origin_station == destination_station:
        raise ValueError(f"{origin_station} is both origin and destination")

    logger.debug("Loading timetable...")
    timetable = read_timetable(input_folder=input_folder, timetable_name=TIMETABLE_FILENAME)

    heuristic = get_heuristic(destination_station, timetable)

    logger.debug("Loading adjacency list...")
    adjacency_list = read_adjacency(output_folder)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time
    dep_secs = sec2str(str2sec(departure_time))
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations + Print journey to destination
    graph = a_star.Graph(adjacency_list, heuristic, timetable, str2sec(departure_time))
    graph.a_star_algorithm(origin_station, destination_station)

    # Save the algorithm output
    algo_output = AstarOutput(
        journeys=destination_journeys,
        departure_time=departure_time,
        date=timetable.date,
        original_gtfs_dir=timetable.original_gtfs_dir
    )
    AstarOutput.save(
        output_dir=output_folder,
        algo_output=algo_output
    )

    # Todo visualizzazione in folium

    end_time = timer()
    return end_time - start_time


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time
    )

