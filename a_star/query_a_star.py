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
from pyraptor.util import str2sec, sec2str, sec2minutes

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

    # Input check
    if origin_station == destination_station:
        raise ValueError(f"{origin_station} is both origin and destination")

    logger.debug("Loading timetable...")
    timetbl_start_time = timer()
    timetable = read_timetable(input_folder=input_folder, timetable_name=TIMETABLE_FILENAME)
    timetbl_end_time = timer()

    logger.debug("Loading adjacency list...")
    adjlst_start_time = timer()
    adjacency_list = read_adjacency(output_folder)
    adjlst_end_time = timer()

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time
    dep_secs = sec2str(str2sec(departure_time))
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations + Print journey to destination
    path_start_time = timer()
    heuristic = get_heuristic(destination_station, timetable)
    graph = a_star.Graph(adjacency_list, heuristic, timetable, str2sec(departure_time))
    graph.a_star_algorithm(origin_station, destination_station)
    path_end_time = timer()

    # Save the algorithm output TODO finish adapting first, need to make a reasonable class for output
    # algo_output = AstarOutput(
    #     journeys=destination_journeys,
    #     departure_time=departure_time,
    #     date=timetable.date,
    #     original_gtfs_dir=timetable.original_gtfs_dir
    # )
    # AstarOutput.save(
    #     output_dir=output_folder,
    #     algo_output=algo_output
    # )

    # Todo visualizzazione in folium

    compute_path_timer = path_end_time - path_start_time
    load_adj_list_timer = adjlst_end_time - adjlst_start_time
    load_timetable_timer = timetbl_end_time - timetbl_start_time

    return load_timetable_timer, load_adj_list_timer, compute_path_timer


if __name__ == "__main__":
    args = parse_arguments()

    timetbl_time, adjlst_time, path_time = main(
        input_folder=args.input,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time
    )

    logger.info(f"Loading timetable time: {timetbl_time} sec ({sec2minutes(timetbl_time)})")
    logger.info(f"Loading adjacency list time: {adjlst_time} sec ({sec2minutes(adjlst_time)})")
    logger.info(f"Computing path time: {path_time} sec ({sec2minutes(path_time)})")
