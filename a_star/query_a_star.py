from loguru import logger
import argparse
from preprocessing import Step
from preprocessing import get_heuristic
from preprocessing import read_adjacency

import a_star

from pyraptor.dao.timetable import read_timetable
from pyraptor.util import str2sec
from pyraptor.util import sec2str


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

    timetable = read_timetable(input_folder)

    heuristic = get_heuristic(destination_station, timetable)
    adjacency_list = read_adjacency(output_folder)

    logger.info(f"Calculating network from: {origin_station}")

    # Departure time
    dep_secs = sec2str(str2sec(departure_time))
    logger.debug("Departure time (s.)  : " + str(dep_secs))

    # Find route between two stations
    graph = a_star.Graph(adjacency_list, heuristic, timetable, str2sec(departure_time))
    graph.a_star_algorithm(origin_station, destination_station)

    # Print journey to destination
    # qua stampa il viaggio trovato scritto --> gia stampa riga 104

    # Save the algorithm output
    # algo_output = AlgorithmOutput(
    #     journey=destination_journey,
    #     date=timetable.date,
    #     departure_time=departure_time,
    #     original_gtfs_dir=timetable.original_gtfs_dir
    # )
    # AlgorithmOutput.save(output_dir=output_folder,
    #                      algo_output=algo_output)

    # Todo visualizzazione in folium


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input_folder=args.input,
        output_folder=args.output,
        origin_station=args.origin,
        destination_station=args.destination,
        departure_time=args.time
    )

