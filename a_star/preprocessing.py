from dataclasses import dataclass
import argparse
import joblib
import os
from loguru import logger
from pathlib import Path
from pyraptor.model.timetable import RaptorTimetable, TransportType
from pyraptor.timetable.io import read_timetable
from pyraptor.util import sec2minutes, mkdir_if_not_exists
from pyraptor.timetable.timetable import TIMETABLE_FILENAME
from timeit import default_timer as timer


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
        output_folder: str
):
    """Run preprocess for A Star algorithm"""

    logger.debug("Input directory       : {}", input_folder)
    logger.debug("Output directory      : {}", output_folder)

    logger.debug("Loading timetable...")
    timetbl_start_time = timer()
    timetable = read_timetable(input_folder=input_folder, timetable_name=TIMETABLE_FILENAME)
    timetbl_end_time = timer()

    logger.debug("Calculating adjacency list...")
    start_time = timer()
    get_adj_list(timetable, output_folder)
    end_time = timer()

    compute_adj_list_timer = end_time - start_time
    load_timetable_timer = timetbl_end_time - timetbl_start_time

    return load_timetable_timer, compute_adj_list_timer


@dataclass
class Step(object):
    """
    Step object to represent the weight of an edge of the graph
    It contains information about the departure from a stop to another, the route, and what transport is used
    """

    stop_to = None
    duration = None
    departure_time = None
    arrive_time = None
    trip_id = None
    route_id = None
    transport_type = None

    def __init__(self, stop_from, stop_to, duration, departure_time, arrive_time, trip_id, route_id, transport_type):
        """
        Initializes the stop node
        :param stop_from: start stop of the trip
        :param stop_to: destination stop of the trip
        :param duration: duration of the trip
        :param departure_time: arrive time from a stop
        :param arrive_time: arrive time to a stop
        :param trip_id: trip id of this step
        :param route_id: route id of this step
        :param transport_type: transport type of this route (route_type : The type of transportation used on a route)
        """

        self.stop_from = stop_from
        self.stop_to = stop_to
        self.duration = duration
        self.departure_time = departure_time
        self.arrive_time = arrive_time
        self.trip_id = trip_id
        self.route_id = route_id
        self.transport_type = transport_type


# def manhattan_heuristic(stop1, stop2) -> float:
#     """
#     Manhattan distance between two stops
#     :param stop1: Stop
#     :param stop2: Stop
#     :return: float distance
#     """
#     (x1, y1) = (stop1.lat, stop1.long)
#     (x2, y2) = (stop2.lat, stop2.long)
#     return abs(x1 - x2) + abs(y1 - y2)
#
#
# def euclidean_heuristic(stop1, stop2) -> float:
#     """
#     Euclidean distance between two stops
#     :param stop1: Stop
#     :param stop2: Stop
#     :return: float distance
#     """
#     return np.linalg.norm(stop1 - stop2)


def get_heuristic(destination, timetable: RaptorTimetable) -> dict[str, float]:
    """
    Time = distance/speed [hour]
    we use the average public transport speed in Italy [km/h]
    reference:
    https://www.statista.com/statistics/828636/public-transport-trip-speed-by-location-in-italy/
    we won't use Manhattan distance or Euclidean distance since we want the fastest travel not the shortest

    :param destination: destination station
    :param timetable: timetable
    :return: assign an heuristic value to every stops
    """
    heuristic = {}
    avg_speed = (14 + ((47 + 56) / 2)) / 2

    for st in timetable.stops:
        heuristic[st.id] = (st.distance_from(timetable.stops.get_stop(destination)) / avg_speed)*3600

    return heuristic


def get_adj_list(timetable: RaptorTimetable, output_folder) -> None:
    """
    contains all the neighbouring stops for some stop

    :param timetable: timetable
    :param output_folder: output directory
    :return: create adjacency list
    """

    adjacency_list: dict[str, list] = {}

    # per ogni fermata
    #   trovo tutte le fermate che riesce a raggiungere tramite un trip
    #   mi salvo la fermata vicina + duration, arrive_time, route_id, transport_type
    #   in qualche modo me li ricavo, transport type sta in route.txt, arrive_time e route_id ci sono gia stop_times.txt
    #   duration me la calcolo, come?
    #   duration deve essere calcolato anche considerando possibili momenti di attesa alla fermata per prendere il mezzo
    for st in timetable.stops:

        is_present_in_trip = {}
        # get all trips where st is in
        for arr in timetable.trips:
            if st.id in arr.trip_stop_ids():
                is_present_in_trip[arr.id] = arr

        # get all the next stops in the same trips of st
        adjacency_list[st.id] = []
        for tripid, tr in is_present_in_trip.items():
            stop_from = None
            got_seq = False
            seq = -2
            dep = 0
            for s in tr.stop_times:
                if s.stop.id == st.id and not got_seq:
                    seq = s.stop_idx
                    got_seq = True
                    dep = s.dts_arr
                    stop_from = s.stop
                if got_seq and s.stop_idx == seq+1:
                    adjacency_list[st.id].append(Step(stop_from=stop_from,
                                                      stop_to=s.stop,
                                                      duration=s.dts_arr-dep,
                                                      departure_time=dep,
                                                      arrive_time=s.dts_arr,
                                                      trip_id=tripid,
                                                      route_id=tr.route_info.name,
                                                      transport_type=tr.route_info.transport_type))

        for arr in timetable.transfers:
            if arr.from_stop == st:
                adjacency_list[st.id].append(Step(stop_from=arr.from_stop,
                                                  stop_to=arr.to_stop,
                                                  duration=arr.transfer_time,
                                                  departure_time="x",
                                                  arrive_time="x",
                                                  trip_id=arr.id,
                                                  route_id="x",
                                                  transport_type=TransportType.Walk))
                # departure time, arrive time and route id set to "x" because it's a transfer

    write_adjacency(output_folder, adjacency_list)


# def write_heuristic(output_folder: str, heuristic: dict[str, float]) -> None:
#     """
#     Write the heuristic list to output directory
#     """
#     with open(output_folder + '/heuristic.json', 'w') as outfile:
#         json.dump(heuristic, outfile, indent=4)
#
#     logger.debug("heuristic ok")
#
#
# def read_heuristic(input_folder: str) -> dict[str, float]:
#     """
#     Read the heuristic data from the cache directory
#     """
#     with open(input_folder + '/heuristic.json', 'r') as file:
#         data = json.load(file)
#
#     logger.debug("heuristic ok")
#
#     return data


def read_adjacency(input_folder: str) -> dict[str, list]:
    """
    Read the adjacency data from the cache directory
    """

    def load_joblib(name):
        logger.debug(f"Loading '{name}'")
        with open(Path(input_folder, f"{name}.pcl"), "rb") as handle:
            return joblib.load(handle)

    if not os.path.exists(input_folder):
        raise IOError(
            "adjacency data not found. Run `python a_star/preprocessing.py`"
            " first to create adjacency data from GTFS files."
        )

    logger.debug("Using cached datastructures")

    adjacency_list: dict[str, list] = load_joblib("adjacency")

    return adjacency_list


def write_adjacency(output_folder: str, adjacency_list: dict[str, list]) -> None:
    """
    Write the adjacency data to output directory
    """

    def write_joblib(state, name):
        with open(Path(output_folder, f"{name}.pcl"), "wb") as handle:
            joblib.dump(state, handle)

    logger.info("Writing adjacency data to output directory")

    mkdir_if_not_exists(output_folder)
    write_joblib(adjacency_list, "adjacency")


if __name__ == "__main__":
    args = parse_arguments()

    timetbl_time, adjlst_time = main(
        input_folder=args.input,
        output_folder=args.output
    )

    logger.info(f"Loading timetable time: {timetbl_time} sec ({sec2minutes(timetbl_time)})")
    logger.info(f"Computing adjacency list time: {adjlst_time} sec ({sec2minutes(adjlst_time)})")
