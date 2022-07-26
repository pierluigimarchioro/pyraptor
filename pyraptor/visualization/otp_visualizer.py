from __future__ import annotations

import argparse
import os
import urllib.parse
import webbrowser as web
from dataclasses import dataclass
from subprocess import Popen, PIPE
from typing import Tuple, List
from datetime import datetime, timedelta, time
from time import sleep

import pandas as pd
from loguru import logger

import pyraptor.gtfs.io as io

from pyraptor.model.structures import AlgorithmOutput


# TODO generate (if not existing) files such as otp-config.json and build-config.json


@dataclass
class Coordinates:
    lat: float
    long: float


@dataclass
class JourneyInfo:
    from_place: Coordinates
    to_place: Coordinates


class OTPVisualizer:
    def __init__(self, otp2_exe_path: str | bytes | os.PathLike,
                 otp_working_dir: str | bytes | os.PathLike,
                 algo_output: AlgorithmOutput):
        self._otp2_exe_path: str = otp2_exe_path
        self._working_dir: str = otp_working_dir
        self._algo_output: AlgorithmOutput = algo_output

    def visualize_journey(self) -> Popen[str]:
        """
        Visualizes the provided journey in the OTP2 GUI.
        Returns the OTP2 server process handle.

        :return: OTP2 server process handle
        """

        logger.info(f"Processing GTFS at {self._algo_output.original_gtfs_dir}...")
        journey_info = self._process_gtfs()

        logger.info(f"Starting OTP2 Server...")
        server_process = self._launch_otp2_server()

        # Wait some seconds to make sure that the server properly started,
        # then launch the server GUI with the browser
        sleep(5)

        gui_url = self._get_gui_url(journey_info=journey_info)

        logger.info(f"Launching OTP2 GUI at {gui_url}")
        web.open(gui_url)

        return server_process

    def _process_gtfs(self) -> JourneyInfo:
        """
        Creates and saves a GTFS containing only the stops included in the journey provided to this instance,
        starting from the GTFS originally used to calculate the Raptor timetable and the aforementioned journey.
        Returns an object containing the coordinates of the first and the last stop of the journey.

        :return: object containing the coordinates of the first and the last stop of the journey
        """

        # Extract all the stop ids included in the journey
        # Convert them to string to make sure comparison work as intended
        journey_stops = []
        for leg in self._algo_output.journey.legs:
            journey_stops.append(leg.from_stop)
            journey_stops.append(leg.to_stop)
        journey_stop_ids: List[str] = list(set([str(s.id) for s in journey_stops]))

        gtfs_tables = io.read_gtfs_tables(self._algo_output.original_gtfs_dir)

        # Keep only the GTFS data related to the stops included in the journey
        # and the departures for the provided time window
        stops_table = gtfs_tables["stops"]
        stops_table["stop_id"] = stops_table["stop_id"].astype(str)
        itinerary_stops = stops_table[stops_table["stop_id"].isin(journey_stop_ids)]
        itinerary_stops = itinerary_stops[itinerary_stops["stop_id"].isin(journey_stop_ids)]

        # Extract only the stop times of stops included in the journey
        deps_table = gtfs_tables["stop_times"]
        deps_table["stop_id"] = deps_table["stop_id"].astype(str)
        deps_table["trip_id"] = deps_table["trip_id"].astype(str)
        itinerary_deps = deps_table[deps_table["stop_id"].isin(itinerary_stops["stop_id"])]

        # Extract only the stop times that are included in the journey time interval
        # i.e. journey departure time - journey arrival time
        dep_time, arriv_time = self._get_journey_time_interval()

        def parse_stop_time(t: str) -> time:
            try:
                return time.fromisoformat(t)
            except Exception:
                # Times in the stop_times table also include 25-26-27 etc. hours,
                # which will cause an exception. Default to midnight
                # TODO what happens if the journey ends after midnight of the next day?
                return time(hour=0, minute=0, second=0)

        stop_dep_times = itinerary_deps["departure_time"].apply(parse_stop_time)
        itinerary_deps = itinerary_deps[(dep_time <= stop_dep_times) & (stop_dep_times <= arriv_time)]

        stop_arriv_times = itinerary_deps["arrival_time"].apply(parse_stop_time)
        itinerary_deps = itinerary_deps[(dep_time <= stop_arriv_times) & (stop_arriv_times <= arriv_time)]

        # Get only trips included in the stop_times table
        trips_table = gtfs_tables["trips"]
        trips_table["trip_id"] = trips_table["trip_id"].astype(str)
        trips_table["route_id"] = trips_table["route_id"].astype(str)
        itinerary_trips = trips_table[trips_table["trip_id"].isin(itinerary_deps["trip_id"])]

        # Get only the routes for the filtered trips
        routes_table = gtfs_tables["routes"]
        routes_table["route_id"] = routes_table["route_id"].astype(str)
        itinerary_routes = routes_table[routes_table["route_id"].isin(itinerary_trips["route_id"])]

        # Get only the transfers that involve only stops included in the journey
        transfers_table = gtfs_tables["transfers"]
        transfers_table["from_stop_id"] = transfers_table["from_stop_id"].astype(str)
        transfers_table["to_stop_id"] = transfers_table["to_stop_id"].astype(str)
        itinerary_transfers = transfers_table[transfers_table["from_stop_id"].isin(itinerary_stops["stop_id"])]
        itinerary_transfers = itinerary_transfers[itinerary_transfers["to_stop_id"].isin(itinerary_stops["stop_id"])]

        # Set colors for visualization
        itinerary_routes.loc[:, "route_color"] = "31EB53"  # green

        # Save the GTFS
        to_save = gtfs_tables
        to_save["trips"] = itinerary_trips
        to_save["stops"] = itinerary_stops
        to_save["stop_times"] = itinerary_deps
        to_save["routes"] = itinerary_routes
        to_save["transfers"] = itinerary_transfers

        io.save_gtfs(gtfs_tables=to_save,
                     out_dir=self._working_dir,
                     gtfs_filename="visualization.gtfs.zip")

        return self._get_journey_info(stops_table=to_save["stops"])

    def _get_journey_time_interval(self) -> Tuple[time, time]:
        departure_datetime = self._get_departure_datetime()
        travel_time = self._algo_output.journey.travel_time()
        arrival_datetime = departure_datetime + timedelta(seconds=travel_time)

        return departure_datetime.time(), arrival_datetime.time()

    def _get_journey_info(self, stops_table: pd.DataFrame) -> JourneyInfo:
        journey_legs = self._algo_output.journey.legs
        first_stop = journey_legs[0].from_stop
        last_stop = journey_legs[len(journey_legs) - 1].to_stop

        first_stop_row = stops_table[stops_table["stop_id"] == first_stop.id].iloc[0]
        first_stop_coords = Coordinates(lat=float(first_stop_row["stop_lat"]),
                                        long=float(first_stop_row["stop_lon"]))

        last_stop_row = stops_table[stops_table["stop_id"] == last_stop.id].iloc[0]
        last_stop_coords = Coordinates(lat=float(last_stop_row["stop_lat"]),
                                       long=float(last_stop_row["stop_lon"]))

        return JourneyInfo(from_place=first_stop_coords, to_place=last_stop_coords)

    def _launch_otp2_server(self) -> Popen[str]:
        command = f"java -Xmx2G -jar {self._otp2_exe_path} --build --serve {self._working_dir} " \
                  f"--port 6900 --securePort 6901"  # TODO add as script args
        logger.debug(f"Starting server with command {command}")

        server_process = Popen(args=command.split(" "),
                               stdout=PIPE,
                               universal_newlines=True)

        while True:
            otp_log = server_process.stdout.readline()
            logger.info(f"[OTP2 Server] {otp_log.strip()}")

            # OTP server stared running (or terminated because of an error)
            # when the log contains the following strings
            if "Grizzly server" in otp_log or "OTP SHUTTING DOWN" in otp_log:
                break
            elif "ERROR" in otp_log:
                server_process.kill()

                raise Exception(otp_log)

        return server_process

    def _get_gui_url(self, journey_info: JourneyInfo) -> str:
        dep_datetime = self._get_departure_datetime()
        query_params = {
            "module": "planner",
            "fromPlace": f"{journey_info.from_place.lat},{journey_info.from_place.long}",
            "toPlace": f"{journey_info.to_place.lat},{journey_info.to_place.long}",
            "date": dep_datetime.date().strftime("%m-%d-%Y"),
            "time": dep_datetime.time().strftime("%I:%M %p"),  # time in am, pm format
            "mode": "TRANSIT,WALK",
            "arriveBy": "false",
            "wheelchair": "false",
            "showIntermediateStops": "true",
            "debugItineraryFilter": "false",
            "locale": "en"
        }
        query = "&".join([f"{key}={urllib.parse.quote_plus(value)}" for key, value in query_params.items()])
        gui_url = f"http://localhost:6900?{query}"  # TODO parameterize port number

        return gui_url

    def _get_departure_datetime(self) -> datetime:
        dep_date = self._algo_output.date
        dep_time = self._algo_output.departure_time

        return datetime.strptime(f"{dep_date} {dep_time}", "%Y%m%d %H:%M:%S")


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo_output",
        type=str,
        default="data/output/algo-output.pcl",
        help=".pcl file of an AlgorithmOutput instance",
    )
    parser.add_argument(
        "-e",
        "--otp2_exe",
        type=str,
        default="otp2/otp-2.1.0-shaded.jar",
        help="Path to an OTP2 executable (.jar file)",
    )
    parser.add_argument(
        "-w",
        "--otp2_working_dir",
        type=str,
        default="otp2/wd",
        help="Path to an OTP2 working directory, containing a .pbf file representing the street map "
             "of the zone covered by the journey to visualize",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    logger.info(f"Loading Raptor algorithm output from {args.algo_output}")
    output = AlgorithmOutput.read_from_file(filepath=args.algo_output)

    logger.info("Starting visualization...")
    visualizer = OTPVisualizer(
        otp2_exe_path=args.otp2_exe,
        otp_working_dir=args.otp2_working_dir,
        algo_output=output
    )
    otp2_process = visualizer.visualize_journey()

    user_input = -1
    while user_input != 0:
        try:
            user_input = int(input(f"Enter 0 to terminate the OTP2 server: "))
        except Exception:
            print("Unrecognized input. Please enter 0 to terminate the OTP2 server.")

    otp2_process.kill()
