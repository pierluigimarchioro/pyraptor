"""Parse timetable from GTFS files"""
from __future__ import annotations

import itertools
import math
import os
import argparse
import calendar as cal
import uuid
from typing import List, Iterable, Any, NamedTuple, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime

import pandas as pd
import pathos.pools as p
from loguru import logger
from pathos.helpers.pp_helper import ApplyResult

from pyraptor.dao import write_timetable
from pyraptor.util import mkdir_if_not_exists, str2sec, TRANSFER_COST
from pyraptor.model.structures import (
    Timetable,
    Stop,
    Stops,
    Trip,
    Trips,
    TripStopTime,
    TripStopTimes,
    Station,
    Stations,
    Routes,
    Transfer,
    Transfers,
)


@dataclass
class GtfsTimetable:
    """Gtfs Timetable data"""

    trips = None
    calendar = None
    stop_times = None
    stops = None


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/input/NL-gtfs",
        help="Input directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Input directory",
    )
    parser.add_argument(
        "-d", "--date", type=str, default="20210906", help="Departure date (yyyymmdd)"
    )
    parser.add_argument("-a", "--agencies", nargs="+", default=["NS"])
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of jobs to run")

    arguments = parser.parse_args()
    return arguments


def main(
        input_folder: str,
        output_folder: str,
        departure_date: str,
        agencies: List[str],
        n_jobs: int
):
    """Main function"""

    logger.info("Parse timetable from GTFS files")
    mkdir_if_not_exists(output_folder)

    gtfs_timetable = read_gtfs_timetable(input_folder, departure_date, agencies)
    timetable = gtfs_to_pyraptor_timetable(gtfs_timetable, n_jobs)
    write_timetable(output_folder, timetable)


def read_gtfs_timetable(
        input_folder: str, departure_date: str, agencies: List[str]
) -> GtfsTimetable:
    """Extract operators from GTFS data"""

    logger.info("Read GTFS data")

    # Read agencies
    logger.debug("Read Agencies")

    agencies_df = pd.read_csv(os.path.join(input_folder, "agency.txt"))
    agencies_df = agencies_df.loc[agencies_df["agency_name"].isin(agencies)][
        ["agency_id", "agency_name"]
    ]
    agency_ids = agencies_df.agency_id.values

    # Read routes
    logger.debug("Read Routes")

    routes = pd.read_csv(os.path.join(input_folder, "routes.txt"))
    routes = routes[routes.agency_id.isin(agency_ids)]
    routes = routes[
        ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]
    ]

    # Read trips
    logger.debug("Read Trips")

    trips = pd.read_csv(os.path.join(input_folder, "trips.txt"))
    trips = trips[trips.route_id.isin(routes.route_id.values)]

    trips_col_selector = [
        "route_id",
        "service_id",
        "trip_id"
    ]

    # The trip short name is an optionally defined attribute in the GTFS standard
    t_short_name_col = "trip_short_name"
    if t_short_name_col in trips.columns:
        trips_col_selector.append(t_short_name_col)

        # TODO error because some values are missing - fillna with -1?
        trips[t_short_name_col] = trips[t_short_name_col].fillna(value=-1)
        trips[t_short_name_col] = trips[t_short_name_col].astype(int)

    trips = trips[trips_col_selector]

    # Read calendar
    logger.debug("Read Calendar")
    calendar_processor = GTFSCalendarProcessor(input_folder=input_folder)

    # Trips here are already filtered by agency ids
    valid_ids = trips["service_id"].values
    active_service_ids = calendar_processor.get_active_service_ids(on_date=departure_date,
                                                                   valid_service_ids=valid_ids)

    # Filter trips based on the service ids active on the provided dep. date
    trips = trips[trips["service_id"].isin(active_service_ids)]

    # Read stop times
    logger.debug("Read Stop Times")

    stop_times = pd.read_csv(
        os.path.join(input_folder, "stop_times.txt"), dtype={"stop_id": str}
    )
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id.values)]
    stop_times = stop_times[
        [
            "trip_id",
            "stop_sequence",
            "stop_id",
            "arrival_time",
            "departure_time",
        ]
    ]
    # Convert times to seconds
    stop_times["arrival_time"] = stop_times["arrival_time"].apply(str2sec)
    stop_times["departure_time"] = stop_times["departure_time"].apply(str2sec)

    # Read stops (platforms)
    logger.debug("Read Stops")

    stops_full = pd.read_csv(
        os.path.join(input_folder, "stops.txt"), dtype={"stop_id": str}
    )
    stops = stops_full.loc[
        stops_full["stop_id"].isin(stop_times.stop_id.unique())
    ].copy()

    # Read stopareas, i.e. stations # TODO parent_station is optional? it might need to be handled
    stopareas = stops["parent_station"].unique()
    # stops = stops.append(.copy())
    stops = pd.concat([stops, stops_full.loc[stops_full["stop_id"].isin(stopareas)]])

    # Make sure that stop_code is of string type
    stop_code_col = "stop_code"
    stops[stop_code_col] = stops[stop_code_col].astype(str)

    stops[stop_code_col] = stops.stop_code.str.upper()

    stops_col_selector = [
        "stop_id",
        "stop_name",  # TODO this is optional too (conditionally required)
        "parent_station",  # TODO conditionally required too
    ]

    platform_code_col = "platform_code",
    if platform_code_col in stops.columns:
        stops_col_selector.append(platform_code_col)

    stops = stops[stops_col_selector]

    # TODO commented because it excludes stops with location_type 0 or empty,
    #   which are standalone stops that should not be discarded.
    #   I think the rationale behind removing parent stations
    #   is that the child stops are already included, so including the station would basically
    #   mean double-counting. This means that the stops that should actually be removed are the ones
    #   with location_type == 1 (Stations) and parent_station == empty
    # Filter out the general station codes
    # stops = stops.loc[~stops.parent_station.isna()]

    gtfs_timetable = GtfsTimetable()
    gtfs_timetable.trips = trips
    gtfs_timetable.stop_times = stop_times
    gtfs_timetable.stops = stops

    return gtfs_timetable


class GTFSCalendarProcessor(object):
    """
    Class that handles the processing of the calendar and calendar_dates tables
    of the provided GTFS feed.
    """

    def __init__(self, input_folder: str | bytes | os.PathLike):
        """
        :param input_folder: path to the folder containing the calendar
            and/or calendar_dates tables
        """

        calendar_path = os.path.join(input_folder, "calendar.txt")
        if os.path.exists(calendar_path):
            self.calendar: pd.DataFrame = pd.read_csv(calendar_path, dtype={"start_date": str, "end_date": str})
        else:
            self.calendar = None

        calendar_dates_path = os.path.join(input_folder, "calendar_dates.txt")
        if os.path.exists(calendar_dates_path):
            self.calendar_dates: pd.DataFrame = pd.read_csv(calendar_dates_path, dtype={"date": str})
        else:
            self.calendar_dates = None

    def get_active_service_ids(self, on_date: str, valid_service_ids: Iterable) -> Iterable:
        """
        Returns the list of service ids active in the provided date. Said service ids
        are extracted from the calendar and calendar_dates tables.

        :param on_date: date to get the active services for
        :param valid_service_ids: list of service ids considered valid.
            Any service_id outside this list will not be considered in the calculations.
        :return: list of service ids active in the provided date
        """

        if self._is_recommended_calendar_repr():
            return self._handle_recommended_calendar(on_date, valid_service_ids)
        elif self._is_alternate_calendar_repr():
            return self._handle_alternate_calendar(on_date, valid_service_ids)
        else:
            raise Exception("Calendar Representation Not Handled")

    def _is_recommended_calendar_repr(self) -> bool:
        """
        Returns true if the service calendar is represented in the recommended way:
        calendar table for regular services, calendar dates table for exceptions.
        :return:
        """

        # If calendar table is present and not empty, the service calendar
        # is in the recommended way
        return self.calendar is not None and len(self.calendar) > 0

    def _handle_recommended_calendar(self, date: str, valid_service_ids: Iterable) -> pd.Series:
        """
        Returns the list of service ids active in the provided date, only if those ids
        are included in the provided valid service ids list.

        :param date:
        :param valid_service_ids:
        :return:
        """

        # Consider only valid service_ids
        only_valid_services = self.calendar[self.calendar["service_id"].isin(valid_service_ids)]

        def is_service_active_on_date(row):
            # Check if date is included in service interval
            date_format = "%Y%m%d"
            start_date = datetime.strptime(row["start_date"], date_format)
            end_date = datetime.strptime(row["end_date"], date_format)
            date_to_check = datetime.strptime(date, date_format)
            is_in_service_interval = start_date <= date_to_check <= end_date

            # Check if the service is active in the weekday of the provided date
            weekday_to_col = {
                0: "monday",
                1: "tuesday",
                2: "wednesday",
                3: "thursday",
                4: "friday",
                5: "saturday",
                6: "sunday",
            }
            weekday = cal.weekday(date_to_check.year, date_to_check.month, date_to_check.day)
            is_weekday_active = row[weekday_to_col[weekday]] == 1

            exception_type = self._get_exception_for_service_date(row["service_id"], date)

            # Service is normally active and no exception on that day
            is_normally_active = is_in_service_interval and is_weekday_active and exception_type == -1

            # Service is active because of an exceptional date
            is_exceptionally_active = not (is_in_service_interval and is_weekday_active) and exception_type == 1

            return is_normally_active or is_exceptionally_active

        # Extract only the rows of the services active on the provided date
        active_on_date_mask = only_valid_services.apply(is_service_active_on_date, axis="columns")

        services_active_on_date = only_valid_services[active_on_date_mask]

        return services_active_on_date["service_id"]

    def _is_alternate_calendar_repr(self):
        """
        Returns true if the service calendar is represented in the alternate way:
        just the calendar dates table where each record represents a service day
        :return:
        """

        # If the only calendar table is calendar_dates, then this is the
        # alternate way of representing the service calendar
        return self.calendar is None \
               and self.calendar_dates is not None \
               and len(self.calendar_dates) > 0

    def _handle_alternate_calendar(self, date: str, valid_service_ids: Iterable) -> Iterable:
        """
        Returns the list of service ids active in the provided date, only if those ids
        are included in the provided valid service ids list.

        :param date:
        :param valid_service_ids:
        :return:
        """
        active_service_ids = []
        for s_id in valid_service_ids:
            ex_type = self._get_exception_for_service_date(service_id=s_id, date=date)

            if ex_type == 1:
                active_service_ids.append(s_id)

        return active_service_ids

    def _get_exception_for_service_date(self, service_id: Any, date: str) -> int:
        """
        Tries to retrieve the exception defined in the calendar_dates table for
        the provided date. Returns an integer code representing the exception type.

        :param date: date to check exception for
        :return: 3 different integer values:
            * -1 if no exception was found for the provided date
            * 1 if the service is exceptionally active in the provided date
            * 2 if the service is exceptionally not active in the provided date
        """

        try:
            # Extract exceptions for the provided service id
            service_exceptions: pd.DataFrame = self.calendar_dates[self.calendar_dates["service_id"] == service_id]

            # Extract the exception type for the provided date
            exception_on_date = service_exceptions[service_exceptions["date"] == date]
            exception_type = int(exception_on_date["exception_type"].iloc[0])

            return exception_type
        except Exception as x:
            # Exception not found
            # logger.debug(f"Exception type for service {service_id} on date {date} not found. Reason: {x}")
            return -1


class TripsProcessor:
    """
    Class that handles the processing of GtfsTimetable trips, written with multi-threading in mind.
    """

    @staticmethod
    def get_processor(trips_row_iterator: Iterable[NamedTuple],
                      stops_info: Stops,
                      stop_times_by_trip_id: Mapping[Any, List],
                      processor_id: uuid.UUID | int | str = uuid.uuid4()) -> Callable[[], Tuple[Trips, TripStopTimes]]:
        """
        Returns a function that processes the provided trips.
        The resulting Trip and TripStopTime instances will be added to the provided storage.

        :param trips_row_iterator: iterator that cycles over the rows of
            a GtfsTimetable.trips dataframe
        :param stops_info: collection of stop instances that contain detailed information
            for each stop
        :param stop_times_by_trip_id: default dictionary where keys are trip ids and
            values are collections of stop times.
        :param processor_id: id to assign to this processor.
            Its purpose is only to identify this instance in the logger output.
        :return: tuple containing the generated collections of Trip and TripStopTime instances
        """

        def process_trips() -> Tuple[Trips, TripStopTimes]:
            trips = Trips()
            trip_stop_times = TripStopTimes()

            # DEBUG: Keep track of progress since this operation is relatively heavy
            processed_trips = -1
            prev_pct_point = -1

            trip_rows = list(trips_row_iterator)
            for row in trip_rows:
                processed_trips += 1
                table_length = len(trip_rows)
                current_pct = math.floor((processed_trips / table_length) * 100)

                if math.floor(current_pct) > prev_pct_point or current_pct == 100:
                    log(f"Progress: {current_pct}% [trip #{processed_trips} of {table_length}]")
                    prev_pct_point = current_pct

                trip = Trip()

                # This is an optionally defined attribute in the GTFS standard
                trip.hint = getattr(row, "trip_short_name", "missing_hint")  # i.e. train number

                # Iterate over stops, ordered by sequence number:
                # the first stop will be the one with stop_sequence == 1
                sort_stop_times = sorted(
                    stop_times_by_trip_id[row.trip_id], key=lambda s: int(s.stop_sequence)
                )
                for stop_number, stop_time in enumerate(sort_stop_times):
                    # Timestamps
                    dts_arr = stop_time.arrival_time
                    dts_dep = stop_time.departure_time

                    # Trip Stop Times
                    stop = stops_info.get(stop_time.stop_id)

                    # TODO ICD fare not calculated since it is specific (apparently) to the sample Dutch GTFS
                    # GTFS files do not contain ICD supplement fare, so hard-coded here
                    # fare = calculate_icd_fare(trip, stop, stations) if icd_fix is True else 0
                    trip_stop_time = TripStopTime(trip, stop_number, stop, dts_arr, dts_dep)

                    trip_stop_times.add(trip_stop_time)
                    trip.add_stop_time(trip_stop_time)

                # Add trip TODO why this if? isn't trip always != None?
                if trip:
                    trips.add(trip)

            log(f"Processing completed")

            return trips, trip_stop_times

        def log(msg: str):
            logger.debug(f"[TripsProcessor {processor_id}] {msg}")

        return process_trips


def gtfs_to_pyraptor_timetable(
        gtfs_timetable: GtfsTimetable,
        n_jobs: int) -> Timetable:
    """
    Convert timetable for usage in Raptor algorithm.
    """

    logger.info("Convert GTFS timetable to timetable for PyRaptor algorithm")

    # Stations and stops, i.e. platforms
    logger.debug("Add stations and stops")

    stations = Stations()
    stops = Stops()

    platform_code_col = "platform_code"
    if platform_code_col in gtfs_timetable.stops.columns:
        gtfs_timetable.stops.platform_code = gtfs_timetable.stops.platform_code.fillna("missing_platform_code")

    for s in gtfs_timetable.stops.itertuples():
        station = Station(s.stop_name, s.stop_name)
        station = stations.add(station)

        platform_code = getattr(s, platform_code_col, "missing_platform_code")
        stop_id = f"{s.stop_name}-{platform_code}"
        stop = Stop(s.stop_id, stop_id, station, platform_code)

        station.add_stop(stop)
        stops.add(stop)

    # Stop Times
    stop_times = defaultdict(list)
    for stop_time in gtfs_timetable.stop_times.itertuples():
        stop_times[stop_time.trip_id].append(stop_time)

    # Trips and Trip Stop Times
    logger.debug("Add trips and trip stop times")

    job_results: dict[int, ApplyResult] = {}
    pool = p.ProcessPool(nodes=n_jobs)
    for i in range(n_jobs):
        processor_id = i
        logger.debug(f"Starting Trips Processor Job #{processor_id}...")

        total_trips = len(gtfs_timetable.trips)
        interval_length = math.floor(total_trips / n_jobs)
        start = i*interval_length
        end = (start + interval_length) - 1  # -1 because the interval_length-th trip belongs to the next round

        processor = TripsProcessor.get_processor(
            trips_row_iterator=itertools.islice(gtfs_timetable.trips.itertuples(), start, end),
            stops_info=stops,
            stop_times_by_trip_id=stop_times,
            processor_id=processor_id
        )

        job_results[processor_id] = pool.apipe(processor)

    pool.close()

    logger.debug(f"Waiting for jobs to finish...")

    trips = Trips()
    trip_stop_times = TripStopTimes()
    for p_id, result in job_results.items():
        res: Tuple[Trips, TripStopTimes] = result.get()
        logger.debug(f"Processor #{p_id} has completed its execution")
        logger.debug(f"Trips produced: {len(res[0])}; TripStopTimes produced: {len(res[1])}")

        # Add the results
        for res_trip in res[0].set_idx.values():
            trips.add(res_trip)

        for res_trip_stop_time in res[1].set_idx.values():
            trip_stop_times.add(res_trip_stop_time)

    # Make sure all the jobs are finished
    pool.join()

    # Routes
    logger.debug("Add routes")

    routes = Routes()
    for trip in trips:
        routes.add(trip)

    # Transfers
    logger.debug("Add transfers")

    transfers = Transfers()
    for station in stations:
        station_stops = station.stops
        station_transfers = [
            Transfer(from_stop=stop_i, to_stop=stop_j, layovertime=TRANSFER_COST)
            for stop_i in station_stops
            for stop_j in station_stops
            if stop_i != stop_j
        ]
        for st in station_transfers:
            transfers.add(st)

    # Timetable
    timetable = Timetable(
        stations=stations,
        stops=stops,
        trips=trips,
        trip_stop_times=trip_stop_times,
        routes=routes,
        transfers=transfers,
    )
    timetable.counts()

    return timetable


if __name__ == "__main__":
    args = parse_arguments()
    main(input_folder=args.input, output_folder=args.output,
         departure_date=args.date, agencies=args.agencies,
         n_jobs=args.jobs)
