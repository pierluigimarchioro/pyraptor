"""Datatypes"""
from __future__ import annotations

import os
import uuid
from abc import abstractmethod, ABC
from collections.abc import Iterable
from itertools import compress
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from itertools import compress
from json import loads
from operator import attrgetter
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Mapping
from urllib.request import urlopen

import attr
import joblib
import numpy as np
from geopy.distance import geodesic
from loguru import logger

from pyraptor.util import sec2str, mkdir_if_not_exists, get_transport_type_description, TRANSFER_TYPE, \
    MEAN_FOOT_SPEED, TransferType, VEHICLE_SPEED, TRANSFER_COST


def same_type_and_id(first, second):
    """Same type and ID"""
    return type(first) is type(second) and first.id == second.id


@dataclass
class TimetableInfo:
    # Path to the directory of the GTFS feed originally
    # used to generate the current Timetable instance
    original_gtfs_dir: str | bytes | os.PathLike = None

    # Date that the timetable refers to.
    # Format: YYYYMMDD which is equal to %Y%m%d
    date: str = None


@dataclass
class Timetable(TimetableInfo):
    """Timetable data"""

    stations: Stations = None
    stops: Stops = None
    trips: Trips = None
    trip_stop_times: TripStopTimes = None
    routes: Routes = None
    transfers: Transfers = None

    def counts(self) -> None:
        """Print timetable counts"""
        logger.debug("Counts:")
        logger.debug("Stations   : {}", len(self.stations))
        logger.debug("Routes     : {}", len(self.routes))
        logger.debug("Trips      : {}", len(self.trips))
        logger.debug("Stops      : {}", len(self.stops))
        logger.debug("Stop Times : {}", len(self.trip_stop_times))
        logger.debug("Transfers  : {}", len(self.transfers))


@attr.s(repr=False, cmp=False)
class Coordinates:
    lat: float = attr.ib(default=None)
    lon: float = attr.ib(default=None)

    @property
    def to_tuple(self) -> Tuple[float, float]:
        return self.lat, self.lon

    @property
    def to_list(self) -> List[float]:
        return [self.lat, self.lon]

    def __eq__(self, coord: Coordinates):
        return self.lat == coord.lat and self.lon == coord.lon

    def __repr__(self):
        return f"({self.lat}, {self.lon})"


@attr.s(repr=False, cmp=False)
class Stop:
    """Stop"""
    id = attr.ib(default=None)
    name = attr.ib(default=None)
    station: Station = attr.ib(default=None)
    platform_code = attr.ib(default=None)
    index = attr.ib(default=None)
    geo: Coordinates = attr.ib(default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, stop):
        return type(self) is type(stop) and self.id == stop.id

    def __repr__(self):
        if self.id == self.name:
            return f"Stop({self.id})"
        return f"Stop({self.name} [{self.id}])"

    @staticmethod
    def stop_distance(a: Stop, b: Stop) -> float:
        """Returns stop distance as the crow flies in km"""
        return geodesic((a.geo.lat, a.geo.lon), (b.geo.lat, b.geo.lon)).km

    def distance_from(self, s: Stop) -> float:
        """Returns stop distance as the crow flies in km"""
        return Stop.stop_distance(self, s)


class Stops:
    """Stops"""

    def __init__(self):
        self.set_idx = dict()
        self.set_index = dict()
        self.last_index = 1

    def __repr__(self):
        return f"Stops(n_stops={len(self.set_idx)})"

    def __getitem__(self, stop_id):
        return self.set_idx[stop_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def get(self, stop_id) -> Stop:
        """Get stop"""
        if stop_id not in self.set_idx:
            raise ValueError(f"Stop ID {stop_id} not present in Stops")
        stop: Stop = self.set_idx[stop_id]
        return stop

    def get_by_index(self, stop_index) -> Stop:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add(self, stop: Stop) -> Stop:
        """Add stop"""
        if stop.id in self.set_idx:
            stop = self.set_idx[stop.id]
        else:
            stop.index = self.last_index
            self.set_idx[stop.id] = stop
            self.set_index[stop.index] = stop
            self.last_index += 1
        return stop

    @property
    def public_transport_stop(self) -> List[Stop]:
        """ Returns its public stops  """
        return self.filter_public_transport(self)

    @property
    def shared_mobility_stops(self) -> List[RentingStation]:
        """ Returns its shared mobility stops  """
        return self.filter_shared_mobility(self)

    @staticmethod
    def filter_public_transport(stops: Iterable[Stop]) -> List[Stop]:
        """ Filter only Stop objects, not its subclasses  """
        return [s for s in stops if type(s) == Stop]

    @staticmethod
    def filter_shared_mobility(stops: Iterable[Stop]) -> list[RentingStation]:
        """ Filter only subclasses of RentingStation  """
        return [s for s in stops if isinstance(s, RentingStation)]


@attr.s(repr=False, cmp=False)
class Station:
    """Stop dataclass"""

    id = attr.ib(default=None)
    name = attr.ib(default=None)
    stops: List[Stop] = attr.ib(default=attr.Factory(list))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, stop):
        return same_type_and_id(self, stop)

    def __repr__(self):
        if self.id == self.name:
            return "Station({})".format(self.id)
        return "Station({} [{}])>".format(self.name, self.id)

    def add_stop(self, stop: Stop):
        self.stops.append(stop)


class Stations:
    """Stations"""

    def __init__(self):
        self.set_idx = dict()

    def __repr__(self):
        return f"<Stations(n_stations={len(self.set_idx)})>"

    def __getitem__(self, station_id):
        return self.set_idx[station_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, station: Station):
        """Add station"""
        if station.id in self.set_idx:
            station = self.set_idx[station.id]
        else:
            self.set_idx[station.id] = station
        return station

    def get(self, station: Station | str):
        """Get station"""
        if isinstance(station, Station):
            station = station.id
        if station not in self.set_idx:
            return None
        return self.set_idx[station]

    def get_stops(self, station_name) -> List[Stop]:
        """Get all stop ids from station, i.e. platform stop ids belonging to station"""
        return self.set_idx[station_name].stops


@attr.s(repr=False)
class TripStopTime:
    """Trip Stop"""

    trip: Trip = attr.ib(default=attr.NOTHING)
    stop_idx = attr.ib(default=attr.NOTHING)
    stop = attr.ib(default=attr.NOTHING)

    # Time of arrival in seconds past midnight
    dts_arr = attr.ib(default=attr.NOTHING)

    # Time of departure in seconds past midnight
    dts_dep = attr.ib(default=attr.NOTHING)
    fare = attr.ib(default=0.0)

    def __hash__(self):
        return hash((self.trip, self.stop_idx))

    def __repr__(self):
        return (
            "TripStopTime(trip_id={hint}{trip_id}, stopidx={0.stopidx},"
            " stop_id={0.stop.id}, dts_arr={0.dts_arr}, dts_dep={0.dts_dep}, fare={0.fare})"
        ).format(
            self,
            trip_id=self.trip.id if self.trip else None,
            hint="{}:".format(self.trip.hint) if self.trip and self.trip.hint else "",
        )


class TripStopTimes:
    """Trip Stop Times"""

    def __init__(self):
        self.set_idx: Dict[Tuple[Trip, int], TripStopTime] = dict()
        self.stop_trip_idx: Dict[Stop, List[TripStopTime]] = defaultdict(list)

    def __repr__(self):
        return f"TripStoptimes(n_tripstoptimes={len(self.set_idx)})"

    def __getitem__(self, trip_id):
        return self.set_idx[trip_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, trip_stop_time: TripStopTime):
        """Add trip stop time"""
        self.set_idx[(trip_stop_time.trip, trip_stop_time.stop_idx)] = trip_stop_time
        self.stop_trip_idx[trip_stop_time.stop].append(trip_stop_time)

    def get_trip_stop_times_in_range(self, stops, dep_secs_min, dep_secs_max):
        """Returns all trip stop times with departure time within range"""
        in_window = [
            tst
            for tst in self
            if tst.dts_dep >= dep_secs_min
               and tst.dts_dep <= dep_secs_max
               and tst.stop in stops
        ]
        return in_window

    def get_earliest_trip(self, stop: Stop, dep_secs: int) -> Trip:
        """Earliest trip"""
        trip_stop_times = self.stop_trip_idx[stop]
        in_window = [tst for tst in trip_stop_times if tst.dts_dep >= dep_secs]
        return in_window[0].trip if len(in_window) > 0 else None

    def get_earliest_trip_stop_time(self, stop: Stop, dep_secs: int) -> TripStopTime:
        """Earliest trip stop time"""
        trip_stop_times = self.stop_trip_idx[stop]
        in_window = [tst for tst in trip_stop_times if tst.dts_dep >= dep_secs]
        return in_window[0] if len(in_window) > 0 else None


@dataclass(frozen=True)
class RouteInfo:
    transport_type: int = None
    name: str = None

    @staticmethod
    def get_transfer_route(vtype: TransferType = None) -> RouteInfo:
        if vtype is None:
            return RouteInfo(transport_type=TRANSFER_TYPE, name="walk path")
        else:
            return RouteInfo(transport_type=TRANSFER_TYPE, name=f"{vtype.value}-sharing")

    def __str__(self):
        return f"Transport: {get_transport_type_description(self.transport_type)} | Route Name: {self.name}"

    def __eq__(self, other):
        if isinstance(other, RouteInfo):
            return other.transport_type == self.transport_type and other.name == self.name
        else:
            raise Exception(f"Cannot compare {RouteInfo.__name__} with {type(other)}")


@attr.s(repr=False, cmp=False)
class Trip:
    """Trip"""

    id = attr.ib(default=None)
    stop_times = attr.ib(default=attr.Factory(list))
    stop_times_index = attr.ib(default=attr.Factory(dict))
    hint = attr.ib(default=None)  # i.e. trip_short_name
    long_name = attr.ib(default=None)  # e.g., Sprinter
    route_info: RouteInfo = attr.ib(default=None)

    @staticmethod
    def get_transfer_trip(from_stop: Stop, to_stop: Stop, dep_time: int,
                          arr_time: int, vtype: TransferType = None) -> Trip:
        transfer_route = RouteInfo.get_transfer_route(vtype)

        transfer_trip = Trip(
            id=f"Transfer Trip - {uuid.uuid4()}",
            long_name=f"Transfer Trip from {from_stop.name} to {to_stop.name}",
            route_info=transfer_route,
            hint=str(transfer_route)
        )

        dep_stop_time = TripStopTime(trip=transfer_trip, stop_idx=0, stop=from_stop, dts_arr=dep_time, dts_dep=dep_time)
        arr_stop_time = TripStopTime(trip=transfer_trip, stop_idx=1, stop=to_stop, dts_arr=arr_time, dts_dep=arr_time)

        transfer_trip.add_stop_time(dep_stop_time)
        transfer_trip.add_stop_time(arr_stop_time)

        return transfer_trip

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return "Trip(hint={hint}, stop_times={stop_times})".format(
            hint=self.hint if self.hint is not None else self.id,
            stop_times=len(self.stop_times),
        )

    def __getitem__(self, n):
        return self.stop_times[n]

    def __len__(self):
        return len(self.stop_times)

    def __iter__(self):
        return iter(self.stop_times)

    def trip_stop_ids(self):
        """Tuple of all stop ids in trip"""
        return tuple([s.stop.id for s in self.stop_times])

    def add_stop_time(self, stop_time: TripStopTime):
        """Add stop time"""
        if np.isfinite(stop_time.dts_arr) and np.isfinite(stop_time.dts_dep):
            assert stop_time.dts_arr <= stop_time.dts_dep
            assert (
                    not self.stop_times or self.stop_times[-1].dts_dep <= stop_time.dts_arr
            )
        self.stop_times.append(stop_time)
        self.stop_times_index[stop_time.stop] = len(self.stop_times) - 1

    def get_stop(self, stop: Stop) -> TripStopTime:
        """Get stop"""
        return self.stop_times[self.stop_times_index[stop]]

    def get_fare(self, depart_stop: Stop) -> int:
        """Get fare from depart_stop"""
        stop_time = self.get_stop(depart_stop)
        return 0 if stop_time is None else stop_time.fare


class Trips:
    """Trips"""

    def __init__(self):
        self.set_idx = dict()
        self.last_id = 1

    def __repr__(self):
        return f"Trips(n_trips={len(self.set_idx)})"

    def __getitem__(self, trip_id):
        return self.set_idx[trip_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, trip):
        """Add trip"""
        assert len(trip) >= 2, "must have 2 stop times"
        trip.id = self.last_id
        self.set_idx[trip.id] = trip
        self.last_id += 1


@attr.s(repr=False, cmp=False)
class Route:
    """Route"""

    id = attr.ib(default=None)
    trips: List[Trip] = attr.ib(default=attr.Factory(list))
    stops: List[Stop] = attr.ib(default=attr.Factory(list))
    stop_order = attr.ib(default=attr.Factory(dict))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return "Route(id={0.id}, trips={trips})".format(self, trips=len(self.trips), )

    def __getitem__(self, n):
        return self.trips[n]

    def __len__(self):
        return len(self.trips)

    def __iter__(self):
        return iter(self.trips)

    def add_trip(self, trip: Trip) -> None:
        """Add trip"""
        self.trips.append(trip)

    def add_stop(self, stop: Stop) -> None:
        """Add stop"""
        self.stops.append(stop)
        # (re)make dict to save the order of the stops in the route
        self.stop_order = {stop: index for index, stop in enumerate(self.stops)}

    def stop_index(self, stop: Stop):
        """Stop index"""
        return self.stop_order[stop]

    def earliest_trip(self, dts_arr: int, stop: Stop) -> Trip:
        """
        Returns the earliest trip that can be boarded at the provided stop in the
        current route after `dts_arr` time (in seconds after midnight)

        :param dts_arr: time in seconds after midnight that a trip can be boarded after
        :param stop: stop to board the trip at
        :return: earliest trip that can be boarded, or None if no trip
        """

        stop_idx = self.stop_index(stop)
        trip_stop_times = [trip.stop_times[stop_idx] for trip in self.trips]
        trip_stop_times = [tst for tst in trip_stop_times if tst.dts_dep >= dts_arr]
        trip_stop_times = sorted(trip_stop_times, key=attrgetter("dts_dep"))

        return trip_stop_times[0].trip if len(trip_stop_times) > 0 else None

    def earliest_trip_stop_time(self, dts_arr: int, stop: Stop) -> TripStopTime:
        """
        Returns the stop time for the provided stop in the current route
        from the earliest trip that can be boarded after `dts_arr` time.

        :param dts_arr: time in seconds after midnight that a trip can be boarded after
        :param stop: stop to board the trip at
        :return: stop time for the provided stop in the earliest boardable trip, or None if any
        """

        stop_idx = self.stop_index(stop)
        trip_stop_times = [trip.stop_times[stop_idx] for trip in self.trips]
        trip_stop_times = [tst for tst in trip_stop_times if tst.dts_dep >= dts_arr]
        trip_stop_times = sorted(trip_stop_times, key=attrgetter("dts_dep"))

        return trip_stop_times[0] if len(trip_stop_times) > 0 else None


class Routes:
    """Routes"""

    def __init__(self):
        self.set_idx = dict()
        self.set_stops_idx = dict()
        self.stop_to_routes = defaultdict(list)  # {Stop: [Route]}
        self.last_id = 1

    def __repr__(self):
        return f"Routes(n_routes={len(self.set_idx)})"

    def __getitem__(self, route_id):
        return self.set_idx[route_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, trip: Trip) -> Route:
        """Add trip to route. Make route if not exists."""
        trip_stop_ids = trip.trip_stop_ids()

        if trip_stop_ids in self.set_stops_idx:
            # Route already exists
            route = self.set_stops_idx[trip_stop_ids]
        else:
            # Route does not exist yet, make new route
            route = Route()
            route.id = self.last_id

            # Maintain stops in route and list of routes per stop
            for trip_stop_time in trip:
                route.add_stop(trip_stop_time.stop)
                self.stop_to_routes[trip_stop_time.stop].append(route)

            # Efficient lookups
            self.set_stops_idx[trip_stop_ids] = route
            self.set_idx[route.id] = route
            self.last_id += 1

        # Add trip
        route.add_trip(trip)
        return route

    def get_routes_of_stop(self, stop: Stop):
        """Get routes of stop"""
        return self.stop_to_routes[stop]


@attr.s(repr=False, cmp=False)
class Transfer:
    """Transfer"""

    id: str | None = attr.ib(default=None)
    from_stop: Stop | None = attr.ib(default=None)
    to_stop: Stop | None = attr.ib(default=None)

    # Time in seconds that the transfer takes to complete
    transfer_time = attr.ib(default=TRANSFER_COST)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return f"Transfer(from_stop={self.from_stop}, to_stop={self.to_stop}, transfer_time={self.transfer_time})"

    @staticmethod
    def get_transfer(sa: Stop, sb: Stop) -> Tuple[Transfer, Transfer]:
        """ Given two stops compute both inbound and outbound transfers
            Transfer time is approximated dividing computed distance by a constant speed """
        dist: float = Stop.stop_distance(sa, sb)
        time: int = int(dist * 3600 / MEAN_FOOT_SPEED)
        return (
            Transfer(from_stop=sa, to_stop=sb, transfer_time=time),
            Transfer(from_stop=sb, to_stop=sa, transfer_time=time)
        )


class Transfers:
    """Transfers"""

    def __init__(self):
        self.set_idx = dict()
        self.stop_to_stop_idx = dict()
        self.last_id = 1

    def __repr__(self):
        return f"Transfers(n_transfers={len(self.set_idx)})"

    def __getitem__(self, transfer_id):
        return self.set_idx[transfer_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, transfer: Transfer):
        """Add trip"""
        transfer.id = self.last_id
        self.set_idx[transfer.id] = transfer
        self.stop_to_stop_idx[(transfer.from_stop, transfer.to_stop)] = transfer
        self.last_id += 1

    def with_from_stop(self, from_: Stop) -> List[Transfer]:
        """ Returns all transfers with given departing stop  """
        return [
            self.stop_to_stop_idx[(f, t)] for f, t in self.stop_to_stop_idx.keys() if f == from_
        ]

    def with_to_stop(self, to: Stop) -> List[Transfer]:
        """ Returns all transfers with given arrival stop  """
        return [
            self.stop_to_stop_idx[(f, t)] for f, t in self.stop_to_stop_idx.keys() if t == to
        ]

    def with_stop(self, s) -> List[Transfer]:
        """ Returns all transfers with given stop as departing or arrival  """
        return self.with_from_stop(s) + self.with_to_stop(s)


@dataclass
class Leg:
    """Leg"""

    from_stop: Stop
    to_stop: Stop
    trip: Trip
    earliest_arrival_time: int
    fare: int = 0
    n_trips: int = 0

    @property
    def criteria(self) -> Iterable:
        """Criteria"""
        return [self.earliest_arrival_time, self.fare, self.n_trips]

    @property
    def dep(self) -> int:
        """Departure time in seconds past midnight"""

        try:
            return [
                tst.dts_dep for tst in self.trip.stop_times if self.from_stop == tst.stop
            ][0]
        except IndexError as ex:
            raise Exception(f"No departure time for to_stop: {self.to_stop}.\n"
                            f"Current Leg: {self}. \n Original Error: {ex}")

    @property
    def arr(self) -> int:
        """Arrival time in seconds past midnight"""

        try:
            return [
                tst.dts_arr for tst in self.trip.stop_times if self.to_stop == tst.stop
            ][0]
        except IndexError as ex:
            raise Exception(f"No arrival time for to_stop: {self.to_stop}.\n"
                            f"Current Leg: {self}. \n Original Error: {ex}")

    def is_same_station_transfer(self) -> bool:
        """
        Returns true if the current instance is a transfer leg between stops
        belonging to the same station (i.e. platforms)
        :return:
        """

        return self.from_stop.station == self.to_stop.station

    def is_compatible_before(self, other_leg: Leg) -> bool:
        """
        Check if Leg is allowed before another leg. That is if:

        - It is possible to go from current leg to other leg concerning arrival time
        - Number of trips of current leg differs by > 1, i.e. a different trip,
          or >= 0 when the other_leg is a transfer_leg
        - The accumulated value of a criteria of current leg is larger or equal to the accumulated value of
          the other leg (current leg is instance of this class)
        """
        arrival_time_compatible = (
                other_leg.earliest_arrival_time >= self.earliest_arrival_time
        )

        n_trips_compatible = (
            other_leg.n_trips >= self.n_trips
            if other_leg.is_same_station_transfer()
            else other_leg.n_trips > self.n_trips
        )

        criteria_compatible = np.all(
            np.array([c for c in other_leg.criteria])
            >= np.array([c for c in self.criteria])
        )

        return all([arrival_time_compatible, n_trips_compatible, criteria_compatible])

    def to_dict(self, leg_index: int = None) -> Dict:
        """Leg to readable dictionary"""
        return dict(
            trip_leg_idx=leg_index,
            departure_time=self.dep,
            arrival_time=self.arr,
            from_stop=self.from_stop.name,
            from_station=self.from_stop.station.name,
            to_stop=self.to_stop.name,
            to_station=self.to_stop.station.name,
            trip_hint=self.trip.hint,
            trip_long_name=self.trip.long_name,
            from_platform_code=self.from_stop.platform_code,
            to_platform_code=self.to_stop.platform_code,
            fare=self.fare,
        )


@dataclass(frozen=True)
class Label:
    """Label"""

    earliest_arrival_time: int
    fare: int  # total fare
    trip: Trip | None  # trip to take to obtain travel_time and fare
    from_stop: Stop  # stop to hop-on the trip
    n_trips: int = 0
    infinite: bool = False

    @property
    def criteria(self):
        """Criteria"""
        # TODO this is the multi-criteria part of the label:
        #   most important is arrival time, then fare (where is fare info retrieved from?),
        #   then, at last, number of trips necessary
        return [self.earliest_arrival_time, self.fare, self.n_trips]

    def update(self, earliest_arrival_time=None, fare_addition=None, from_stop=None) -> Label:
        """
        Updates the current label with the provided earliest arrival time and fare_addition.
        Returns the updated label

        :param earliest_arrival_time: new earliest arrival time
        :param fare_addition: fare to add to the current
        :param from_stop: new stop that the label refers to
        :return: updated label
        """
        return copy(
            Label(
                earliest_arrival_time=earliest_arrival_time
                if earliest_arrival_time is not None
                else self.earliest_arrival_time,
                fare=self.fare + fare_addition
                if fare_addition is not None
                else self.fare,
                trip=self.trip,
                from_stop=from_stop if from_stop is not None else self.from_stop,
                n_trips=self.n_trips,
                infinite=self.infinite,
            )
        )

    def update_trip(self, trip: Trip, new_boarding_stop: Stop) -> Label:
        """
        Updates the trip and the boarding stop associated with the current label.
        Returns the updated label.

        :param trip: new trip to update the label with
        :param new_boarding_stop: new boarding stop to update the label with, only if the provided
            trip is different from the current one. Otherwise, the current stop is not updated.
        :return: updated label
        """

        return copy(
            Label(
                earliest_arrival_time=self.earliest_arrival_time,
                fare=self.fare,
                trip=trip,
                from_stop=new_boarding_stop if self.trip != trip else self.from_stop,
                n_trips=self.n_trips + 1 if self.trip != trip else self.n_trips,
                infinite=self.infinite,
            )
        )


@dataclass(frozen=True)
class Bag:
    """
    Bag B(k,p) or route bag B_r
    """

    labels: List[Label] = field(default_factory=list)
    updated: bool = False

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Bag({self.labels}, updated={self.updated})"

    def add(self, label: Label):
        """Add"""
        self.labels.append(label)

    def merge(self, other_bag: Bag) -> Bag:
        """Merge other bag in current bag and return updated Bag"""

        pareto_labels = self.labels + other_bag.labels

        if len(pareto_labels) == 0:
            return Bag(labels=[], updated=False)

        pareto_labels = pareto_set(pareto_labels)
        bag_update = True if pareto_labels != self.labels else False

        return Bag(labels=pareto_labels, updated=bag_update)

    def labels_with_trip(self):
        """All labels with trips, i.e. all labels that are reachable with a trip with given criterion"""
        return [l for l in self.labels if l.trip is not None]

    def earliest_arrival(self) -> int:
        """Earliest arrival"""
        return min([self.labels[i].earliest_arrival_time for i in range(len(self))])


@dataclass(frozen=True)
class Journey:
    """
    Journey from origin to destination specified as Legs
    """

    legs: List[Leg] = field(default_factory=list)

    def __len__(self):
        return len(self.legs)

    def __repr__(self):
        return f"Journey(n_legs={len(self.legs)})"

    def __getitem__(self, index):
        return self.legs[index]

    def __iter__(self):
        return iter(self.legs)

    def __lt__(self, other):
        return self.dep() < other.dep()

    def number_of_trips(self):
        """Return number of distinct trips"""
        trips = set([l.trip for l in self.legs])
        return len(trips)

    def prepend_leg(self, leg: Leg) -> Journey:
        """Add leg to journey"""
        legs = self.legs
        legs.insert(0, leg)
        jrny = Journey(legs=legs)
        return jrny

    def remove_empty_legs(self) -> Journey:
        """Remove all transfer legs"""
        legs = [
            leg
            for leg in self.legs
            if (leg.trip is not None)
               # TODO might want to remove this part: I just want to remove empty legs,
               #   and not transfer legs between parent and child stops
               and (leg.from_stop.station != leg.to_stop.station)
        ]
        jrny = Journey(legs=legs)
        return jrny

    def is_valid(self) -> bool:
        """
        Returns true if the journey is considered valid.
        Notably, a journey is valid if, for each leg, leg k arrival time
        is not greater than leg k+1 departure time
        :return: True if journey is valid, False otherwise
        """

        for index in range(len(self.legs) - 1):
            if self.legs[index].arr > self.legs[index + 1].dep:
                return False
        return True

    def from_stop(self) -> Stop:
        """Origin stop of Journey"""
        return self.legs[0].from_stop

    def to_stop(self) -> Stop:
        """Destination stop of Journey"""
        return self.legs[-1].to_stop

    def fare(self) -> float:
        """Total fare of Journey"""
        return self.legs[-1].fare

    def dep(self) -> int:
        """Departure time"""
        return self.legs[0].dep

    def arr(self) -> int:
        """Arrival time"""
        return self.legs[-1].arr

    def travel_time(self) -> int:
        """Travel time in seconds"""
        return self.arr() - self.dep()

    def dominates(self, jrny: Journey):
        """Dominates other Journey"""
        return (
            True
            if (
                       (self.dep() >= jrny.dep())
                       and (self.arr() <= jrny.arr())
                       and (self.fare() <= jrny.fare())
                       and (self.number_of_trips() <= jrny.number_of_trips())
               )
               and (self != jrny)
            else False
        )

    def print(self, dep_secs=None):
        """Print the given journey to logger info"""

        logger.info("Journey:")

        if len(self) == 0:
            logger.info("No journey available")
            return

        # Print all legs in journey
        first_trip = self.legs[0].trip
        prev_route = first_trip.route_info if first_trip is not None else None
        for leg in self:
            current_trip = leg.trip
            if current_trip is not None:
                hint = current_trip.hint

                if current_trip.route_info != prev_route:
                    logger.info("-- Trip Change --")

                prev_route = current_trip.route_info
            else:
                raise Exception(f"Unhandled leg trip. Value: {current_trip}")

            msg = (
                    str(sec2str(leg.dep))
                    + " "
                    + leg.from_stop.station.name.ljust(20)
                    + " (p. "
                    + str(leg.from_stop.platform_code).rjust(3)
                    + ") TO "
                    + str(sec2str(leg.arr))
                    + " "
                    + leg.to_stop.station.name.ljust(20)
                    + " (p. "
                    + str(leg.to_stop.platform_code).rjust(3)
                    + ") WITH "
                    + str(hint)
            )
            logger.info(msg)

        logger.info(f"Fare: €{self.fare()}")

        msg = f"Duration: {sec2str(self.travel_time())}"
        if dep_secs:
            msg += " ({} from request time {})".format(
                sec2str(self.arr() - dep_secs), sec2str(dep_secs),
            )
        logger.info(msg)
        logger.info("")

    def to_list(self) -> List[Dict]:
        """Convert journey to list of legs as dict"""
        return [leg.to_dict(leg_index=idx) for idx, leg in enumerate(self.legs)]


def pareto_set(labels: List[Label], keep_equal=False):
    """
    Find the pareto-efficient points

    :param labels: list with labels
    :param keep_equal: return also labels with equal criteria
    :return: list with pairwise non-dominating labels
    """

    is_efficient = np.ones(len(labels), dtype=bool)
    labels_criteria = np.array([label.criteria for label in labels])
    for i, label in enumerate(labels_criteria):
        if is_efficient[i]:
            # Keep any point with a lower cost
            # TODO qui vengono effettuati i confronti multi-criterio:
            #   i criteri sono hardcoded dentro la proprietà criteria di structures.Label
            #   bisogna trovare un modo di definirli dinamicamente
            if keep_equal:
                # keep point with all labels equal or one lower
                # Note: list1 < list2 determines if list1 is smaller than list2
                #   based on lexicographic ordering
                #   (i.e. the smaller list is the one with the smaller leftmost element)
                is_efficient[is_efficient] = np.any(
                    labels_criteria[is_efficient] < label, axis=1
                ) + np.all(labels_criteria[is_efficient] == label, axis=1)

            else:
                is_efficient[is_efficient] = np.any(
                    labels_criteria[is_efficient] < label, axis=1
                )

            is_efficient[i] = True  # And keep self

    return list(compress(labels, is_efficient))


_ALGO_OUTPUT_FILENAME = "algo-output"


@dataclass
class AlgorithmOutput(TimetableInfo):
    """
    Class that represents the data output of a Raptor algorithm execution.
    Contains the best journey found by the algorithm, the departure date and time of said journey
    and the path to the directory of the GTFS feed originally used to build the timetable
    provided to the algorithm.
    """

    # Best journey found by the algorithm
    journey: Journey = None

    # string in the format %H:%M:%S
    departure_time: str = None

    @staticmethod
    def read_from_file(filepath: str | bytes | os.PathLike) -> AlgorithmOutput:
        """
        Returns the AlgorithmOutput instance read from the provided folder
        :param filepath: path to an AlgorithmOutput .pcl file
        :return: AlgorithmOutput instance
        """

        def load_joblib() -> AlgorithmOutput:
            logger.debug(f"Loading '{filepath}'")
            with open(Path(filepath), "rb") as handle:
                return joblib.load(handle)

        if not os.path.exists(filepath):
            raise IOError(
                "PyRaptor AlgorithmOutput not found. Run `python pyraptor/query_raptor`"
                " first to generate an algorithm output .pcl file."
            )

        logger.debug("Using cached datastructures")

        algo_output: AlgorithmOutput = load_joblib()

        return algo_output

    @staticmethod
    def save_to_dir(output_dir: str | bytes | os.PathLike,
                    algo_output: AlgorithmOutput):
        """
        Write the algorithm output to the provided directory
        """

        def write_joblib(state, name):
            with open(Path(output_dir, f"{name}.pcl"), "wb") as handle:
                joblib.dump(state, handle)

        logger.info(f"Writing PyRaptor output to {output_dir}")

        mkdir_if_not_exists(output_dir)
        write_joblib(algo_output, _ALGO_OUTPUT_FILENAME)


""" Shared Mobility: Renting Stations """


@attr.s(cmp=False, repr=False)
class RentingStation(Stop):
    """
    Interface representing a Renting Station used
    This class represents a Physical Renting Station used in urban network for shared mobility
    """
    system_id: str = attr.ib(default=None)  # Shared mobility system identifier
    vtype: TransferType = attr.ib(default=None)  # Type of vehicle rentable in the station

    @property
    # @abstractmethod TODO check AttributeError
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        return False

    @property
    # @abstractmethod
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        return False


@attr.s(cmp=False, repr=False)
class RentingStations(Stops, ABC):
    """
    Interface representing a set of renting stations
    """

    system_id: str = attr.ib(default=None)
    system_vtype: TransferType = attr.ib(default=None)

    @property
    def no_source(self) -> List[RentingStation]:
        """ Returns all renting stations with no available vehicles for departure """
        return [s for s in self if not s.valid_source]

    @property
    def no_destination(self) -> List[RentingStation]:
        """ Returns all renting stations with no available docks for arrival """
        return [s for s in self if not s.valid_destination]

    @abstractmethod
    def init_download(self):
        """ Downloads static datas """
        pass

    @abstractmethod
    def update(self):
        """ Update datas using real-time feeds  """
        pass


@attr.s(cmp=False, repr=False)
class PhysicalRentingStation(RentingStation):
    capacity: int = attr.ib(default=0)
    vehicles_available: int = attr.ib(default=0)  # Available vehicles number (real-time)
    docks_available: int = attr.ib(default=0)  # Available docks number (real-time)
    is_installed: bool = attr.ib(default=False)  # Station currently on the street (real-time)
    is_renting: bool = attr.ib(default=False)  # Station renting vehicles (real-time)
    is_returning: bool = attr.ib(default=False)  # Station accepting vehicles returns (real-time)

    @property
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        valid = self.vehicles_available > 0 and \
                self.is_installed and \
                self.is_renting
        return valid

    @property
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        valid = self.vehicles_available < self.capacity and \
                self.docks_available > 0 and \
                self.is_returning
        return valid


@attr.s(cmp=False, repr=False)
class PhysicalRentingStations(RentingStations):

    # New dictionaries types
    set_idx: Dict[str, RentingStation] = dict()
    set_index: Dict[int, RentingStation] = dict()
    last_index: int = 1

    station_info_url: str = attr.ib(default=None)
    station_status_url: str = attr.ib(default=None)

    """ Override superclass methods with stub, subsuming to RentingStations """

    def get(self, stop_id) -> PhysicalRentingStation:
        return super(PhysicalRentingStations, self).get(stop_id)

    def get_by_index(self, stop_index) -> PhysicalRentingStation:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add(self, stop: PhysicalRentingStation) -> PhysicalRentingStation:
        return super(PhysicalRentingStations, self).add(stop)

    """ Override abstract methods """

    def init_download(self):
        """ Downloads static datas """

        stations: List[Dict] = SharedMobilityFeed.open_json(self.station_info_url)['data']['stations']
        for station in stations:
            new_station: Station = Station(id=station['name'], name=station['name'])
            new_: PhysicalRentingStation = PhysicalRentingStation(
                id=station['station_id'], name=station['name'], station=new_station,
                platform_code=-1, index=None, geo=Coordinates(station['lat'], station['lon']),
                system_id=self.system_id, vtype=self.system_vtype, capacity=station['capacity']
            )
            new_station.add_stop(new_)
            self.add(new_)

    def update(self):
        """ Update datas using real-time feeds  """
        status: List[Dict] = SharedMobilityFeed.open_json(self.station_status_url)['data']['stations']
        for state in status:
            station: PhysicalRentingStation = self.get(state['station_id'])
            station.is_installed = state['is_installed']
            station.is_renting = state['is_renting']
            station.is_returning = state['is_returning']
            station.docks_available = state['num_docks_available']
            vname = 'bike' if self.system_vtype == TransferType.Bicycle else 'other'  # TODO check for possible vehicles names
            station.vehicles_available = state[f'num_{vname}s_available']


@attr.s(cmp=False, repr=False)
class GeofenceArea(RentingStation):

    @property
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        return False

    @property
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        return False


@attr.s(cmp=False, repr=False)
class GeofenceAreas(RentingStations):

    # New dictionaries types
    set_idx: Dict[str, GeofenceArea] = dict()
    set_index: Dict[int, GeofenceArea] = dict()
    last_index: int = 1

    geofencing_zones_url: str = attr.ib(default=None)
    free_bike_status_url: str = attr.ib(default=None)

    """ Override superclass methods with stub, subsuming to RentingStations """

    def get(self, stop_id) -> GeofenceArea:
        return super(GeofenceAreas, self).get(stop_id)

    def get_by_index(self, stop_index) -> GeofenceArea:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add(self, stop: GeofenceArea) -> GeofenceArea:
        return super(GeofenceAreas, self).add(stop)

    """ Override abstract methods """

    def init_download(self):
        """ Downloads static datas """
        pass

    def update(self):
        """ Update datas using real-time feeds  """
        pass


@attr.s
class VehicleTransfer(Transfer):
    """
    This class represents a generic Transfer between two
    """
    vtype: TransferType = attr.ib(default=None)

    # TODO can we override Transfer.get_vehicle?
    @staticmethod
    def get_vehicle_transfer(sa: RentingStation, sb: RentingStation,
                             vtype: TransferType, speed: float | None = None) \
            -> Tuple[VehicleTransfer, VehicleTransfer]:
        """ Given two renting stations compute both inbound and outbound vtype transfers
            Transfer time is approximated dividing computed distance by vtype constant speed """
        dist: float = Stop.stop_distance(sa, sb)
        if speed is None:
            speed: float = VEHICLE_SPEED[vtype]
        time: int = int(dist * 3600 / speed)
        return (
            VehicleTransfer(from_stop=sa, to_stop=sb, transfer_time=time, vtype=vtype),
            VehicleTransfer(from_stop=sb, to_stop=sa, transfer_time=time, vtype=vtype)
        )


class VehicleTransfers(Transfers):
    """ This class represent a set of VehicleTransfers  """

    """ Override superclass methods with stub, subsuming to VehicleTransfer """

    def add(self, transfer: VehicleTransfer):
        super(VehicleTransfers, self).add(transfer)

    def with_from_stop(self, from_: RentingStation) -> List[VehicleTransfer]:
        """ Returns all transfers with given departing stop  """
        return super(VehicleTransfers, self).with_from_stop(from_)

    def with_to_stop(self, to: Stop) -> List[VehicleTransfer]:
        """ Returns all transfers with given arrival stop  """
        return super(VehicleTransfers, self).with_to_stop(to)


class SharedMobilityFeed:
    """ This class represent a GBFS feed
        All datas comes from gbfs.json (see https://github.com/NABSA/gbfs/blob/v2.3/gbfs.md#gbfsjson)"""

    def __init__(self, url: str, lang: str = 'it'):
        self.url: str = url  # gbfs.json url
        self.lang: str = lang  # lang of feed
        self.feeds_url: Mapping[str, str] = self._get_feeds_url()  # mapping between feed_name and url
        self.system_id: str = self._get_items_list(feed_name='system_information')['system_id']  # feed sysyem_id
        self.vtype: TransferType = self._get_vtype()
        self.renting_stations: RentingStations = self._get_station()

    @property
    def feeds(self):
        """ Name of feeds """
        return list(self.feeds_url.keys())

    @staticmethod
    def open_json(url: str) -> Dict:
        """ Reads json from url """
        return loads(urlopen(url=url).read())

    def _get_feeds_url(self) -> Mapping[str, str]:
        """ Returns dictionary keyed by feed name and mapped to associated feed url"""
        info: Dict = SharedMobilityFeed.open_json(url=self.url)
        feeds: List[Dict] = info['data'][self.lang]['feeds']  # list of feed items
        feed_url: Dict[str, str] = {feed['name']: feed['url'] for feed in feeds}
        return feed_url

    def _get_items_list(self, feed_name: str):
        """ Returns items list of given feed """
        if feed_name not in self.feeds:
            raise Exception(f"{feed_name} not in {self.feeds}")
        feed = SharedMobilityFeed.open_json(url=self.feeds_url[feed_name])
        datas = feed['data']
        if feed_name != 'system_information':
            items_name = next(
                iter(datas.keys()))  # name of items is only key in datas (e.g. 'stations', 'vehicles', ...)
            return datas[items_name]
        else:
            return datas  # in system_information datas is an items list

    def _get_vtype(self) -> TransferType:
        """ Retrieves vehicle type from associated feeds
            if more than one, raise an exception """
        vtypes = set([vtype['form_factor'] for vtype in self._get_items_list(feed_name='vehicle_types')])
        if len(vtypes) > 1:
            raise Exception(f"Multiple vehicles: {vtypes}")
        else:
            value = next(iter(list(vtypes)))
            return TransferType(value)

    def _get_station(self) -> RentingStations:
        """ Basing on available feeds distinguish from PhysicalRentingStation or GeofanceArea"""
        if 'station_information' not in self.feeds or \
                'station_status' not in self.feeds or \
                len(self._get_items_list(feed_name='station_information')) == 0 or \
                len(self._get_items_list(feed_name='station_status')) == 0:
            if 'geofencing_zones' in self.feeds and 'free_bike_status_url' in self.feeds:
                stations: GeofenceAreas = GeofenceAreas(system_id=self.system_id, system_vtype=self.vtype,
                                                        geofencing_zones_url=self.feeds_url['geofencing_zones'],
                                                        free_bike_status_url=self.feeds_url['free_bike_status_url']
                                                        )
            else:
                raise Exception(f"No compatible stations with feeds {self.feeds}")
        else:
            stations: PhysicalRentingStations = PhysicalRentingStations(system_id=self.system_id,
                                                                        system_vtype=self.vtype,
                                                                        station_info_url=self.feeds_url['station_information'],
                                                                        station_status_url=self.feeds_url['station_status']
                                                                        )
        stations.init_download()
        stations.update()
        return stations
