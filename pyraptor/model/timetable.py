from __future__ import annotations

import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter
from typing import Tuple, Dict, Any, List, TypeVar, Generic

import attr
import numpy as np
from geopy.distance import geodesic
from loguru import logger

from pyraptor.util import DEFAULT_TRANSFER_COST, MEAN_FOOT_SPEED, MEAN_BIKE_SPEED, MEAN_ELECTRIC_BIKE_SPEED, \
    MEAN_CAR_SPEED


def same_type_and_id(first, second):
    """
    Returns true if `first` and `second` have the same type and `id` attribute

    :param first: first object to compare
    :param second: second object to compare
    """

    return type(first) is type(second) and first.id == second.id


@dataclass
class TimetableInfo:
    original_gtfs_dir: str | bytes | os.PathLike = None
    """
    Path to the directory of the GTFS feed originally
    used to generate the current Timetable instance
    """

    date: str = None
    """
    Date that the timetable refers to.

    Format: `YYYYMMDD`, which is equal to %Y%m%d
    """


@dataclass
class RaptorTimetable(TimetableInfo):
    """Timetable data"""

    stations: Stations = None
    stops: Stops = None
    trips: Trips = None
    trip_stop_times: TripStopTimes = None
    routes: Routes = None
    transfers: Transfers = None

    def counts(self) -> None:
        """Prints timetable counts"""

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
    def to_list(self) -> [float, float]:
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


_Stop = TypeVar("_Stop", bound=Stop)


class Stops(Generic[_Stop]):
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

    def get_stop(self, stop_id) -> _Stop:
        """Get stop"""
        if stop_id not in self.set_idx:
            raise ValueError(f"Stop ID {stop_id} not present in Stops")
        stop: _Stop = self.set_idx[stop_id]
        return stop

    def get_by_index(self, stop_index) -> _Stop:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add_stop(self, stop: _Stop) -> _Stop:
        """Add stop"""
        if stop.id in self.set_idx:
            stop = self.set_idx[stop.id]
        else:
            stop.index = self.last_index
            self.set_idx[stop.id] = stop
            self.set_index[stop.index] = stop
            self.last_index += 1
        return stop


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
    """
    Class that represents the arrival and departure times for some stop in some trip
    """

    trip: Trip = attr.ib(default=attr.NOTHING)
    stop: Stop = attr.ib(default=attr.NOTHING)

    stop_idx: int = attr.ib(default=attr.NOTHING)
    """Sequence number of the stop in the trip"""

    dts_arr: int = attr.ib(default=attr.NOTHING)
    """Time of arrival in seconds past midnight"""

    dts_dep: int = attr.ib(default=attr.NOTHING)
    """Time of departure in seconds past midnight"""

    travelled_distance: float = attr.ib(default=0.0)
    """Distance in km covered by the trip from its beginning"""

    def __hash__(self):
        return hash((self.trip, self.stop_idx))

    def __repr__(self):
        return (
            "TripStopTime(trip_id={hint}{trip_id}, stop_idx={0.stop_idx},"
            " stop_id={0.stop.id}, dts_arr={0.dts_arr}, dts_dep={0.dts_dep},"
            "travelled_distance={0.travelled_distance})"
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
        return f"{TripStopTimes.__name__}(n_tripstoptimes={len(self.set_idx)})"

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
            if (dep_secs_min <= tst.dts_dep <= dep_secs_max) and tst.stop in stops
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


class TransportType(Enum):
    Walk = -1

    # Starting from 9000 is completely arbitrary, it just needs
    # to be different from the parent TransportType enum
    Car = 9001
    Bike = 9002
    ElectricBike = 9003

    # The following values match the integer codes defined for the `route_type` field at
    # https://developers.google.com/transit/gtfs/reference#routestxt
    LightRail = 0
    Metro = 1
    Rail = 2
    Bus = 3
    Ferry = 4
    CableTram = 5
    AerialLift = 6
    Funicular = 7
    TrolleyBus = 11
    Monorail = 12

    def __str__(self):
        tt = TransportType
        to_str = {
            tt.Walk: 'Walk',
            tt.Car: 'Car sharing',
            tt.Bike: 'Bike sharing',
            tt.ElectricBike: 'Bike sharing (electric bike)',
            tt.LightRail: 'Light Rail',
            tt.Metro: 'Metro',
            tt.Rail: 'Rail',
            tt.Bus: 'Bus',
            tt.Ferry: 'Ferry',
            tt.CableTram: 'Cable Tram',
            tt.AerialLift: 'Aerail Lift',
            tt.Funicular: 'Funicular',
            tt.TrolleyBus: 'Trolley Bus',
            tt.Monorail: 'Monorail'
        }
        return to_str[self]

    def get_description(self) -> str:
        """
        Returns a more verbose description for the value of the current instance.

        :return: transport type description
        """

        transport_descriptions: Dict[TransportType, str] = {
            item: item.name for item in TransportType
        }

        return transport_descriptions[self]


PUBLIC_TRANSPORT_TYPES: List[TransportType] = [TransportType.LightRail, TransportType.Metro, TransportType.Rail,
                                               TransportType.Bike, TransportType.Ferry, TransportType.CableTram,
                                               TransportType.AerialLift, TransportType.Funicular,
                                               TransportType.TrolleyBus, TransportType.Monorail, TransportType.Bus]

SHARED_MOBILITY_TYPES: List[TransportType] = [TransportType.Bike, TransportType.ElectricBike, TransportType.Car]


@dataclass(frozen=True)
class RouteInfo:
    transport_type: TransportType = None
    name: str = None

    def __str__(self):
        return f"Transport: {self.transport_type.get_description()} | Route Name: {self.name}"

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, RouteInfo):
            return other.transport_type == self.transport_type and other.name == self.name
        else:
            raise Exception(f"Cannot compare {RouteInfo.__name__} with {type(other)}")


class TransferRouteInfo(RouteInfo):
    """
    Class that represents information about a transfer route
    """

    def __init__(self, transport_type: TransportType):
        """
        :param transport_type: transport that the transfer is carried out with
        """

        super(TransferRouteInfo, self).__init__(
            transport_type=transport_type,
            name=f"Transfer ({transport_type.name})"
        )


@attr.s(repr=False, cmp=False, init=False)
class Trip:
    """
    Class that represents a Trip, which is a sequence of consecutive stops
    """

    def __init__(self,
                 id_: Any = None,
                 long_name: str = None,
                 route_info: RouteInfo = None,
                 hint: str = None):
        """
        :param id_: id of the trip
        :param long_name: long name of the trip
        :param route_info: information about the route that the trip belongs to
        :param hint: additional information about the trip.
            Defaults to `str(route_info)`.
        """

        self.id = id_
        self.long_name: str = long_name
        self.route_info: RouteInfo = route_info

        self.hint: str = str(route_info) if hint is None else hint
        self.stop_times: List[TripStopTime] = []
        self.stop_times_index: Dict[Stop, int] = {}

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return "Trip(id={trip_id}, hint={hint}, stop_times={stop_times})".format(
            trip_id=self.id,
            hint=self.hint if self.hint is not None else self.id,
            stop_times=len(self.stop_times),
        )

    def __getitem__(self, n: int) -> TripStopTime:
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

    def get_stop_time(self, stop: Stop) -> TripStopTime:
        """Get stop"""
        return self.stop_times[self.stop_times_index[stop]]


TRANSPORT_TYPE_SPEEDS: Mapping[TransportType, float] = {
    TransportType.Walk: MEAN_FOOT_SPEED,
    TransportType.Bike: MEAN_BIKE_SPEED,
    TransportType.ElectricBike: MEAN_ELECTRIC_BIKE_SPEED,
    TransportType.Car: MEAN_CAR_SPEED,
}


class TransferTrip(Trip):
    """
    Class that represents a transfer trip made between to stops
    """

    def __init__(
            self,
            from_stop: Stop,
            to_stop: Stop,
            dep_time: int,
            arr_time: int,
            transport_type: TransportType
    ):
        """
        :param from_stop: stop that the transfer starts from
        :param to_stop: stop that the transfer ends at
        :param dep_time: departure time in seconds past midnight
        :param arr_time: arrival time in seconds past midnight
        :param transport_type: type of the transport that the transfer is carried out with
        """

        transfer_route = TransferRouteInfo(transport_type=transport_type)
        super(TransferTrip, self).__init__(id_=f"Transfer Trip - {uuid.uuid4()}",
                                           long_name=f"Transfer from {from_stop.name} to {to_stop.name}",
                                           route_info=transfer_route)

        # Add stop times for both origin and end stops
        travelling_time = arr_time - dep_time

        if transport_type not in TRANSPORT_TYPE_SPEEDS.keys():
            raise ValueError(f"Unhandled transport type `{transport_type}`: average speed is not defined")

        travelled_distance = (travelling_time / 3600) * TRANSPORT_TYPE_SPEEDS[transport_type]
        dep_stop_time = TripStopTime(
            trip=self, stop_idx=0, stop=from_stop, dts_arr=dep_time, dts_dep=dep_time,
            travelled_distance=travelled_distance
        )
        self.add_stop_time(dep_stop_time)

        arr_stop_time = TripStopTime(
            trip=self, stop_idx=1, stop=to_stop, dts_arr=arr_time, dts_dep=arr_time,
            travelled_distance=travelled_distance
        )
        self.add_stop_time(arr_stop_time)


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
class Transfer(Generic[_Stop]):
    """Transfer"""

    id: str | None = attr.ib(default=None)
    from_stop: _Stop | None = attr.ib(default=None)
    to_stop: _Stop | None = attr.ib(default=None)

    transfer_time: int = attr.ib(default=DEFAULT_TRANSFER_COST)
    """Time in seconds that the transfer takes to complete"""

    transport_type: TransportType = attr.ib(default=TransportType.Walk)
    """Transport type that the transfer is carried out with"""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return (f"Transfer(from_stop={self.from_stop}, "
                f"to_stop={self.to_stop}, "
                f"transfer_time={self.transfer_time})")

    @staticmethod
    def get_transfer(sa: _Stop, sb: _Stop) -> Tuple[Transfer, Transfer]:
        """
        Given two stops compute both inbound and outbound transfers
        Transfer time is approximated dividing computed distance by a constant speed
        """

        dist: float = Stop.stop_distance(sa, sb)
        time: int = int(dist * 3600 / MEAN_FOOT_SPEED)
        return (
            Transfer(from_stop=sa, to_stop=sb, transfer_time=time),
            Transfer(from_stop=sb, to_stop=sa, transfer_time=time)
        )


_Transfer = TypeVar("_Transfer", bound=Transfer)


class Transfers(Generic[_Transfer]):
    """
    Class that represents a transfer collection with some additional easier to use access methods.
    """

    def __init__(self):
        self.set_idx: Dict[Any, _Transfer] = dict()
        """Dictionary that maps transfer ids with the corresponding transfer instance"""

        self.stop_to_stop_idx: Dict[Tuple[Stop, Stop], _Transfer] = dict()
        """Dictionary that maps (from_stop, to_stop) pairs with the corresponding transfer instance"""

        self.last_id: int = 1
        """
        Field used to store the id of the last added transfer.
        It is incremented by one after a transfer is added.
        """

    def __repr__(self):
        return f"Transfers(n_transfers={len(self.set_idx)})"

    def __getitem__(self, transfer_id):
        return self.set_idx[transfer_id]

    def __len__(self):
        return len(self.set_idx)

    def __iter__(self):
        return iter(self.set_idx.values())

    def add(self, transfer: _Transfer):
        """Add trip"""
        transfer.id = self.last_id
        self.set_idx[transfer.id] = transfer
        self.stop_to_stop_idx[(transfer.from_stop, transfer.to_stop)] = transfer
        self.last_id += 1

    def with_from_stop(self, from_: Stop) -> List[_Transfer]:
        """ Returns all transfers with given departing stop  """
        return [
            self.stop_to_stop_idx[(f, t)] for f, t in self.stop_to_stop_idx.keys() if f == from_
        ]

    def with_to_stop(self, to: Stop) -> List[_Transfer]:
        """ Returns all transfers with given arrival stop  """
        return [
            self.stop_to_stop_idx[(f, t)] for f, t in self.stop_to_stop_idx.keys() if t == to
        ]

    def with_stop(self, s) -> List[_Transfer]:
        """ Returns all transfers with given stop as departing or arrival  """
        return self.with_from_stop(s) + self.with_to_stop(s)
