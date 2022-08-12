"""Datatypes"""
from __future__ import annotations

import os
import uuid
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import Enum
from itertools import compress
from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from typing import List, Dict, Tuple, Any, TypeVar, Type
from dataclasses import dataclass, field
from copy import copy

import attr
import joblib
import numpy as np
from loguru import logger

from pyraptor.util import sec2str, mkdir_if_not_exists, LARGE_NUMBER


# TODO split module in different smaller modules (do it after merging into the pyraptor-flexymob branch):
#   - timetable.py with Stop, Trip, Route and Transfer related classes + same_type_and_id() func
#   - criteria.py with Label and Criteria related classes + pareto_set() func
#   - output.py with Leg, Journey and AlgorithmOutput

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
class Timetable(TimetableInfo):
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
class Stop:
    """Stop"""

    id = attr.ib(default=None)
    name = attr.ib(default=None)
    station: Station = attr.ib(default=None)
    platform_code = attr.ib(default=None)
    index = attr.ib(default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, stop):
        return type(self) is type(stop) and self.id == stop.id

    def __repr__(self):
        if self.id == self.name:
            return f"Stop({self.id})"
        return f"Stop({self.name} [{self.id}])"


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

    def get(self, stop_id):
        """Get stop"""
        if stop_id not in self.set_idx:
            raise ValueError(f"Stop ID {stop_id} not present in Stops")
        stop = self.set_idx[stop_id]
        return stop

    def get_by_index(self, stop_index) -> Stop:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add(self, stop):
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

    # TODO remove since it is never set; also remove from Leg and Trip.get_fare()
    #   Substitute with co2 and distance related attributes/getters
    fare: float = attr.ib(default=0.0)

    travelled_distance: float = attr.ib(default=0.0)
    """Distance in km covered by the trip from its beginning"""

    def __hash__(self):
        return hash((self.trip, self.stop_idx))

    def __repr__(self):
        return (
            "TripStopTime(trip_id={hint}{trip_id}, stop_idx={0.stop_idx},"
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
        return f"TripStopTimes(n_tripstoptimes={len(self.set_idx)})"

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
    Walk = 9001
    Bike = 9002
    Car = 9003
    ElectricBike = 9004

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

    def get_description(self) -> str:
        """
        Returns a more verbose description for the value of the current instance.

        :return: transport type description
        """

        transport_descriptions: Dict[TransportType, str] = {
            item: item.name for item in TransportType
        }

        return transport_descriptions[self]


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
        :param transport_type:
        """

        super(TransferRouteInfo, self).__init__(transport_type=transport_type, name="Transfer")


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

    def get_stop_time(self, stop: Stop) -> TripStopTime:
        """Get stop"""
        return self.stop_times[self.stop_times_index[stop]]

    def get_fare(self, depart_stop: Stop) -> float:
        """Get fare from depart_stop"""
        stop_time = self.get_stop_time(depart_stop)
        return 0 if stop_time is None else stop_time.fare


class TransferTrip(Trip):
    """
    Class that represents a transfer trip made between to stops
    """

    def __init__(self,
                 from_stop: Stop,
                 to_stop: Stop,
                 dep_time: int,
                 arr_time: int,
                 transport_type: TransportType):
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
        dep_stop_time = TripStopTime(
            trip=self, stop_idx=0, stop=from_stop, dts_arr=dep_time, dts_dep=dep_time
        )
        self.add_stop_time(dep_stop_time)

        arr_stop_time = TripStopTime(
            trip=self, stop_idx=1, stop=to_stop, dts_arr=arr_time, dts_dep=arr_time
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
class Transfer:
    """Transfer"""

    id = attr.ib(default=None)
    from_stop = attr.ib(default=None)
    to_stop = attr.ib(default=None)

    # Time in seconds that the transfer takes to complete
    transfer_time = attr.ib(default=300)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, trip):
        return same_type_and_id(self, trip)

    def __repr__(self):
        return f"Transfer(from_stop={self.from_stop}, to_stop={self.to_stop}, transfer_time={self.transfer_time})"


class Transfers:
    """
    Class that represents a transfer collection with some additional easier to use access methods.
    """

    def __init__(self):
        self.set_idx: Dict[Any, Transfer] = dict()
        """Dictionary that maps transfer ids with the corresponding transfer instance"""

        self.stop_to_stop_idx: Dict[Tuple[Stop, Stop], Transfer] = dict()
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

    def add(self, transfer: Transfer):
        """Add trip"""
        transfer.id = self.last_id
        self.set_idx[transfer.id] = transfer
        self.stop_to_stop_idx[(transfer.from_stop, transfer.to_stop)] = transfer
        self.last_id += 1


@dataclass
class Leg:
    """Leg"""

    # TODO adapt Leg to new MultiCriteria label
    from_stop: Stop
    to_stop: Stop
    trip: Trip
    criteria: Iterable[Criterion]

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

        # TODO remove?
        # arrival_time_compatible = (
        #         other_leg.earliest_arrival_time >= self.earliest_arrival_time
        # )
#
        # n_trips_compatible = (
        #     other_leg.n_trips >= self.n_trips
        #     if other_leg.is_same_station_transfer()
        #     else other_leg.n_trips > self.n_trips
        # )

        # TODO rationale here is that other_leg (which comes after the current)
        #  can't have smaller values in each criteria.
        #  I.e. the X criteria of leg_{y} must be smaller or equal to the X criteria of leg_{y+1}
        criteria_compatible = np.all(
            np.array([c for c in other_leg.criteria])
            >= np.array([c for c in self.criteria])
        )

        return all([criteria_compatible])

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
            criteria=self.criteria
        )


@dataclass(frozen=True)
class LabelUpdate:
    """
    Class that represents all the necessary data to update a label
    """
    # TODO it would be cool to parameterize this class with a generic _L that indicates
    #   the Label type. This would make best_labels of type Dict[Stop, _L], and would hence
    #   improve type checking and remove the isinstance() check in get_best_stop_criterion().
    #   it would also be easy to alias: e.g. McLabelUpdate = LabelUpdate[MultiCriteriaLabel]

    boarding_stop: Stop
    """Stop at which the trip is boarded"""
    # TODO """If None, the old boarding stop is kept"""

    arrival_stop: Stop
    """Stop at which the trip is hopped off"""

    old_trip: Trip
    """Trip currently used to get from `boarding_stop` to `arrival_stop`"""

    new_trip: Trip
    """New trip to board to get from `boarding_stop` to `arrival_stop`."""
    # TODO """If None, old_trip is used as default value"""

    # TODO make sure, in the algorithm code, that the reference to the best labels does not change
    best_labels: Dict[Stop, BaseLabel]
    """
    Reference to the best labels for each stop, independent from the number of rounds.
    This data is needed by criteria that have a dependency on other labels to calculate their cost.
    (e.g. the distance cost of label x+1 depends on the distance cost of label x)
    """


@dataclass(frozen=True)
class BaseLabel(ABC):
    """
    Abstract class representing the base characteristics that a RAPTOR label
    needs to have. Depending on the algorithm version, there are different types of
    labels.

    Generally speaking, each label contains the trip with which one arrives at the label's associated stop
    with k legs by boarding the trip at the boarding stop. It also contains the criteria with which each
    stop is evaluated by the algorithm.

    Reference (RAPTOR paper):
    https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf
    """

    trip: Trip | None = None
    """Trip to take to arrive at the destination stop at `earliest_arrival_time`"""

    boarding_stop: Stop = None
    """Stop at which the trip is boarded"""

    # TODO add field `to_stop`?

    @abstractmethod
    def update(self, data: LabelUpdate) -> BaseLabel:
        """
        Returns a new label with updated attributes.
        If the provided values are None, the corresponding attributes are not updated.

        :param data: label update data
        :return: new updated label
        """
        pass

    @abstractmethod
    def is_dominating(self, other: BaseLabel) -> bool:
        """
        Returns true if the current label is dominating the provided label,
        meaning that it is not worse in any of the valuation criteria.

        :param other: other label to compare
        :return:
        """
        pass


@dataclass(frozen=True)
class Label(BaseLabel):
    """
    Class that represents a label used in the base RAPTOR version
    described in the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf).
    """

    # TODO generalize this class to have just an ArrivalTimeCriterion? at that point
    #   might as well use a MultiCriteriaLabel with just one criterion
    earliest_arrival_time: int = LARGE_NUMBER
    """Earliest time to get to the destination stop by boarding the current trip"""

    def update(self, data: LabelUpdate) -> Label:
        trip = data.new_trip if self.trip != data.new_trip else self.trip
        boarding_stop = data.boarding_stop if data.boarding_stop is not None else self.boarding_stop

        # Earliest arrival time to the arrival stop on the updated trip
        earliest_arrival_time = trip.get_stop_time(data.arrival_stop).dts_arr

        return Label(
            earliest_arrival_time=earliest_arrival_time,
            boarding_stop=boarding_stop,
            trip=trip
        )

    def is_dominating(self, other: Label) -> bool:
        return self.earliest_arrival_time <= other.earliest_arrival_time

    def __repr__(self) -> str:
        return f"{Label.__name__}(earliest_arrival_time={self.earliest_arrival_time}, " \
               f"trip={self.trip}, boarding_stop={self.boarding_stop})"


@dataclass(frozen=True)
class Criterion(ABC):
    """
    Base class for a RAPTOR label criterion
    """

    name: str
    """Name of the criterion"""

    weight: float
    """Weight used to determine the cost of this criterion"""

    raw_value: float
    """
    Raw value of the criterion, that is before any weight is applied.
    This value maintains is expressed in the original unit of measurement.
    """

    upper_bound: float
    """
    Maximum value allowed for this criterion.
    Such threshold is also used to scale the raw value into the [0,1] range.
    """
    # TODO If the raw value surpasses this threshold, the associated label should be discarded
    #   How to enforce maximum values? set a high cost? add a `upper_bound_surpassed` flag to discard the label?
    #   Or just filter the itineraries in post processing (this I don't like)

    @property
    def cost(self) -> float:
        """
        Returns the weighted cost of this criterion.
        The raw cost is scaled on the range [0, `upper_bound`] and is then
        multiplied by the provided weight.

        :return: weighted scaled cost
        """

        if self.raw_value > self.upper_bound:
            # TODO is this correct way to enforce upper bound?
            #   see above
            return LARGE_NUMBER
        else:
            return self.weight * (self.raw_value / self.upper_bound)  # lower bound is always 0

    def __add__(self, other: object) -> Criterion | float:
        """
        Returns the sum between two criteria, which is:
        - a Criterion instance if the two objects are of type Criterion
            and have the same name and weight;
        - a float, which is the weighted sum of their values,
            if the two objects are of type Criterion but have different names,
            or if the other object is of type float (which is assumed to be a cost);
        - an exception if the two objects are not of type Criterion or float
            or if they have the same name but differ in the other characteristics
            (weight, upper bound)
        :param other: second addend of the sum operation
        :return: Criterion or float instance
        """

        if isinstance(other, Criterion):
            if other.name == self.name:
                if other.weight != self.weight or other.upper_bound != self.upper_bound:
                    raise Exception(f"Cannot add criteria with the same name but different characteristics"
                                    f"(weight, upper bound).\n"
                                    f"First addend: {self} - Second addend: {other}")
                else:
                    return Criterion(
                        name=self.name,
                        weight=self.weight,
                        raw_value=(self.raw_value + other.raw_value),
                        upper_bound=self.upper_bound
                    )
            else:
                return self.cost + other.cost
        elif isinstance(other, float):
            return self.cost + other
        else:
            raise TypeError(f"Cannot add type {Criterion.__name__} with type {other.__class__.__name__}.\n"
                            f"Second addend: {other}")

    def __radd__(self, other) -> Criterion | float:
        return self.__add__(other)

    def __lt__(self, other):
        return self.__cmp__(other) == -1

    def __le__(self, other):
        cmp = self.__cmp__(other)
        return cmp <= 0

    def __gt__(self, other):
        return self.__cmp__(other) == 1

    def __ge__(self, other):
        cmp = self.__cmp__(other)
        return cmp >= 0

    def __cmp__(self, other) -> int:
        # 0 if equal, -1 if < other, +1 if > other
        if self.cost < other.cost:
            return -1
        elif self.cost == other.cost:
            return 0
        else:
            return 1

    def __float__(self) -> float:
        return self.cost

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name};weight={self.weight};" \
               f"raw_value={self.raw_value};upper_bound={self.upper_bound})"

    @abstractmethod
    def update(self, data: LabelUpdate) -> Criterion:
        pass


# Generic var for Criterion subclasses
_C = TypeVar('_C', bound=Criterion)


def _get_best_stop_criterion(criterion_class: Type[_C], stop: Stop, best_labels: Dict[Stop, BaseLabel]) -> _C:
    """
    Returns the instance of the specified type of criterion, which is retrieved from the
    best label associated with the provided stop.
    An exception is raised if the criterion instance couldn't be found.

    :param criterion_class: type of the criterion to retrieve
    :param stop: stop to retrieve the best criterion for
    :param best_labels: collection that pairs the best labels with their associated stop
    :return: instance of the specified criterion type
    """

    # Criteria can be retrieved only from MultiCriteriaLabel instances
    stop_lbl = best_labels[stop]
    if isinstance(stop_lbl, MultiCriteriaLabel):
        criterion = next(
            filter(lambda c: isinstance(c, criterion_class), stop_lbl.criteria),
            None
        )
        if criterion is None:
            raise ValueError(f"The provided best labels do not include "
                             f"a criterion of type {criterion_class.__name__}")

        return criterion
    else:
        raise TypeError("The provided best labels are not multi-criteria labels")


class DistanceCriterion(Criterion):
    """
    Class that represents and handles calculations for the distance criterion.
    The value represents the total number of km travelled.
    """

    def __str__(self):
        return f"Travelled Distance: {self.raw_value} [Km]"

    def update(self, data: LabelUpdate) -> DistanceCriterion:
        arrival_distance = self._get_total_arrival_distance(data=data)

        return DistanceCriterion(
            name=self.name,
            weight=self.weight,
            raw_value=arrival_distance,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_distance(data: LabelUpdate) -> float:
        """
        Returns the updated distance (in km) for the criterion instance based on the
        new provided boarding and arrival stop. Such value represents the total travelled
        distance between the origin stop and the provided arrival stop.

        :param data: update data
        :return: distance in km
        """

        # The formula is the following:
        # total_distance(arrival) = total_distance(boarding) + [trip_distance(arrival) - trip_distance(boarding)]
        # where trip_distance(x) is the cumulative distance of the trip T that leads to x, starting from
        # the beginning of T
        same_trip_distance = _get_same_trip_distance(
            trip=data.new_trip,
            from_stop=data.boarding_stop,
            to_stop=data.arrival_stop
        )

        # Extract the total distance of the previous stop (boarding stop) in the journey
        # from its distance criterion instance
        prev_stop_dist_criterion = _get_best_stop_criterion(
            criterion_class=DistanceCriterion,
            stop=data.boarding_stop,
            best_labels=data.best_labels
        )

        return prev_stop_dist_criterion.raw_value + same_trip_distance


def _get_same_trip_distance(trip: Trip, from_stop: Stop, to_stop: Stop) -> float:
    """
    Returns the distance between the two provided stops in the specified trip.

    :param trip: trip covering the two stops
    :param from_stop: first stop
    :param to_stop: second stop
    :return: distance between the first and second stop, in km
    """

    from_stop_time = trip.get_stop_time(from_stop)
    to_stop_time = trip.get_stop_time(to_stop)

    return to_stop_time.travelled_distance - from_stop_time.travelled_distance


class EmissionsCriterion(Criterion):
    """
    Class that represents and handles calculations for the co2 emissions criterion
    """

    def __str__(self):
        return f"Total Emissions: {self.raw_value} [CO2 grams / passenger Km]"

    def update(self, data: LabelUpdate) -> EmissionsCriterion:
        arrival_emissions = self._get_total_arrival_emissions(data=data)

        return EmissionsCriterion(
            name=self.name,
            weight=self.weight,
            raw_value=arrival_emissions,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_emissions(data: LabelUpdate) -> float:
        """
        Returns the updated total emissions (in co2 grams / passenger km) for
        this criterion instance, based on the new provided boarding and arrival stop.
        Such value represents the total emissions between the origin stop
        and the provided arrival stop.

        :param data: update data
        :return: emissions in co2 grams / passenger km
        """

        same_trip_distance = _get_same_trip_distance(
            trip=data.new_trip,
            from_stop=data.boarding_stop,
            to_stop=data.arrival_stop
        )

        co2_multiplier = EmissionsCriterion.get_emission_multiplier(
            transport_type=data.new_trip.route_info.transport_type
        )
        same_trip_emissions = same_trip_distance * co2_multiplier

        prev_stop_emissions_crit = _get_best_stop_criterion(
            criterion_class=EmissionsCriterion,
            stop=data.boarding_stop,
            best_labels=data.best_labels
        )

        return prev_stop_emissions_crit.raw_value + same_trip_emissions

    @staticmethod
    def get_emission_multiplier(transport_type: TransportType) -> float:
        """
        Returns the emission multiplier for the provided transport type,
        expressed in `co2 grams / passenger km`
        :return:
        """

        # Sources (values expressed in co2 grams/passenger km):
        # - https://ourworldindata.org/travel-carbon-footprint
        # - Ferry https://www.thrustcarbon.com/insights/how-to-calculate-emissions-from-a-ferry-journey
        # - Electric Bike https://www.bosch-ebike.com/us/service/sustainability
        co2_grams_per_passenger_km: Dict[TransportType, float] = {
            TransportType.Walk: 0,
            TransportType.Bike: 0,
            TransportType.ElectricBike: 14,
            TransportType.Car: (192 + 172) / 2,  # Avg between petrol and diesel

            # It is assumed that all rail vehicles have the same impact,
            # since, even if different sources point to different numbers,
            # the average emissions per passenger km between the different
            # rail transports are approximately equal
            TransportType.Rail: 41,
            TransportType.LightRail: 35,

            # Monorail and cable cars are all assumed to have
            # the same impact of light rail transport, since they are
            # all usually electrically powered (couldn't find specific data)
            TransportType.Monorail: 35,
            TransportType.CableTram: 35,
            TransportType.Funicular: 35,
            TransportType.AerialLift: 35,
            TransportType.Metro: 31,

            # Since trolleybus are very similar to trams, except they have wheels,
            # it is assumed that their emissions are equal. I couldn't find
            # recent data about trolleybus co2 emissions per passenger/km
            TransportType.TrolleyBus: 35,
            TransportType.Bus: 105,
        }

        return co2_grams_per_passenger_km[transport_type]


class ArrivalTimeCriterion(Criterion):
    """
    Class that represents and handles calculations for the arrival time criterion
    """

    def __str__(self):
        return f"Arrival Time: {sec2str(scnds=int(self.raw_value))}"

    def update(self, data: LabelUpdate) -> ArrivalTimeCriterion:
        new_arrival_time = data.new_trip.get_stop_time(data.arrival_stop).dts_arr

        return ArrivalTimeCriterion(
            name=self.name,
            weight=self.weight,
            raw_value=new_arrival_time,
            upper_bound=self.upper_bound
        )


class TransfersNumberCriterion(Criterion):
    """
    Class that represents and handles calculations for the number of transfers criterion
    """

    def __str__(self):
        return f"Total Transfers: {self.raw_value}"

    def update(self, data: LabelUpdate) -> TransfersNumberCriterion:
        # The leg counter is updated only if the new trip isn't a transfer
        # between stops of the same station
        add_new_leg = data.new_trip != data.old_trip
        if add_new_leg and isinstance(data.new_trip, TransferTrip):
            # Transfer trips refer to movements between just two stops
            from_stop = data.new_trip.stop_times[0].stop
            to_stop = data.new_trip.stop_times[1].stop

            if from_stop.station == to_stop.station:
                add_new_leg = False

        return TransfersNumberCriterion(
            name=self.name,
            weight=self.weight,
            raw_value=self.raw_value if not add_new_leg else self.raw_value + 1,
            upper_bound=self.upper_bound
        )


class CriteriaProvider:
    """
    Class that provides parsing functionality for the criteria configuration file.

    Such file is a JSON format where keys represent criteria names and
    values represent criteria weights.
    """

    def __init__(self, criteria_config_path: str | bytes | os.PathLike):
        """
        :param criteria_config_path: path to the criteria configuration file,
            containing the weights of each supported criteria
        """

        self._criteria_config_path: str | bytes | os.PathLike = criteria_config_path
        self._criteria_config: Dict[str, Dict[str, float]] = {}

    def get_criteria(self, defaults: Dict[Type[Criterion], float] = None) -> Sequence[Criterion]:
        """
        Returns a collection of criteria objects that are based on the name and weights provided
        in the configuration file.

        :param: dictionary containing the default values for each criterion type.
            The default value for an unspecified criterion is `0`.
        :return: criteria objects
        """

        # Load criteria only if necessary
        if len(self._criteria_config) == 0:
            self._load_config()

        if defaults is None:
            defaults = {}

        # Pair criteria names with their class (and constructor)
        criterion_classes = {
            "distance": DistanceCriterion,
            "arrival_time": ArrivalTimeCriterion,
            "co2": EmissionsCriterion,
            "transfers": TransfersNumberCriterion,
        }

        criteria = []
        for name, criteria_info in self._criteria_config.items():
            weight = criteria_info["weight"]
            upper_bound = criteria_info["max"]

            c_class = criterion_classes[name]
            default_val = defaults.get(c_class, 0)

            c = c_class(
                name=name,
                weight=weight,
                raw_value=default_val,
                upper_bound=upper_bound
            )

            criteria.append(c)

        return criteria

    def _load_config(self):
        with open(self._criteria_config_path) as f:
            self._criteria_config = json.load(f)


@dataclass(frozen=True)
class MultiCriteriaLabel(BaseLabel):  # TODO is it MultiCriteria or MultiCriterion?
    """
    Class that represents a multi-criteria label.

    The concept this is class is modeled after is that of the multi-label in the
    `McRAPTOR` section of the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf)
    """

    criteria: Sequence[Criterion] = attr.ib(default=list)
    """Collection of criteria used to compare labels"""

    @property
    def total_cost(self) -> float:
        """
        Returns the total cost assigned to the label, which corresponds to
        the weighted sum of its criteria.
        :return: float instance representing the total cost
        """

        if len(self.criteria) == 0:
            raise Exception("No criteria to calculate cost with")

        return sum(self.criteria, start=0.0)

    @property
    def earliest_arrival_time(self) -> int:
        """
        Returns the earliest arrival time associated to this label

        :return: arrival time in seconds past the midnight of the departure day
        """

        arrival_time_crit = next(
            filter(lambda c: isinstance(c, ArrivalTimeCriterion), self.criteria),
            None
        )

        if arrival_time_crit is None:
            raise ValueError(f"No {ArrivalTimeCriterion.__name__} is defined for this label")
        else:
            return int(arrival_time_crit.raw_value)

    def update(self, data: LabelUpdate) -> MultiCriteriaLabel:
        if len(self.criteria) == 0:
            raise Exception("Trying to update an instance with no criteria set")

        updated_criteria = []
        for c in self.criteria:
            updated_c = c.update(data=data)
            updated_criteria.append(updated_c)

        updated_trip = data.new_trip if data.new_trip is not None else data.old_trip

        return MultiCriteriaLabel(
            # TODO is this the correct way of updating stop? original way
            boarding_stop=data.boarding_stop if self.trip != updated_trip else self.boarding_stop,
            # TODO way 2
            # boarding_stop=data.boarding_stop if data.boarding_stop is not None else self.boarding_stop,

            trip=data.new_trip if data.new_trip is not None else data.old_trip,
            criteria=updated_criteria
        )

    def is_dominating(self, other: MultiCriteriaLabel) -> bool:
        return self.total_cost <= other.total_cost


@dataclass(frozen=True)
class Bag:
    """
    Bag B(k,p) or route bag B_r
    """

    labels: List[MultiCriteriaLabel] = field(default_factory=list)
    updated: bool = False

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Bag({self.labels}, updated={self.updated})"

    def add(self, label: MultiCriteriaLabel):
        """Add"""
        self.labels.append(label)

        if len(self.labels) > 1:
            logger.debug(f"Adding label to Bag with {len(self.labels)} labels")

    def merge(self, other_bag: Bag) -> Bag:
        """Merge other bag in current bag and return updated Bag"""

        pareto_labels = self.labels + other_bag.labels

        if len(pareto_labels) == 0:
            return Bag(labels=[], updated=False)

        pareto_labels = pareto_set(pareto_labels)
        bag_update = True if pareto_labels != self.labels else False

        if len(pareto_labels) > 1:
            logger.debug(f"Creating Bag with {len(pareto_labels)} labels")

        return Bag(labels=pareto_labels, updated=bag_update)

    def get_best_label(self) -> MultiCriteriaLabel:
        """
        Returns the label with the best (lowest) cost in the bag
        :return:
        """

        if len(self.labels) == 0:
            raise Exception("There are no labels to retrieve the best from")

        by_cost_asc = list(sorted(self.labels, key=lambda l: l.total_cost))
        return by_cost_asc[0]

    def labels_with_trip(self):
        """All labels with trips, i.e. all labels that are reachable with a trip with given criterion"""
        return [lbl for lbl in self.labels if lbl.trip is not None]


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
        trips = set([lbl.trip for lbl in self.legs])
        return len(trips)

    def prepend_leg(self, leg: Leg) -> Journey:
        """Add leg to journey"""
        legs = self.legs
        legs.insert(0, leg)
        jrny = Journey(legs=legs)
        return jrny

    def remove_empty_and_same_station_legs(self) -> Journey:
        """
        Removes all empty legs (where the trip is not set)
        and transfer legs between stops of the same station.

        :return: updated journey
        """

        legs = [
            leg
            for leg in self.legs
            if (leg.trip is not None)
               # TODO might want to remove this part: I just want to remove empty legs,
               #   and not transfer legs between parent and child stops
               #   Also remember that removing this changes test outcomes
               and (leg.from_stop.station != leg.to_stop.station)
        ]
        jrny = Journey(legs=legs)

        return jrny

    def is_valid(self) -> bool:
        """
        Returns true if the journey is considered valid.
        Notably, a journey is valid if, for each leg, leg k arrival time
        is not greater than leg k+1 departure time.

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

    def dep(self) -> int:
        """Departure time"""
        return self.legs[0].dep

    def arr(self) -> int:
        """Arrival time"""
        return self.legs[-1].arr

    def travel_time(self) -> int:
        """Travel time in seconds"""
        return self.arr() - self.dep()

    def total_cost(self) -> float:
        """
        Returns the total cost of the journey
        :return:
        """

        # journey cost is the total cost at its last leg
        last_leg = self.legs[-1]

        return sum(last_leg.criteria, start=0.0)

    def dominates(self, jrny: Journey):
        """Dominates other Journey"""
        return (
            True
            if (
                       self.total_cost() <= jrny.total_cost()
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
                raise Exception(f"Leg trip cannot be {None}. Value: {current_trip}")

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

        # TODO make it print all the final criteria values
        # logger.info(f"Fare: €{self.fare()}")

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


def pareto_set(labels: List[MultiCriteriaLabel], keep_equal=False):
    """
    Find the pareto-efficient points

    :param labels: list with labels
    :param keep_equal: return also labels with equal criteria
    :return: list with pairwise non-dominating labels
    """

    is_efficient = np.ones(len(labels), dtype=bool)
    label_costs = np.array([label.total_cost for label in labels])
    for i, cost in enumerate(label_costs):
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
                    label_costs[is_efficient] < cost, axis=0
                ) + np.all(label_costs[is_efficient] == cost, axis=0)

            else:
                is_efficient[is_efficient] = np.any(
                    label_costs[is_efficient] < cost, axis=0
                )

            is_efficient[i] = True  # And keep self

    return list(compress(labels, is_efficient))


@dataclass
class AlgorithmOutput(TimetableInfo):
    """
    Class that represents the data output of a Raptor algorithm execution.
    Contains the best journey found by the algorithm, the departure date and time of said journey
    and the path to the directory of the GTFS feed originally used to build the timetable
    provided to the algorithm.
    """

    _DEFAULT_FILENAME = "algo-output"

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
        write_joblib(algo_output, AlgorithmOutput._DEFAULT_FILENAME)
