from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import List, Tuple, Any

import numpy as np
from loguru import logger

from pyraptor.model.shared_mobility import (
    SharedMobilityFeed,
    RentingStation,
    filter_shared_mobility,
    VehicleTransfers,
    VehicleTransfer,
    VEHICLE_SPEED)
from pyraptor.model.timetable import RaptorTimetable, Route, Stop, TransportType, Transfer


class BaseRaptorAlgorithm(ABC):
    """
    Base class that defines the structure of RAPTOR algorithm implementations
    """

    timetable: RaptorTimetable

    def __init__(self, timetable: RaptorTimetable):
        self.timetable = timetable

    @abstractmethod  # TODO use class generics to defined label type (i.e. Bag, BasicLabel, etc.)
    def run(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> Mapping[int, Mapping[Stop, Any]]:
        """
        Executes the round-based algorithm and returns the stop-label mappings, keyed by round.

        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param rounds: total number of rounds to execute
        :return: stop-label mappings, keyed by round.
        """
        pass

    def _accumulate_routes(self, marked_stops: List[Stop]) -> List[Tuple[Route, Stop]]:
        """Accumulate routes serving marked stops from previous round, i.e. Q"""

        route_marked_stops = {}  # i.e. Q
        for marked_stop in marked_stops:
            routes_serving_stop = self.timetable.routes.get_routes_of_stop(marked_stop)
            for route in routes_serving_stop:
                # Check if new_stop is before existing stop in Q
                current_stop_for_route = route_marked_stops.get(route, None)  # p'
                if (current_stop_for_route is None) or (
                        route.stop_index(current_stop_for_route)
                        > route.stop_index(marked_stop)
                ):
                    route_marked_stops[route] = marked_stop
        route_marked_stops = [(r, p) for r, p in route_marked_stops.items()]

        return route_marked_stops

    @abstractmethod
    def _traverse_routes(
            self,
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        """
        Traverses through all the marked route-stops pairs
        and updates the labels accordingly.

        :param k: current round
        :param route_marked_stops: list of marked (route, stop) pairs
        :return: new list of marked stops,
            i.e. stops for which an improvement in some criteria was made
        """

        pass

    @abstractmethod
    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Iterable[Transfer]
    ) -> List[Stop]:
        """
        Tries to use to improve the criteria values for the provided marked stops
        by applying the provided transfers.

        :param k: current round
        :param marked_stops: currently marked stops,
            i.e. stops for which there was an improvement in the current round
        :param transfers: transfers to evaluate
        :return: new list of marked stops,
            i.e. stops for which an improvement in some criteria was made
        """

        pass

    def _get_transfer(self, stop_from: Stop, stop_to: Stop) -> Transfer:
        """
        Retrieves the transfer from a stop to another stop
        """

        transfers = self.timetable.transfers
        return transfers.stop_to_stop_idx[(stop_from, stop_to)]


@dataclass(frozen=True)
class SharedMobilityConfig:
    feeds: Iterable[SharedMobilityFeed]
    """Shared mobility data to include in the itinerary calculation"""

    preferred_vehicle: TransportType
    """Preferred vehicle type for shared mob transport"""

    enable_car: bool  # TODO chiedere a Seba perché serve
    """If True, car transport is enabled"""


class BaseSMRaptor(BaseRaptorAlgorithm, ABC):
    """
    Base class for RAPTOR implementations that use shared mobility data.

    # TODO explain how shared mob is used in RAPTOR
    """

    enable_sm: bool
    """If True, shared mobility data is included in the itinerary computation"""

    sm_config: SharedMobilityConfig
    """Shared mobility configuration data. Ignored if `enable_sm` is False."""

    visited_renting_stations: List[RentingStation]
    """List containing all the renting stations visited during the computation"""

    no_source: List[RentingStation]
    """List of renting stations not available as source"""

    no_dest: List[RentingStation]
    """List of renting stations not available as destination"""

    vehicle_transfers: VehicleTransfers
    """Collection of vehicle transfers between visited renting stations,
    populated during the computation"""

    def __init__(
            self,
            timetable: RaptorTimetable,
            enable_sm: bool = False,
            sm_config: SharedMobilityConfig = None
    ):
        """
        :param timetable: RAPTOR timetable
        :param enable_sm: if True, shared mobility data is included in the itinerary computation.
            If False, any provided shared mobility data is ignored.
        :param sm_config: shared mobility configuration data. Ignored if `enable_sm` is False.
        """

        super(BaseSMRaptor, self).__init__(timetable=timetable)

        self.enable_sm = enable_sm
        self.sm_config = sm_config

        self.visited_renting_stations = []
        self.no_source = []
        self.no_dest = []
        self.vehicle_transfers = VehicleTransfers()

    def _initialize_shared_mob(self, origin_stops: Sequence[Stop]):
        """
        Executes shared mob data initialization phase.

        :param origin_stops: stops to depart from
        """

        # Download information about shared-mob stops availability
        self._update_availability_info()
        sm_feeds_info = [
            f'{feed.system_id} ({[t.name for t in feed.transport_types]})'
            for feed in self.sm_config.feeds
        ]
        logger.debug(f"Shared mobility feeds: {sm_feeds_info} ")
        logger.debug(f"{len(self.no_source)} shared-mob stops not available as source: {self.no_source} ")
        logger.debug(f"{len(self.no_dest)} shared-mob stops not available as destination: {self.no_dest} ")

        # Mark any renting station to depart from as visited
        for s in origin_stops:
            if (isinstance(s, RentingStation)
                    and s not in self.visited_renting_stations):
                self.visited_renting_stations.append(s)

        logger.debug(f"Starting from {len(origin_stops)} stops "
                     f"({len(self.visited_renting_stations)} are renting stations)")

    def _improve_with_shared_mob(
            self,
            k: int,
            marked_stops: Iterable[Stop]
    ) -> Iterable[Stop]:
        """
        Tries to improve the criteria values for the provided marked stops
        with shared mob data.

        :param k: current round
        :param marked_stops: currently marked stops,
            i.e. stops for which there was an improvement in the current round
        :return: new list of marked stops,
            i.e. stops for which an improvement in some criteria was made
        """

        # Part 4
        # There may be some renting stations in  `marked_stops`:
        # indeed we can reach a public stop with a trip
        # and then use a footpath (a transfer) and walk to a renting station
        #
        # We filter these renting station in `marked_renting_stations`
        marked_renting_stations: List[RentingStation] = filter_shared_mobility(marked_stops)
        logger.debug(f"{len(marked_renting_stations)} renting stations reachable")

        # then we keep only new renting stations
        new_renting_stations: List[RentingStation] = list(
            set(marked_renting_stations).difference(self.visited_renting_stations)
        )
        logger.debug(f"New {len(new_renting_stations)} renting station reachable")

        # Update visited renting stations with the new ones
        self.visited_renting_stations = list(set(self.visited_renting_stations).union(new_renting_stations))

        # We add a VehicleTransfer foreach (old, new) renting station
        # (according to system_id and availability)
        t1 = len(self.vehicle_transfers)  # debugging

        for old in self.visited_renting_stations:
            for new in new_renting_stations:
                self._add_vehicle_transfer(old, new)

        t2 = len(self.vehicle_transfers)  # debugging

        logger.debug(f"New {t2 - t1} vehicle transfers created")

        # We can try to improve the best arrival-time taking advantage of shared-mobility network:
        #
        # We consider only:
        #     - vehicle-transfers
        #     - just Transfers which to_stop is in `marked_renting_stations`
        #
        # List of vehicle-transfers arriving to reachable renting stations
        t_to_new_renting_stations: List[List[VehicleTransfer]] = [
            self.vehicle_transfers.with_to_stop(s) for s in new_renting_stations
        ]
        t_to_new_renting_stations: List[VehicleTransfer] = [
            i for sub in t_to_new_renting_stations for i in sub
        ]  # flatten

        # List of departing renting stations from previous filtered vehicle-transfers
        # NOTE: vehicle transfers are generated only between known renting stations
        dep_to_new_renting_stations: List[RentingStation] = list(
            set([t.from_stop for t in t_to_new_renting_stations])
        )

        # We can get compute transfer-time from these selected renting stations using only filtered transfers
        improved_new_renting_stations = self._improve_with_transfers(
            k=k,
            marked_stops=dep_to_new_renting_stations,
            transfers=t_to_new_renting_stations
        )
        logger.debug(f"{len(improved_new_renting_stations)} transferable renting stations improved")

        # Part 5
        # `renting_stations_from_trip_improved` contains all improved renting-stations
        # These improvements must reflect to public-transport network, so we compute footpaths (Transfers)
        # between improved renting stations and associated transferable public stops
        bag_round_stop, marked_shared_mob_stops = self._improve_with_transfers(
            k=k,
            marked_stops=improved_new_renting_stations,
            transfers=self.timetable.transfers
        )
        logger.debug(f"{len(marked_shared_mob_stops)} using shared-mobility stops upgraded")

        return marked_shared_mob_stops

    def _add_vehicle_transfer(self, stop_a: RentingStation, stop_b: RentingStation):
        """ Given two stop adds associated outdoor vehicle-transfer
            to a vehicles transfers depending on availability and system belongings

            If stops have common available multiple vehicles:
             * uses preferred vehicle if present,
             * otherwise uses an other vehicle TODO more option criteria
        """

        # 1. they are part of same system
        if stop_a.system_id == stop_b.system_id:
            # 2.1. evaluate common transport type
            common_t_types: List[TransportType] = list(
                set(stop_a.transport_types).intersection(stop_b.transport_types)
            )

            # 2.2. remove car transfer if disabled
            if not self.sm_config.enable_car:
                common_t_types = list(set(common_t_types).difference([TransportType.Car]))

            # 2.3. create a vehicle transfer if at least one common transport type found
            if len(common_t_types) > 0:
                # 3.1. if preferred vehicle is present, transfer is generated
                if self.sm_config.preferred_vehicle in common_t_types:
                    best_t_type = self.sm_config.preferred_vehicle
                # 3.2. else the fastest transport type is chosen # TODO different possible criteria
                else:
                    # TODO before it was np.argmin(). why?
                    ind = np.argmax([VEHICLE_SPEED[t_type] for t_type in common_t_types])
                    best_t_type = common_t_types[ind]

                # 4. Create transfer (only A to B is needed)
                t_ab, _ = VehicleTransfer.get_vehicle_transfer(stop_a, stop_b, best_t_type)

                # 5. Validate transfer against real-time availability
                if stop_a not in self.no_source and stop_b not in self.no_dest:
                    self.v_transfers.add(t_ab)

    def _update_availability_info(self):
        """ Updates stops availability based on real-time query
            Also clears all vehicle transfers computed """

        for feed in self.sm_config.feeds:
            feed.renting_stations.update()

        no_source_: List[List[RentingStation]] = [feed.renting_stations.no_source for feed in
                                                  self.sm_config.feeds]
        no_dest_: List[List[RentingStation]] = [feed.renting_stations.no_destination for feed in
                                                self.sm_config.feeds]

        self.no_source: List[RentingStation] = [i for sub in no_source_ for i in sub]  # flatten
        self.no_dest: List[RentingStation] = [i for sub in no_dest_ for i in sub]  # flatten

        self.v_transfers = VehicleTransfers()
