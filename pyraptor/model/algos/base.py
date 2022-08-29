from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Generic, Dict

import numpy as np
from loguru import logger

from pyraptor.model.criteria import BaseLabel
from pyraptor.model.shared_mobility import (
    RentingStation,
    filter_shared_mobility,
    VehicleTransfers,
    VehicleTransfer,
    TRANSPORT_TYPE_SPEEDS, RaptorTimetableSM)
from pyraptor.model.timetable import RaptorTimetable, Route, Stop, TransportType, Transfer

_BagType = TypeVar("_BagType")
"""Type of the bag of labels assigned to each stop by the RAPTOR algorithm"""

_LabelType = TypeVar("_LabelType", bound=BaseLabel)
"""Type of the label used by the RAPTOR algorithm"""


class BaseRaptorAlgorithm(ABC, Generic[_BagType, _LabelType]):
    """
    Base class that defines the structure of RAPTOR algorithm implementations.

    It also provides some improvements over the original algorithm
    described in the RAPTOR paper:
    - transfers from the origin stops are evaluated immediately to widen
        the set of reachable stops before the first round is executed
    """

    timetable: RaptorTimetable
    """RAPTOR timetable, containing stops, routes, trips and transfers data"""

    bag_round_stop: Dict[int, Dict[Stop, _BagType]]
    """Dictionary that keeps the stop-bag associations 
    created in each round of the algorithm"""

    best_bag: Dict[Stop, _LabelType]
    """Dictionary that keeps the best stop-label associations 
    created by the algorithm, independently by the round number"""

    def __init__(self, timetable: RaptorTimetable):
        self.timetable = timetable
        self.bag_round_stop = {}
        self.best_bag = {}

    @abstractmethod
    def run(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int,
            rounds: int
    ) -> Mapping[Stop, _BagType]:
        """
        Executes the round-based algorithm and returns the stop-label mappings, keyed by round.

        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param rounds: total number of rounds to execute
        :return: mapping of the best labels for each stop
        """

        initial_marked_stops = self._initialization(
            from_stops=from_stops,
            dep_secs=dep_secs,
            rounds=rounds
        )

        # Get stops immediately reachable with a transfer
        # and add them to the marked stops list
        logger.debug("Computing immediate transfers from origin stops")
        immediately_reachable_stops = self._improve_with_transfers(
            k=0,  # still initialization round
            marked_stops=initial_marked_stops,
            transfers=self.timetable.transfers
        )

        n_stops_1 = len(initial_marked_stops)  # debugging

        marked_stops = list(
            set(initial_marked_stops).union(immediately_reachable_stops)
        )

        n_stops_2 = len(marked_stops)  # debugging
        logger.debug(f"Added {n_stops_2 - n_stops_1} immediate stops")

        # Run rounds
        for k in range(1, rounds + 1):
            logger.info(f"Analyzing possibilities at round {k}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.bag_round_stop[k] = copy(self.bag_round_stop[k - 1])

            # Get list of stops to evaluate in the process
            logger.debug(f"Stops to evaluate: {len(marked_stops)}")

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            route_marked_stops = self._accumulate_routes(marked_stops)

            # Update stop arrival times calculated basing on reachable stops
            marked_trip_stops = self._traverse_routes(
                k, route_marked_stops
            )
            logger.debug(f"{len(marked_trip_stops)} reachable stops added")

            # Add footpath transfers and update
            marked_transfer_stops = self._improve_with_transfers(
                k=k,
                marked_stops=marked_trip_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"{len(marked_transfer_stops)} transferable stops added")

            marked_stops = set(marked_trip_stops).union(marked_transfer_stops)

            logger.debug(f"{len(marked_stops)} stops to evaluate in next round")

        return self.bag_round_stop[rounds]

    @abstractmethod
    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> List[Stop]:
        """
        Initialization phase of the algorithm.

        This basically corresponds to the first section of the pseudocode described in the
        RAPTOR paper, where bags are initialized and the departure stops are marked.

        :param from_stops: departure stops
        :param dep_secs: departure time in seconds from midnight
        :param rounds: number of rounds to execute
        :return: list of marked stops, which should contain the departure stops
        """
        pass

    def _accumulate_routes(self, marked_stops: List[Stop]) -> List[Tuple[Route, Stop]]:
        """
        # TODO document better
        Accumulate routes serving marked stops from previous round, i.e. Q
        """

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
        Traverses through all the marked route-stops pairs and updates the labels accordingly.

        This basically corresponds to the second section of the pseudocode that can be found
        in the RAPTOR paper, where the algorithm tries to improve the label of each marked stop
        by boarding the earliest trip on its associated route.

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
        Tries to use to improve the labels of each marked stop by factoring in
        the use of (foot) transfer paths.

        This basically corresponds to the third section of the pseudocode that can
        be found in the RAPTOR paper, where the algorithm tries to improve each label
        by seeing if it is better to use a transfer than boarding the associated trip.

        :param k: current round
        :param marked_stops: currently marked stops,
            i.e. stops for which there was an improvement in the current round
        :param transfers: transfers to evaluate
        :return: new list of marked stops,
            i.e. stops for which an improvement in some criteria was made
        """

        pass


@dataclass(frozen=True)
class SharedMobilityConfig:

    preferred_vehicle: TransportType
    """Preferred vehicle type for shared mob transport"""

    enable_car: bool
    """If True, car transport is enabled"""


class BaseSharedMobRaptor(BaseRaptorAlgorithm[_BagType, _LabelType], ABC):
    """
    Base class for RAPTOR implementations that use shared mobility data.

    It improves on basic RAPTOR implementations by making it possible to include
    shared mobility, real-time data in the itinerary computation.

    # TODO explain how shared mob is used in RAPTOR
    """

    timetable: RaptorTimetableSM
    """RAPTOR timetable, containing stops, routes, trips and transfers data,
    as well as shared mobility transfer data"""

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
            timetable: RaptorTimetableSM,
            enable_sm: bool = False,
            sm_config: SharedMobilityConfig = None
    ):
        """
        :param timetable: RAPTOR timetable
        :param enable_sm: if True, shared mobility data is included in the itinerary computation.
            If False, any provided shared mobility data is ignored.
        :param sm_config: shared mobility configuration data. Ignored if `enable_sm` is False.
        """

        super(BaseSharedMobRaptor, self).__init__(timetable=timetable)

        self.enable_sm = enable_sm
        self.sm_config = sm_config

        self.visited_renting_stations = []
        self.no_source = []
        self.no_dest = []
        self.vehicle_transfers = VehicleTransfers()

    def run(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int,
            rounds: int
    ) -> Mapping[Stop, _BagType]:
        initial_marked_stops = self._initialization(
            from_stops=from_stops,
            dep_secs=dep_secs,
            rounds=rounds
        )

        # Setup shared mob data only if enabled
        # Important to do this BEFORE calculating immediate transfers,
        # else there is a possibility that available shared mob station won't be included
        if self.enable_sm:
            self._initialize_shared_mob(origin_stops=initial_marked_stops)

        # Get stops immediately reachable with a transfer
        # and add them to the marked stops list
        logger.debug("Computing immediate transfers from origin stops")
        immediately_reachable_stops = self._improve_with_transfers(
            k=0,  # still initialization round
            marked_stops=initial_marked_stops,
            transfers=self.timetable.transfers
        )

        # Add any immediately reachable via transfer
        if self.enable_sm:
            self._update_visited_renting_stations(stops=immediately_reachable_stops)

        n_stops_1 = len(initial_marked_stops)  # debugging

        marked_stops = list(
            set(initial_marked_stops).union(immediately_reachable_stops)
        )

        n_stops_2 = len(marked_stops)  # debugging
        logger.debug(f"Added {n_stops_2 - n_stops_1} immediate stops:\n"
                     f"{immediately_reachable_stops}")

        # Run rounds
        for k in range(1, rounds + 1):
            logger.info(f"Analyzing possibilities at round {k}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.bag_round_stop[k] = copy(self.bag_round_stop[k - 1])

            # Get list of stops to evaluate in the process
            logger.debug(f"Stops to evaluate: {len(marked_stops)}")

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            route_marked_stops = self._accumulate_routes(marked_stops)

            # Update stop arrival times calculated basing on reachable stops
            marked_trip_stops = self._traverse_routes(
                k=k, 
                route_marked_stops=route_marked_stops
            )
            logger.debug(f"{len(marked_trip_stops)} reachable stops added")

            # Add footpath transfers and update
            marked_transfer_stops = self._improve_with_transfers(
                k=k,
                marked_stops=marked_trip_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"{len(marked_transfer_stops)} transferable stops added")

            if self.enable_sm:
                # Mark stops that were improved with shared mob data
                shared_mob_marked_stops = self._improve_with_sm_transfers(
                    k=k,

                    # Only transfer stops can be passed because shared mob stations
                    # are reachable just by foot transfers
                    marked_stops=marked_transfer_stops
                )
                logger.debug(f"{len(shared_mob_marked_stops)} shared mob transferable stops added")

                # Shared mob legs are a special kind of transfer legs
                marked_transfer_stops = set(marked_transfer_stops).union(shared_mob_marked_stops)

            marked_stops = set(marked_trip_stops).union(marked_transfer_stops)

            logger.debug(f"{len(marked_stops)} stops to evaluate in next round")

        return self.bag_round_stop[rounds]

    def _initialize_shared_mob(self, origin_stops: Sequence[Stop]):
        """
        Executes shared mobility data initialization phase.

        :param origin_stops: stops to depart from
        """

        # Download information about shared-mob stops availability
        self._update_availability_info()
        sm_feeds_info = [
            f'{feed.system_id} ({[t.name for t in feed.transport_types]})'
            for feed in self.timetable.shared_mobility_feeds
        ]
        logger.debug(f"Shared mobility feeds: {sm_feeds_info} ")
        logger.debug(f"{len(self.no_source)} shared-mob stops not available as source: {self.no_source} ")
        logger.debug(f"{len(self.no_dest)} shared-mob stops not available as destination: {self.no_dest} ")

        # Mark any renting station to depart from as visited
        self._update_visited_renting_stations(stops=origin_stops)

        logger.debug(f"Starting from {len(origin_stops)} stops "
                     f"({len(self.visited_renting_stations)} are renting stations)")

    def _update_visited_renting_stations(self, stops: Iterable[Stop]):
        for s in stops:
            if (isinstance(s, RentingStation)
                    and s not in self.visited_renting_stations):
                self.visited_renting_stations.append(s)

    def _improve_with_sm_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop]
    ) -> Sequence[Stop]:
        """
        Tries to improve the criteria values for the provided marked stops
        with shared mob data.

        # TODO explain better and refactor commenting below

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

        # We add a VehicleTransfer foreach (old, new) renting station
        # (according to system_id and availability)

        new_v_transfers = VehicleTransfers()

        for old in self.visited_renting_stations:
            for new in marked_renting_stations:
                if old != new:
                    new_vt = self._add_vehicle_transfer(stop_a=old, stop_b=new)
                    if new_vt is not None:
                        new_v_transfers.add(new_vt)

        logger.debug(f"New {len(new_v_transfers)} vehicle transfers created")

        # We can get compute transfer-time from these selected renting stations using only filtered transfers
        improved_new_renting_stations = self._improve_with_transfers(
            k=k,
            marked_stops=self.visited_renting_stations,
            transfers=new_v_transfers
        )
        logger.debug(f"{len(improved_new_renting_stations)} transferable renting stations improved")

        # Update visited renting stations with the new ones
        self.visited_renting_stations = list(set(self.visited_renting_stations).union(marked_renting_stations))

        # Part 5
        # `renting_stations_from_trip_improved` contains all improved renting-stations
        # These improvements must reflect to public-transport network, so we compute footpaths (Transfers)
        # between improved renting stations and associated transferable public stops
        marked_shared_mob_stops = self._improve_with_transfers(
            k=k,
            marked_stops=improved_new_renting_stations,
            transfers=self.timetable.transfers
        )
        logger.debug(f"{len(marked_shared_mob_stops)} using shared-mobility stops upgraded")

        return marked_shared_mob_stops

    def _add_vehicle_transfer(self, stop_a: RentingStation, stop_b: RentingStation) -> VehicleTransfer:
        """ Given two stop adds associated outdoor vehicle-transfer
            to a vehicles transfers depending on availability and system belongings

            If stops have common available multiple vehicles:
             * uses preferred vehicle if present,
             * otherwise uses another vehicle (the fastest on average)
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
                # 3.2. else the fastest transport type is chosen
                else:
                    ind = np.argmax([TRANSPORT_TYPE_SPEEDS[t_type] for t_type in common_t_types])
                    best_t_type = common_t_types[ind]

                # 4. Create transfer (only A to B is needed)
                t_ab, _ = VehicleTransfer.get_vehicle_transfer(stop_a, stop_b, best_t_type)

                # 5. Validate transfer against real-time availability
                if stop_a not in self.no_source and stop_b not in self.no_dest:
                    return t_ab

    def _update_availability_info(self):
        """ Updates stops availability based on real-time query
            Also clears all vehicle transfers computed """

        for feed in self.timetable.shared_mobility_feeds:
            feed.renting_stations.update()

        no_source_: List[List[RentingStation]] = [feed.renting_stations.no_source for feed in
                                                  self.timetable.shared_mobility_feeds]
        no_dest_: List[List[RentingStation]] = [feed.renting_stations.no_destination for feed in
                                                self.timetable.shared_mobility_feeds]

        self.no_source: List[RentingStation] = [i for sub in no_source_ for i in sub]  # flatten
        self.no_dest: List[RentingStation] = [i for sub in no_dest_ for i in sub]  # flatten
