from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Generic, Dict, Callable

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
            max_rounds: int = -1
    ) -> Mapping[Stop, _BagType]:
        """
        Executes the round-based algorithm and returns the stop-label mappings, keyed by round.

        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param max_rounds: maximum number of rounds to execute.
            The algorithm may stop before if convergence is reached.
            If -1, the algorithm always runs until it converges.
        :return: mapping of the best labels for each stop
        """
        # Initialize data structures and origin stops
        initial_marked_stops = self._initialization(
            from_stops=from_stops,
            dep_secs=dep_secs
        )

        # Get stops immediately reachable with a transfer
        # and add them to the marked stops list
        logger.debug("Computing transfers from origin stops")
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
        logger.debug(f"Added {n_stops_2 - n_stops_1} stops immediately reachable on foot:\n"
                     f"{list(set(immediately_reachable_stops))}")

        # Only cap rounds if max_rounds != -1
        k = 1
        while(len(marked_stops) > 0
              or (k <= max_rounds and max_rounds != -1)):
            logger.info(f"Analyzing possibilities at round {k}")
            logger.debug(f"Marked stops to evaluate: {len(marked_stops)}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.bag_round_stop[k] = copy(self.bag_round_stop[k - 1])

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            marked_route_stops = self._accumulate_routes(marked_stops)
            logger.debug(f"{len(marked_route_stops)} routes to evaluate")

            # Update stop arrival times calculated basing on reachable stops
            trip_marked_stops = self._traverse_routes(
                k=k,
                marked_route_stops=marked_route_stops
            )
            logger.debug(f"Marked {len(trip_marked_stops)} public transport improved stops")

            # Improve arrival times with foot transfers starting from
            # the stops marked at the previous step
            transfer_marked_stops = self._improve_with_transfers(
                k=k,
                marked_stops=trip_marked_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"Marked {len(transfer_marked_stops)} foot transfer improved stops")

            marked_stops = set(trip_marked_stops).union(transfer_marked_stops)

            if len(marked_stops) == 0:
                logger.info("No more stops to evaluate in the following rounds. Terminating...")

                # Since there are no more stops to evaluate, the current round has found
                # reached the best paths, so return it
                return self.bag_round_stop[k]
            else:
                logger.debug(f"{len(marked_stops)} stops to evaluate in the next round")

            k += 1

        return self.bag_round_stop[max_rounds]

    @abstractmethod
    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int) -> List[Stop]:
        """
        Initialization phase of the algorithm.

        This basically corresponds to the first section of the pseudocode described in the
        RAPTOR paper, where bags are initialized and the departure stops are marked.

        :param from_stops: departure stops
        :param dep_secs: departure time in seconds from midnight
        :return: list of marked stops, which should contain the departure stops
        """
        pass

    def _accumulate_routes(self, marked_stops: List[Stop]) -> List[Tuple[Route, Stop]]:
        """
        Generates a list of (R, S) pairs where:
            - S is a marked stop from the provided list, and it is the earliest
                marked stop at which you can board route R
            - R is a route that serves at least one marked stop

        At the end, such list will contain all the routes that serve the provided marked stops,
        paired with the earliest marked stop that they can be boarded on.

        :param marked_stops: list of stops marked by the algorithm
        :return: list of (route, stop) pairs, described above
        """

        marked_route_stops = {}  # i.e. Q in the RAPTOR pseudocode
        for marked_stop in marked_stops:
            routes_serving_stop = self.timetable.routes.get_routes_of_stop(marked_stop)
            for route in routes_serving_stop:
                # Check if new_stop is before existing stop in Q
                current_stop_for_route = marked_route_stops.get(route, None)  # p'
                if (current_stop_for_route is None) or (
                        route.stop_index(current_stop_for_route)
                        > route.stop_index(marked_stop)
                ):
                    marked_route_stops[route] = marked_stop
        marked_route_stops = [(r, p) for r, p in marked_route_stops.items()]

        return marked_route_stops

    @abstractmethod
    def _traverse_routes(
            self,
            k: int,
            marked_route_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        """
        Traverses through all the marked route-stops pairs and updates the labels accordingly.
        For each route-stop pair (R, S), traverses R starting from S and tries to improve
        the labels of all the stops reachable via a trip of R.

        This basically corresponds to the second section of the pseudocode that can be found
        in the RAPTOR paper, where the algorithm tries to improve the label of each marked stop
        by boarding the earliest trip on its associated route.

        :param k: current round
        :param marked_route_stops: list of marked (route, stop) pairs
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
        Considers all the time-independent transfers starting from the marked stops
        and tries to improve all the reachable stops.

        This basically corresponds to the third section of the pseudocode that can
        be found in the RAPTOR paper, where the algorithm tries to improve each
        stop transfer-reachable from the marked stops.

        :param k: current round
        :param marked_stops: currently marked stops,
            i.e. stops for which there was an improvement in the current round
        :param transfers: transfers to use to seek improvements
        :return: list of stops marked because they were improved in some criteria via transfer
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
            timetable: RaptorTimetable | RaptorTimetableSM,
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

        if enable_sm and not isinstance(timetable, RaptorTimetableSM):
            raise ValueError("The provided timetable does not contain the necessary shared mobility data")

        self.sm_config = sm_config

        self.visited_renting_stations = []
        self.no_source = []
        self.no_dest = []
        self.vehicle_transfers = VehicleTransfers()

    def run(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int,
            max_rounds: int = -1
    ) -> Mapping[Stop, _BagType]:
        # Initialize data structures and origin stops
        initial_marked_stops = self._initialization(
            from_stops=from_stops,
            dep_secs=dep_secs
        )

        # Setup shared mob data only if enabled
        # Important to do this BEFORE calculating immediate transfers,
        # else there is a possibility that available shared mob station won't be included
        if self.enable_sm:
            self._initialize_shared_mob(origin_stops=initial_marked_stops)

        # Get stops immediately reachable with a transfer
        # and add them to the marked stops list
        logger.debug("Computing transfers from origin stops")
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
        logger.debug(f"Added {n_stops_2 - n_stops_1} stops immediately reachable on foot:\n"
                     f"{list(set(immediately_reachable_stops))}")

        # Only cap rounds if max_rounds != -1
        k = 1
        while(len(marked_stops) > 0
              or (k <= max_rounds and max_rounds != -1)):
            logger.info(f"Analyzing possibilities at round {k}")
            logger.debug(f"Marked stops to evaluate: {len(marked_stops)}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.bag_round_stop[k] = copy(self.bag_round_stop[k - 1])

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            marked_route_stops = self._accumulate_routes(marked_stops)
            logger.debug(f"{len(marked_route_stops)} routes to evaluate")

            # Update stop arrival times calculated basing on reachable stops
            trip_marked_stops = self._traverse_routes(
                k=k,
                marked_route_stops=marked_route_stops
            )
            logger.debug(f"Marked {len(trip_marked_stops)} public transport improved stops")

            # Improve arrival times with foot transfers starting from
            # the stops marked at the previous step
            transfer_marked_stops = self._improve_with_transfers(
                k=k,
                marked_stops=trip_marked_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"Marked {len(transfer_marked_stops)} foot transfer improved stops")

            if self.enable_sm:
                # Mark stops that were improved with shared mob data
                shared_mob_marked_stops = self._improve_with_sm_transfers(
                    k=k,

                    # Only transfer stops can be passed because shared mob stations
                    # are reachable just by foot transfers
                    marked_stops=transfer_marked_stops
                )
                logger.debug(f"Marked {len(shared_mob_marked_stops)} shared-mob improved stops")

                # Shared mob legs are a special kind of transfer legs
                transfer_marked_stops = set(transfer_marked_stops).union(shared_mob_marked_stops)

            marked_stops = set(trip_marked_stops).union(transfer_marked_stops)

            if len(marked_stops) == 0:
                logger.info("No more stops to evaluate in the following rounds. Terminating...")

                # Since there are no more stops to evaluate, the current round has found
                # reached the best paths, so return it
                return self.bag_round_stop[k]
            else:
                logger.debug(f"{len(marked_stops)} stops to evaluate in the next round")

            k += 1

        return self.bag_round_stop[max_rounds]

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

    def _vehicle_transfer_processor_job(
        self,
        old_rt: List[RentingStation],
        new_rt: List[RentingStation],
        job_id: int | str | uuid.UUID = uuid.uuid4()
    ) -> Callable[[], Iterable[VehicleTransfer]]:
        """
        Returns a function that creates and returns vehicle transfers between
        the provided old renting stations and new discovered renting stations

        :param old_rt: old known renting stations
        :param new_rt: new discovered renting stations
        :param job_id: id to assign to this job.
            Its purpose is only to identify the job in the logger output.
        :return: collection of vehicle transfers between the aforementioned renting stations
        """

        def job():
            def log(msg: str):
                logger.debug(f"[VehicleTransferProcessor {job_id}] {msg}")

            v_transfers: List[VehicleTransfer] = []

            for i, old in zip(range(len(old_rt)), old_rt):
                log(f'Progress: {i * 100 / len(old_rt):0.0f}%')
                for new in new_rt:
                    new_vt, _ = self._create_vehicle_transfer(stop_a=old, stop_b=new)
                    if new_vt is not None:
                        v_transfers.append(new_vt)

            return v_transfers

        return job

    def _improve_with_sm_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop]
    ) -> Sequence[Stop]:
        """
        Tries to improve the criteria values for the provided marked stops
        with shared mob data.

        :param k: current round
        :param marked_stops: currently marked stops,
            i.e. stops for which there was an improvement in the current round
        :return: new list of marked stops,
            i.e. stops for which an improvement in some criteria was made
        """

        # There may be some renting stations in `marked_stops`, from which we could
        # reach a public transport stop with a shared-mobility trip
        # and then use a footpath (a transfer) and walk to a renting station.
        # We filter these renting station in `marked_renting_stations`
        marked_renting_stations: List[RentingStation] = filter_shared_mobility(marked_stops)
        logger.debug(f"{len(marked_renting_stations)} marked renting stations")

        # We create a VehicleTransfer links each old renting station with newfound ones,
        # according to system_id (id of the shared-mob dataset) and availability

        new_v_transfers = VehicleTransfers()

        """ TODO multiprocessor brings to large elapsed time
        n_jobs = 1
        jobs = []
        for i in range(n_jobs):
            visited_renting_station_no = len(self.visited_renting_stations)
            interval_length = math.floor(visited_renting_station_no / n_jobs)
            start = i * interval_length

            if i == (n_jobs - 1):
                # Make sure that all the shared mob stops are processed and
                # that no stop is left out due to rounding errors
                # in calculating interval_length
                end = visited_renting_station_no
            else:
                end = (start + interval_length) - 1  # -1 because the interval_length-th stop belongs to the next round

            job = self._vehicle_transfer_processor_job(
                # +1 because end would not be included
                old_rt=list(itertools.islice(self.visited_renting_stations, start, end + 1)),
                new_rt=marked_renting_stations,
                job_id=f"#{i}"
            )
            jobs.append(job)

        logger.debug(f"Starting {n_jobs} jobs to process vehicle transfers")
        job_results = execute_jobs(jobs=jobs, cpus=n_jobs)

        for vtransfer in itertools.chain.from_iterable(job_results):
            new_v_transfers.add(vtransfer)
        """

        for old in self.visited_renting_stations:
            for new in marked_renting_stations:
                if old != new:
                    new_vt = self._create_vehicle_transfer(stop_a=old, stop_b=new)
                    if new_vt is not None:
                        new_v_transfers.add(new_vt)

        logger.debug(f"New {len(new_v_transfers)} shared-mob transfers created")

        # We can compute transfer-time from these selected renting stations using only filtered transfers
        new_reachable_renting_stations = self._improve_with_transfers(
            k=k,
            marked_stops=self.visited_renting_stations,
            transfers=new_v_transfers
        )
        logger.debug(f"{len(new_reachable_renting_stations)} new renting stations can be reached"
                     f" through shared-mobility")

        # Mark the new renting stations as visited
        self.visited_renting_stations = list(set(self.visited_renting_stations).union(marked_renting_stations))

        # `new_reachable_renting_stations` contains all the improved renting-stations,
        # i.e. that were reachable with a shared-mob transfer.
        # These improvements must be reflected to the public-transport network,
        # so we compute footpaths (Transfers) between improved renting stations
        # and any associated public stop
        sm_marked_stop = self._improve_with_transfers(
            k=k,
            marked_stops=new_reachable_renting_stations,
            transfers=self.timetable.transfers
        )
        logger.debug(f"{len(sm_marked_stop)} new stops marked using shared-mobility")

        return sm_marked_stop

    def _create_vehicle_transfer(self, stop_a: RentingStation, stop_b: RentingStation) -> VehicleTransfer:
        """
        Given two stop, adds an associated outdoor vehicle-transfer
        to depending on the vehicle availability of their associated shared-mob feed.

        If stops have common available multiple vehicles:
            - uses preferred vehicle if present
            - otherwise uses another vehicle (the fastest on average)

        :param stop_a: departing station
        :param stop_b: arrival station
        :return: vehicle transfer between `stop_a` and `stop_b`
        """

        # 1. they are part of same system (feed)
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
        """
        Updates stops availability based on real-time information provided
        by the shared-mob system feeds.
        """

        for feed in self.timetable.shared_mobility_feeds:
            feed.renting_stations.update()

        no_source_: List[List[RentingStation]] = [
            feed.renting_stations.no_source for feed in
            self.timetable.shared_mobility_feeds
        ]
        no_dest_: List[List[RentingStation]] = [
            feed.renting_stations.no_destination for feed in
            self.timetable.shared_mobility_feeds
        ]

        self.no_source: List[RentingStation] = [i for sub in no_source_ for i in sub]  # flatten
        self.no_dest: List[RentingStation] = [i for sub in no_dest_ for i in sub]  # flatten
