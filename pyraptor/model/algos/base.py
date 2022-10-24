from __future__ import annotations

from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Iterable, Mapping, Sequence, MutableMapping
from copy import copy
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Generic, Dict, Set

import numpy as np
from loguru import logger

from pyraptor.model.criteria import BaseLabel, Bag, LabelUpdate, SingleLabelBag
from pyraptor.model.shared_mobility import (
    RentingStation,
    filter_shared_mobility,
    VehicleTransfers,
    VehicleTransfer,
    TRANSPORT_TYPE_SPEEDS,
    RaptorTimetableSM
)
from pyraptor.model.timetable import (
    RaptorTimetable,
    Route,
    Stop,
    TransportType,
    Transfer,
    TransferTrip,
    Trip,
    Transfers
)

_LabelType = TypeVar("_LabelType", bound=BaseLabel)
"""Type of the labels used by the RAPTOR algorithm"""

_BagType = TypeVar("_BagType", bound=Bag)
"""Type of the label bags used by the RAPTOR algorithm"""


@dataclass(frozen=True)
class SharedMobilityConfig:
    preferred_vehicle: TransportType
    """Preferred vehicle type for shared mob transport"""

    enable_car: bool
    """If True, car transport is enabled"""


class BaseRaptor(ABC, Generic[_LabelType, _BagType]):
    """
    Base class that defines the structure of RAPTOR algorithm implementations.

    When compared to the original RAPTOR discussed in the homonymous paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf),
    it provides some additional functionalities, namely:

    - transfers from the origin stops are evaluated immediately to widen
        the set of reachable stops before the first round is executed
    - shared-mobility stations, based on real-time GBFS feeds, can be evaluated as part
        of the itinerary search (option has to be enabled)
    - a heuristic that improves computation times in most multi-criteria scenarios
        can be enabled (Forward Dependencies Heuristic)
    """

    timetable: RaptorTimetable | RaptorTimetableSM
    """RAPTOR timetable, containing stops, routes, trips and transfers data. 
    It could also contain shared mobility data, which will be used by the algorithm
    only if the related option is enabled"""

    round_stop_bags: Dict[int, Dict[Stop, _BagType]]
    """Dictionary that keeps the stop-bag associations 
    created in each round of the algorithm"""

    enable_fwd_deps_heuristic: bool
    """If True, the algorithm uses the forward dependencies heuristic"""

    stop_forward_dependencies: Dict[Stop, Set[Stop]]
    """Dictionary that pairs each stop with the list of stops that depend on it. 
    For example, in a journey x1, x2, ..., xn, stop x2 depends on
    stop x1 because it comes later, hence the name 'forward' dependency"""

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
            enable_fwd_deps_heuristic: bool = True,
            enable_sm: bool = False,
            sm_config: SharedMobilityConfig = None
    ):
        """
        :param timetable: RAPTOR timetable
        :param enable_fwd_deps_heuristic: if True, the algorithm makes use
            of the forward dependencies heuristic
        :param enable_sm: if True, shared mobility data is included in the itinerary computation.
            If False, any provided shared mobility data is ignored.
        :param sm_config: shared mobility configuration data. Ignored if `enable_sm` is False.
        """

        self.timetable = timetable
        self.round_stop_bags = {}
        self.enable_fwd_deps_heuristic = enable_fwd_deps_heuristic

        # Shared Mobility attrs initialization
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
        """
        Executes the RAPTOR algorithm and returns a map that pairs each stop with
        a collection of labels, which contain the information about the best journey(s)
        to reach said stops.

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

        # Setup shared mob data only if enabled
        # Important to do this BEFORE calculating immediate transfers,
        # else there is a possibility that available shared mob station won't be included
        if self.enable_sm:
            self._initialize_shared_mob(origin_stops=list(initial_marked_stops))

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
        while (
                max_rounds == -1
                or (k <= max_rounds and max_rounds != -1)
        ):
            logger.info(f"Analyzing possibilities at round {k}")
            logger.debug(f"Marked stops to evaluate: {len(marked_stops)}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.round_stop_bags[k] = copy(self.round_stop_bags[k - 1])

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            marked_route_stops = self._accumulate_routes(marked_stops)
            logger.debug(f"{len(marked_route_stops)} routes marked")

            # Update stop arrival times calculated basing on reachable stops
            trip_marked_stops = self._traverse_routes(
                k=k,
                marked_route_stops=marked_route_stops
            )
            logger.debug(f"{len(trip_marked_stops)} stops marked with public transport trips")

            # Improve arrival times with foot transfers starting from
            # the stops marked at the previous step
            transfer_marked_stops = self._improve_with_transfers(
                k=k,
                marked_stops=trip_marked_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"{len(transfer_marked_stops)} stops marked with foot transfers")

            if self.enable_sm:
                # Mark stops that were improved with shared mob data
                shared_mob_marked_stops = self._improve_with_sm_transfers(
                    k=k,

                    # Only transfer stops can be passed because shared mob stations
                    # are reachable just by foot transfers
                    marked_stops=transfer_marked_stops
                )
                logger.debug(f"{len(shared_mob_marked_stops)} stops improved with shared-mobility")

                # Shared mob legs are a special kind of transfer legs
                transfer_marked_stops = set(transfer_marked_stops).union(shared_mob_marked_stops)

            marked_stops = set(trip_marked_stops).union(transfer_marked_stops)

            if len(marked_stops) == 0:
                logger.info("No more stops to evaluate in the following rounds. Terminating...")

                # Since there are no more stops to evaluate, the current round has found
                # reached the best paths, so return it
                return self.round_stop_bags[k]
            else:
                logger.debug(f"{len(marked_stops)} stops to evaluate in the next round")

            k += 1

        return self.round_stop_bags[max_rounds]

    @abstractmethod
    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int) -> Set[Stop]:
        """
        Initialization phase of the algorithm.

        This basically corresponds to the first section of the pseudocode described in the
        RAPTOR paper, where bags are initialized and the departure stops are marked.

        :param from_stops: departure stops
        :param dep_secs: departure time in seconds from midnight
        :return: list of marked stops, which should contain the departure stops
        """
        pass

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

    def _update_availability_info(self):
        """
        Updates shared-mob stations availability based on real-time information provided
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

    def _update_visited_renting_stations(self, stops: Iterable[Stop]):
        for s in stops:
            if (isinstance(s, RentingStation)
                    and s not in self.visited_renting_stations):
                self.visited_renting_stations.append(s)

    def _accumulate_routes(self, marked_stops: Set[Stop]) -> List[Tuple[Route, Stop]]:
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
    ) -> Set[Stop]:  # TODO why not return Set?
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
    ) -> Set[Stop]:
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

    def _improve_with_sm_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop]
    ) -> Set[Stop]:
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

        # Create a VehicleTransfer that links each old renting station with newfound ones,
        # according to system_id (id of the shared-mob dataset) and availability
        new_v_transfers = VehicleTransfers()
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

    def _update_stop(
            self,
            k: int,
            stop_to_update: Stop,
            update_with: List[_LabelType],
            currently_marked_stops: Set[Stop]
    ):
        """
        TODO docs
        :param k:
        :param stop_to_update:
        :param update_with:
        :param currently_marked_stops:
        :return:
        """

        def update_fwd_deps_set(s: Stop):
            # Remove the current stop from forward dependencies of the old boarding stop
            old_bag = self.round_stop_bags[k][s]
            for old_lbl in old_bag.labels:
                if old_lbl.boarding_stop is None:
                    # First initialization of labels has boarding stop == None
                    # In that case, skip
                    continue

                self.stop_forward_dependencies[old_lbl.boarding_stop].remove(s)

            # Add the current stop to the forward dependencies of the new boarding stop
            # Note: it is important to remove first and add second, because if the order
            #   is inverted, the newly added stop might be deleted immediately
            for lbl in updated_bag.labels:
                self.stop_forward_dependencies[lbl.boarding_stop].add(s)

        updated_bag = self.round_stop_bags[k][stop_to_update].merge(with_labels=update_with)

        # Mark the stop if bag is updated and update the best label(s) for that stop
        # Updated bag means that the current stop brought some improvements
        if updated_bag.improved:
            # Only run heuristic if enabled
            if self.enable_fwd_deps_heuristic:
                update_fwd_deps_set(stop_to_update)

                updated_fwd_dependencies = self._update_forward_dependencies(
                    k=k,
                    updated_stop=stop_to_update,
                    updated_bag=updated_bag
                )

                # Now update all the bags of the stops dependent on the current one
                # with what was previously calculated
                for dep_stop, dep_labels in updated_fwd_dependencies.items():
                    old_bag = self.round_stop_bags[k][dep_stop]
                    self.round_stop_bags[k][dep_stop] = old_bag.merge(with_labels=dep_labels.labels)

            # Update the bag of the current stop_to_update
            self.round_stop_bags[k][stop_to_update] = updated_bag
            currently_marked_stops.add(stop_to_update)

    def _update_forward_dependencies(
            self,
            k: int,
            updated_stop: Stop,
            updated_bag: _BagType
    ) -> MutableMapping[Stop, _BagType]:
        """
        # TODO sistemare docs
        This step is needed because Weighted MC RAPTOR can't guarantee that temporal
        dependencies between labels won't be broken by future updates to the labels:
        since time is a criteria just like the others, it can be 0-weighted, which means
        it won't be considered when dominating labels are identified. This also means
        that a dominating label for a stop P at round X can worsen the arrival time for
        said stop P at round X-1, therefore potentially breaking the time relationships
        between stop and all the stops that depend on it (i.e. are visited later in the journey).

        Example:
        Given some stops X,Y and Z, at the end of round K you can arrive at Y from X at time T
        and at Z from Y at time T+1.
        At some point during the next round, then, it is found that there is a path
        that improves the generalized cost of going from X to Y, but said path worsens the
        arrival time at Y, from X, to T+2. This would mean that you could arrive at Z, from Y,
        at time T+1, but the best (minimum cost) path to Y has an arrival time of T+2.
        This is an absurd.

        It is therefore necessary, when some label A is updated, to recursively update
        all the labels that depend on A, i.e. that come later than A in an itinerary that contains A.
        This forward dependencies update for A, so called because said labels, as aforementioned,
        are a later part of the journey with respect to A, cannot however always be performed:
        it might be that, at some point during the dependency updates, no available trips are
        found after the new arrival time. This means that this process must resolve in
        one of two ways:
        1. If, at some point, it is found that there are no available paths, all the
            dependency updates must be rolled back, and the original update must be discarded;
        2. If the dependency updates are all successful, apply them and keep the original update.

        :param k: current round
        :param updated_stop: stop whose bag (labels) was updated and whose forward dependencies
            must be updated
        :param updated_bag: bag containing the updated labels
        :return: tuple containing:
            - a boolean that indicates if the update process was successful
            - the updated forward dependencies if the update process was successful,
                else an empty dictionary
        """

        updated_fwd_deps: MutableMapping[Stop, Bag[_LabelType]] = ChainMap()

        for updated_label in updated_bag.labels:
            fwd_dependent_stops: Iterable[Stop] = self.stop_forward_dependencies[updated_stop]

            for fwd_dep_stop in fwd_dependent_stops:
                # It is certain that a dependent stop has at least one label assigned to it:
                # since it is dependent, it means that it has been reached by the algorithm
                updated_fwd_dep_labels = []
                old_fwd_dep_bag = self.round_stop_bags[k][fwd_dep_stop]
                for fwd_dep_label in old_fwd_dep_bag.labels:
                    if fwd_dep_label.trip is None:
                        # TODO why can this happen? - maybe with proper testing it can be pin-pointed
                        continue

                    # Consider the trip of the fwd dependent label: it is certain that
                    # such trip contains both the updated stop (which acts as the boarding stop)
                    # and the dependent stop (which acts as the arrival stop).
                    if isinstance(fwd_dep_label.trip, TransferTrip):
                        old_transfer_trip = fwd_dep_label.trip
                        transfer_time = (
                                old_transfer_trip.stop_times[1].dts_arr - old_transfer_trip.stop_times[0].dts_dep
                        )

                        new_arrival_time = updated_label.arrival_time + transfer_time
                        new_transfer_trip = TransferTrip(
                            from_stop=updated_stop,
                            to_stop=fwd_dep_stop,
                            dep_time=updated_label.arrival_time,
                            arr_time=new_arrival_time,
                            transport_type=old_transfer_trip.route_info.transport_type
                        )

                        update_data = LabelUpdate(
                            boarding_stop=updated_stop,
                            arrival_stop=fwd_dep_stop,
                            new_trip=new_transfer_trip,
                            boarding_stop_label=updated_label
                        )
                    else:
                        current_route = fwd_dep_label.trip.route_info.route
                        new_earliest_trip = current_route.earliest_trip(
                            dts_arr=updated_label.arrival_time,
                            stop=updated_stop
                        )

                        if new_earliest_trip is None:
                            # No trip available means that the label can't be updated:
                            # continue the loop onto the next fwd dep
                            continue
                        else:
                            update_data = LabelUpdate(
                                boarding_stop=updated_stop,
                                arrival_stop=fwd_dep_stop,
                                new_trip=new_earliest_trip,
                                boarding_stop_label=updated_label
                            )

                    updated_fwd_dep_labels.append(fwd_dep_label.update(data=update_data))

                # Merge the old bag with the updated labels of the forward dependencies
                # This ensures that only the best labels before
                # and after the fwd deps update are kept
                updated_fwd_dep_bag = old_fwd_dep_bag.merge(with_labels=updated_fwd_dep_labels)
                updated_fwd_deps[fwd_dep_stop] = old_fwd_dep_bag.merge(with_labels=updated_fwd_dep_labels)

                # Only continue if there was any improvement
                if updated_fwd_dep_bag.improved:
                    rec_updated_fwd_deps = self._update_forward_dependencies(
                        k=k,
                        updated_stop=fwd_dep_stop,
                        updated_bag=updated_fwd_dep_bag
                    )

                    # It is noted that the merge between the current and the recursively created
                    # mappings cannot lead to conflict (i.e. a key is present in more than one mapping),
                    # because RAPTOR itineraries (paths) do not contain cycles
                    updated_fwd_deps = ChainMap(updated_fwd_deps, rec_updated_fwd_deps)

        # Return the computed fwd updates
        return updated_fwd_deps


_SingleLabelBagType = TypeVar("_SingleLabelBagType", bound=SingleLabelBag)


class SingleCriterionRaptor(BaseRaptor[_LabelType, _SingleLabelBagType], ABC):
    """
    Abstract class that serves as the base for any RAPTOR implementation
    that optimizes just a single criterion.
    """

    def _traverse_routes(
            self,
            k: int,
            marked_route_stops: List[Tuple[Route, Stop]],
    ) -> Set[Stop]:
        logger.debug(f"Traverse routes for round {k}")

        new_marked_stops = set()

        # Traverse each route, starting from the earliest marked stop in arrival order
        for marked_route, earliest_marked_stop in marked_route_stops:
            # This represents the best trip that can be taken to traverse the route
            current_trip: Trip | None = None

            # Iterate over all stops after current stop within the current route
            arr_stop_idx = marked_route.stop_index(earliest_marked_stop)
            remaining_stops_in_route = marked_route.stops[arr_stop_idx:]

            # The first boarding stop is the first marked stop of the route
            boarding_stop = earliest_marked_stop

            for arr_stop_idx, arrival_stop in enumerate(remaining_stops_in_route):
                # t != _|_ in the pseudocode
                if current_trip is not None:
                    # Try to improve the journey to the current arrival stop
                    # by boarding the current trip
                    self._try_to_improve_arrival_stop(
                        k=k,
                        boarding_stop=boarding_stop,
                        arrival_stop=arrival_stop,
                        trip=current_trip,
                        marked_stops=new_marked_stops
                    )

                # Always try to board the earliest possible trip
                # The question that is asked is the following: we arrived at stop arrival_stop
                #   in the previous round (k-1), what is the earliest trip that can be boarded
                #   at that stop, after arriving?
                prev_round_arr_time = self.round_stop_bags[k - 1][
                    arrival_stop
                ].get_label().arrival_time
                earliest_trip = marked_route.earliest_trip(
                    prev_round_arr_time, arrival_stop
                )

                if earliest_trip is not None:
                    # If the trip is different from the previous one, we board the trip
                    #   at arrival_stop, else the trip is still boarded at the old boarding stop.
                    # This basically means that the boarding_stop will always be the earliest one,
                    #   in terms of arrival order in the current route
                    if earliest_trip != current_trip:
                        boarding_stop = arrival_stop

                    current_trip = earliest_trip

        return new_marked_stops

    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Transfers
    ) -> Set[Stop]:
        new_marked_stops: Set[Stop] = set()

        # Add in transfers from the transfers table
        for dep_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)
            other_station_stops = [
                t.to_stop for t in transfers if t.from_stop == dep_stop
            ]

            time_sofar = self.round_stop_bags[k][dep_stop].get_label().arrival_time
            for arrival_stop in other_station_stops:
                # Create transfer trip
                transfer = transfers.stop_to_stop_idx[(dep_stop, arrival_stop)]
                arrival_time_with_transfer = time_sofar + transfer.transfer_time
                transfer_trip = TransferTrip(
                    from_stop=dep_stop,
                    to_stop=arrival_stop,
                    dep_time=time_sofar,
                    arr_time=arrival_time_with_transfer,
                    transport_type=transfer.transport_type
                )

                # Can the journey to the current arrival stop be improved by taking
                # the current transfer trip?
                self._try_to_improve_arrival_stop(
                    k=k,
                    boarding_stop=dep_stop,
                    arrival_stop=arrival_stop,
                    trip=transfer_trip,
                    marked_stops=new_marked_stops
                )

        return new_marked_stops

    def _try_to_improve_arrival_stop(
            self,
            k: int,
            boarding_stop: Stop,
            arrival_stop: Stop,
            trip: Trip,
            marked_stops: Set[Stop]
    ):
        """
        Tries to improve the journey to the provided arrival stop with the provided data.
        If there is an improvement, the arrival stop is added to the provided set
        of marked stops.

        :param k: current round
        :param boarding_stop: stop that the trip is boarded at
        :param arrival_stop: stop that the trip is hopped off at
        :param trip: trip to board to reach the arrival stop
        :param marked_stops: set of stops currently marked by the algorithm
        """

        update_data = LabelUpdate(
            boarding_stop=boarding_stop,
            arrival_stop=arrival_stop,
            new_trip=trip,
            boarding_stop_label=self.round_stop_bags[k][boarding_stop].get_label()
        )
        previous_best = self.round_stop_bags[k][arrival_stop].get_label()
        candidate_label = previous_best.update(
            data=update_data
        )

        # Can the journey to the current arrival stop be improved with the
        # new data (candidate_label)?
        if candidate_label.is_dominating(previous_best, strict=True):
            # Update arrival stop with new label
            self._update_stop(
                k=k,
                stop_to_update=arrival_stop,
                update_with=[candidate_label],
                currently_marked_stops=marked_stops,
            )

            marked_stops.add(arrival_stop)
