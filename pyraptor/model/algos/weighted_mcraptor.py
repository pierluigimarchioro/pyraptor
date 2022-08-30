"""
Weighted implementation of the McRAPTOR algorithm described in the Microsoft paper.
In this implementation, each label has a cost, which is defined as the weighted sum
of all the criteria (i.e. distance, emissions, arrival time, etc.).
This means that the `dominates` changes as follows:
X1 dominates X2 if X1 hasn't got a worse cost than X2,
where both X1 and X2 are either Labels or Journeys.
This differs from the original implementation of McRAPTOR described in the paper,
which instead says that a label L1 dominates a label L2 if L1 is not worse
than L2 in any criterion.
"""

from __future__ import annotations

import os.path
from collections.abc import Iterable, Mapping
from copy import copy
from typing import List, Tuple, Dict

from loguru import logger

from pyraptor.model.algos.base import BaseSharedMobRaptor, SharedMobilityConfig
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import (
    Stop,
    Route,
    TransferTrip,
    Transfers,
    RaptorTimetable
)
from pyraptor.model.criteria import (
    Bag,
    MultiCriteriaLabel,
    CriteriaProvider,
    ArrivalTimeCriterion,
    LabelUpdate,
    DEFAULT_ORIGIN_TRIP
)


class WeightedMcRaptorAlgorithm(BaseSharedMobRaptor[Bag, MultiCriteriaLabel]):
    """
    Implementation of the More Criteria RAPTOR Algorithm discussed in the original RAPTOR paper,
    with some modifications and improvements:
    - each criterion is weighted and each label has a generalized cost that is used
        to make comparison and determine domination
    - transfers from the origin stops are evaluated immediately to widen
        the set of reachable stops before the first round is executed
    - it is possible to use shared mobility, real-time data
    """

    criteria_file_path: str | bytes | os.PathLike
    """Path to the criteria configuration file"""

    def __init__(self,
                 timetable: RaptorTimetable | RaptorTimetableSM,
                 enable_sm: bool,
                 sm_config: SharedMobilityConfig,
                 criteria_file_path: str | bytes | os.PathLike):
        """
        :param timetable: object containing the data that will be used by the algorithm
        :param criteria_file_path: path to the criteria configuration file
        """

        super(WeightedMcRaptorAlgorithm, self).__init__(
            timetable=timetable,
            enable_sm=enable_sm,
            sm_config=sm_config
        )

        if not os.path.exists(criteria_file_path):
            raise FileNotFoundError(f"'{criteria_file_path}' is not a valid path to a criteria configuration file.")

        self.criteria_file_path: str | bytes | os.PathLike = criteria_file_path
        """Path to the criteria configuration file"""

    def run(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int,
            rounds: int
    ) -> Mapping[Stop, Bag]:
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

            # TODO I don't know how to refactor code to avoid duplicating the 80 lines above this part.
            #   this method is in fact the copy of the base class method, except for this part below.
            #   However, the below step needs to be performed strictly on the stops marked
            #   by the transfer step, and is specific to the Weighted MC variant.
            if k == rounds:
                self._update_until_convergence(
                    last_round=k,
                    last_round_transfer_marked_stops=marked_transfer_stops,
                    last_round_marked_route_stops=route_marked_stops
                )

        return self.bag_round_stop[rounds]

    def _initialization(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int,
            rounds: int) -> List[Stop]:
        # Initialize empty bag, i.e. B_k(p) = [] for every k and p
        self.bag_round_stop: Dict[int, Dict[Stop, Bag]] = {}
        for k in range(0, rounds + 1):
            self.bag_round_stop[k] = {}

            for p in self.timetable.stops:
                self.bag_round_stop[k][p] = Bag()

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        criteria_provider = CriteriaProvider(criteria_config_path=self.criteria_file_path)

        # Add to bag multi-criterion label for origin stops
        for from_stop in from_stops:
            with_departure_time = criteria_provider.get_criteria(
                defaults={
                    # Default arrival time for origin stops is the departure time
                    ArrivalTimeCriterion: dep_secs
                }
            )
            mc_label = MultiCriteriaLabel(
                boarding_stop=from_stop,
                trip=DEFAULT_ORIGIN_TRIP,
                criteria=with_departure_time
            )

            self.bag_round_stop[0][from_stop].add(mc_label)
            self.best_bag[from_stop] = mc_label

        marked_stops = [s for s in from_stops]
        return marked_stops

    def _traverse_routes(
            self,
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        new_marked_stops = set()

        for marked_route, marked_stop in route_marked_stops:
            # Traversing through route from marked stop
            route_bag = Bag()

            # Get all stops after current stop within the current route
            marked_stop_index = marked_route.stop_index(marked_stop)
            remaining_stops_in_route = marked_route.stops[marked_stop_index:]

            # The following steps refer to the three-part processing done for each stop,
            # described in the McRAPTOR section of the MSFT paper.
            for current_stop_idx, current_stop in enumerate(remaining_stops_in_route):

                # Step 1: update the earliest arrival times and criteria for each label L in route-bag
                updated_labels = []
                for label in route_bag.labels:
                    # Here the arrival time for the label needs to be updated to that
                    #   of the current stop for its associated trip (Step 1).
                    # This means that the (implicit) arrival stop associated
                    #   to the label is `current_stop`
                    # Boarding stop and trip stay the same, because
                    #   they have been updated at the final part of Step 3
                    update_data = LabelUpdate(
                        boarding_stop=label.boarding_stop,  # Boarding stop stays the same
                        arrival_stop=current_stop,
                        old_trip=label.trip,
                        new_trip=label.trip,  # Trip stays the same
                        best_labels=self.best_bag
                    )
                    label = label.update(data=update_data)

                    updated_labels.append(label)

                # Route bag B_{r} basically represents all the updated labels of
                #   the stops in the current marked route. Notably, each updated label means
                #   that you can board a trip with better characteristics
                #   (i.e. earlier arrival time, better fares, lesser number of trips)
                route_bag = Bag(labels=updated_labels)

                # Step 2: merge bag_route into bag_round_stop and remove dominated labels
                # NOTE: merging the current stop bag with route bag basically means to keep
                #   the labels that allow to get to the current stop in the most efficient way
                self.bag_round_stop[k][current_stop] = self.bag_round_stop[k][current_stop].merge(
                    route_bag
                )
                bag_update = self.bag_round_stop[k][current_stop].updated

                # Mark the stop if bag is updated and update the best label(s) for that stop
                # Updated bag means that the current stop brought some improvements
                if bag_update:
                    self.best_bag[current_stop] = self.bag_round_stop[k][current_stop].get_best_label()
                    new_marked_stops.add(current_stop)

                # Step 3: merge B_{k-1}(p) into B_r
                # Merging here creates a bag where the best labels from the previous round,
                #   for the current stop, and the best from the current route are kept
                route_bag = route_bag.merge(self.bag_round_stop[k - 1][current_stop])

                # Assign trips to all newly added labels in route_bag
                # This is the trip on which we board
                # This step is needed because the labels from the previous round need
                #   to be updated with the new earliest boardable trip
                updated_labels = []
                for label in route_bag.labels:
                    earliest_trip = marked_route.earliest_trip(
                        label.get_earliest_arrival_time(), current_stop
                    )
                    if earliest_trip is not None:
                        # If the trip is different from the previous one, we board the trip
                        #   at current_stop, else the trip is still boarded at the old boarding stop.
                        # This basically results in consecutive stops of the same journey
                        #   only pointing to the first stop of each different trip in the journey,
                        #   as opposed to pointing to each intermediate stop of each different trip.
                        if label.trip != earliest_trip:
                            boarding_stop = current_stop
                        else:
                            boarding_stop = label.boarding_stop

                        # This update is just "temporary", meaning that we board the
                        # current trip at the current stop and also arrive at the current_stop,
                        # but then the actual arrival stop of the label is assigned
                        # at Step 1 of the next iteration
                        update_data = LabelUpdate(
                            boarding_stop=boarding_stop,
                            arrival_stop=current_stop,
                            old_trip=label.trip,
                            new_trip=earliest_trip,
                            best_labels=self.best_bag
                        )
                        label = label.update(data=update_data)

                        updated_labels.append(label)

                route_bag = Bag(labels=updated_labels)

        logger.debug(f"{len(new_marked_stops)} reachable stops added")

        return list(new_marked_stops)

    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Transfers
    ) -> List[Stop]:
        marked_stops_transfers = set()

        # Add in transfers to other platforms (same station) and stops
        for current_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)
            other_station_stops = [t.to_stop for t in transfers if t.from_stop == current_stop]

            for other_stop in other_station_stops:
                # Create temp copy of B_k(p_i)
                temp_bag = Bag()
                for label in self.bag_round_stop[k][current_stop].labels:
                    transfer = transfers.stop_to_stop_idx[(current_stop, other_stop)]

                    # Update label with new transfer trip, because the arrival time
                    # at other_stop is better with said trip
                    # NOTE: Each label contains the trip with which one arrives at the current stop
                    #   with k legs by boarding the trip at from_stop, along with the criteria
                    #   (i.e. boarding time, fares, number of legs)
                    transfer_arrival_time = label.get_earliest_arrival_time() + transfer.transfer_time
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=other_stop,
                        dep_time=label.get_earliest_arrival_time(),
                        arr_time=transfer_arrival_time,
                        transport_type=transfer.transport_type
                    )

                    update_data = LabelUpdate(
                        boarding_stop=current_stop,
                        arrival_stop=other_stop,
                        old_trip=label.trip,
                        new_trip=transfer_trip,
                        best_labels=self.best_bag
                    )
                    label = label.update(data=update_data)

                    temp_bag.add(label)

                # Merge temp bag into B_k(p_j)
                self.bag_round_stop[k][other_stop] = self.bag_round_stop[k][other_stop].merge(
                    temp_bag
                )
                bag_update = self.bag_round_stop[k][other_stop].updated

                # Mark the stop and update the best label collection
                # if there were improvements (bag is updated)
                if bag_update:
                    self.best_bag[other_stop] = self.bag_round_stop[k][other_stop].get_best_label()
                    marked_stops_transfers.add(other_stop)

        logger.debug(f"{len(marked_stops_transfers)} transferable stops added")

        return list(marked_stops_transfers)

    def _update_until_convergence(
            self,
            last_round: int,
            last_round_transfer_marked_stops: List[Stop],
            last_round_marked_route_stops: List[Tuple[Route, Stop]],
    ):
        """
        This step is needed because Weighted MC RAPTOR can't guarantee that temporal
        dependencies between labels won't be broken in the transfer steps (shared mob included):
        since time is a criteria just like the others, it can be 0-weighted, which means
        it won't be considered when dominating labels are identified. This also means
        that a dominating label at round X can worsen the time of round X-1, therefore
        potentially breaking the correctness of the transfer steps in relation to the
        route traversing step.

        Example:
        Given stop X,Y and Z, after the route traversing step (2nd) of the last round,
        you can arrive at Y from X at time K and at Z from Y at time K+1.
        Then, the transfer steps are executed, and it is found that you can use a transfer
        that improves the generalized cost of going from X to Y, but now the arrival
        time at Y from X is K+2. This means that you arrive at Z from Y at time K+1, but
        the best (minimum cost) path to Y has an arrival time of K+2. This is an absurd.

        It is therefore necessary, after the last round, to continue updating all the labels
        (and hence their dependencies) until convergence is reached, so that arrival time
        relationships are correct. This is done by repeating the route traversing and
        transfer steps only on last round transfer-improved stops, until no more stops
        (labels) are improved.

        :param last_round: the number of the last RAPTOR round
        :param last_round_transfer_marked_stops: stops marked (improved) at the transfer
            step of the last round
        :param last_round_marked_route_stops: (route, stop) pairs marked at the
            route accumulation step of the last round
        :return:
        """

        # Select the (route, stop) pairs whose routes contain stops updated
        # at the last transfer steps.
        # It is important to note that this filtering is done on the accumulated
        # routes of the last round: this way, it is guaranteed that no additional
        # routes are explored, therefore preserving the concept that in K rounds
        # a maximum of K routes can be hopped on to reach any stop.
        transfer_improved_route_stops: List[Tuple[Route, Stop]] = [
            (r, s) for r, s in last_round_marked_route_stops
            if len(set(r.stops).intersection(last_round_transfer_marked_stops)) > 0
        ]

        logger.warning("Starting convergence step...")
        convergence_round = 0
        while len(transfer_improved_route_stops) > 0:  # update until convergence
            convergence_round += 1
            logger.debug(f"Convergence round #{convergence_round}")

            actual_round = last_round + convergence_round
            self.bag_round_stop[actual_round] = copy(self.bag_round_stop[actual_round - 1])

            # Keep doing step 2-3-4 of the algorithm until no more stops are marked
            trip_improved_stops = self._traverse_routes(
                k=actual_round,
                route_marked_stops=transfer_improved_route_stops
            )
            transfer_improved_stops = self._improve_with_transfers(
                k=actual_round,
                marked_stops=trip_improved_stops,
                transfers=self.timetable.transfers
            )

            if self.enable_sm:
                sm_improved_stops = self._improve_with_sm_transfers(
                    k=actual_round,
                    marked_stops=transfer_improved_stops
                )
                transfer_improved_stops = set(transfer_improved_stops).union(sm_improved_stops)

            transfer_improved_route_stops = [
                (r, s) for r, s in transfer_improved_route_stops
                if len(set(r.stops).intersection(transfer_improved_stops)) > 0
            ]

        # Apply the results of the convergence step to the last actual RAPTOR round
        self.bag_round_stop[last_round] = copy(self.bag_round_stop[len(self.bag_round_stop) - 1])
