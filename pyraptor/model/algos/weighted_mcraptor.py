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
from collections.abc import Iterable
from typing import List, Tuple, Dict

from loguru import logger

from pyraptor.model.algos.base import BaseSharedMobRaptor, SharedMobilityConfig
from pyraptor.model.timetable import (
    RaptorTimetable,
    Stop,
    Route,
    TransferTrip,
    Transfers
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
                 timetable: RaptorTimetable,
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

    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> List[Stop]:
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
                        # Update label with the earliest trip in route leaving from this station
                        boarding_stop = current_stop

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
