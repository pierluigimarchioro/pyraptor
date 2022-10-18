"""
# TODO aggiorna
Weighted implementation of the McRAPTOR algorithm described in the Microsoft paper.
In this implementation, each label has a cost, which is defined as the weighted sum
of all the criteria (i.e. distance, emissions, arrival time, etc.).
This means that the `dominates` relationship changes as follows:
X1 dominates X2 if X1 hasn't got a worse total cost than X2, where both
X1 and X2 are either Labels or Journeys.
This differs from the original implementation of McRAPTOR described in the paper,
which instead says that a label L1 dominates a label L2 if L1 is not worse
than L2 in any criterion.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple, Dict

from loguru import logger

from pyraptor.model.algos.base import BaseRaptor, SharedMobilityConfig
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import (
    Stop,
    Route,
    TransferTrip,
    Transfers,
    RaptorTimetable
)
from pyraptor.model.criteria import (
    GeneralizedCostLabel,
    GeneralizedCostBag,
    GeneralizedCostCriterion,
    CriteriaProvider,
    ArrivalTimeCriterion,
    LabelUpdate,
    DEFAULT_ORIGIN_TRIP
)


class GeneralizedCostRaptor(BaseRaptor[GeneralizedCostLabel, GeneralizedCostBag]):
    """
    Implementation of the Generalized Cost RAPTOR algorithm.
    Just a single criterion is optimized, that is, generalized cost, which is obtained
    as the weighted and normalized sum of a set of criteria.
    """

    def __init__(
            self,
            timetable: RaptorTimetable | RaptorTimetableSM,
            enable_fwd_deps_heuristic: bool,
            enable_sm: bool,
            sm_config: SharedMobilityConfig,
            criteria_provider: CriteriaProvider,
    ):
        """
        :param timetable: object containing the data that will be used by the algorithm
        :param criteria_provider: object that provides properly parameterized criteria for
            the algorithm to use
        """

        super(GeneralizedCostRaptor, self).__init__(
            timetable=timetable,
            enable_sm=enable_sm,
            sm_config=sm_config,
            enable_fwd_deps_heuristic=enable_fwd_deps_heuristic
        )

        self.criteria_provider: CriteriaProvider = criteria_provider
        """Object that provides properly parameterized criteria for
            the algorithm to use"""

    def _initialization(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int
    ) -> List[Stop]:

        # Initialize Round 0 with empty bags.
        # Following rounds are initialized by copying the previous one
        self.round_stop_bags[0] = {}
        for p in self.timetable.stops:
            self.round_stop_bags[0][p] = GeneralizedCostBag()

        self.stop_forward_dependencies: Dict[Stop, List[Stop]] = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = []

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        # Initialize origin stops labels, bags and dependencies
        for from_stop in from_stops:
            with_departure_time = self.criteria_provider.get_criteria(
                defaults={
                    # Default arrival time for origin stops is the departure time
                    ArrivalTimeCriterion: dep_secs
                }
            )

            mc_label = GeneralizedCostLabel(
                arrival_time=dep_secs,
                boarding_stop=from_stop,
                trip=DEFAULT_ORIGIN_TRIP,
                gc_criterion=GeneralizedCostCriterion(criteria=with_departure_time)
            )

            self.round_stop_bags[0][from_stop] = GeneralizedCostBag(labels=[mc_label])

            # From stop has a dependency on itself since it's an origin stop
            self.stop_forward_dependencies[from_stop] = [from_stop]

        marked_stops = [s for s in from_stops]
        return marked_stops

    def _traverse_routes(
            self,
            k: int,
            marked_route_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        new_marked_stops = set()

        for marked_route, marked_stop in marked_route_stops:
            # Get all stops after current stop within the current route
            marked_stop_index = marked_route.stop_index(marked_stop)

            # Traversing through route from marked stop
            remaining_stops_in_route = marked_route.stops[marked_stop_index:]

            # The following steps refer to the three-part processing done for each stop,
            # described in the MC RAPTOR section of the MSFT paper.
            route_bag = GeneralizedCostBag()
            for current_stop_idx, current_stop in enumerate(remaining_stops_in_route):

                # Step 1: route_bag contains all the labels of the stops traversed until this point.
                #   From those labels we then generate the labels for the current stop, to which
                #   we arrive with the same trip boarded at the same boarding stop.
                updated_labels = []
                for label in route_bag.labels:
                    # Boarding stop and trip stay the same, because:
                    #  - they have been updated at the final part of Step 3
                    #  - we are still in the same route, so they are compatible with the
                    #       currently considered stop
                    update_data = LabelUpdate(
                        boarding_stop=label.boarding_stop,
                        arrival_stop=current_stop,
                        new_trip=label.trip,
                        boarding_stop_label=self.round_stop_bags[k][label.boarding_stop].get_label()
                    )
                    label = label.update(data=update_data)

                    updated_labels.append(label)

                # Note that bag B_{r} basically represents all the updated labels of
                #   the stops traversed until this point in the current route
                route_bag = GeneralizedCostBag(labels=updated_labels)

                # Step 2: merge the route bag into the bag currently assigned to the stop
                self._update_stop(
                    k=k,
                    stop_to_update=current_stop,
                    update_with=route_bag.labels,
                    currently_marked_stops=new_marked_stops
                )

                # TODO update comment
                # Step 3: merge B_k(p) into B_r
                # In the non-weighted MC variant, it is B_{k-1}(p) that is merged into B_r,
                #   creating a bag where the best labels from the previous round,
                #   for the current stop, and the best from the current route are kept.
                # However, doing this in the Weighted MC variant breaks forward dependencies
                #   between stops (labels): it may happen that the labels for the current stop
                #   have been updated as part of a forward-dependency resolution in the current round,
                #   which would mean that the arrival times at round k-1 aren't valid anymore.
                #   This is why B_k(p) is used instead.
                current_round_stop_bag = self.round_stop_bags[k][current_stop]
                if len(current_round_stop_bag.labels) > 0:
                    route_bag = route_bag.merge(with_labels=self.round_stop_bags[k][current_stop].labels)
                else:
                    # This happens if the current stop hasn't been visited in the current round
                    # (it has an empty bag): in this case, we take the results of the prev round
                    route_bag = route_bag.merge(with_labels=self.round_stop_bags[k - 1][current_stop].labels)

                # Assign the earliest boardable trips to all the labels in the route_bag
                updated_labels = []
                for label in route_bag.labels:
                    earliest_trip = marked_route.earliest_trip(
                        label.arrival_time, current_stop
                    )

                    if earliest_trip is not None:
                        # If the trip is different from the previous one, we board the trip
                        #   at current_stop, else the trip is still boarded at the old boarding stop.
                        # This basically results in consecutive stops (labels) of the same journey
                        #   only pointing to the first stop of each different trip in the journey,
                        #   as opposed to pointing to each intermediate stop of each different trip.
                        if label.trip != earliest_trip:
                            boarding_stop = current_stop
                        else:
                            boarding_stop = label.boarding_stop

                        # Update the current label with the new trip and associated boarding stop
                        update_data = LabelUpdate(
                            boarding_stop=boarding_stop,
                            arrival_stop=current_stop,
                            new_trip=earliest_trip,
                            boarding_stop_label=self.round_stop_bags[k][boarding_stop].get_label()
                        )
                        label = label.update(data=update_data)

                        updated_labels.append(label)

                route_bag = GeneralizedCostBag(labels=updated_labels)

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

            for stop_to_improve in other_station_stops:
                # Create temp copy of B_k(p_i)
                temp_bag = GeneralizedCostBag()
                for label in self.round_stop_bags[k][current_stop].labels:
                    transfer = transfers.stop_to_stop_idx[(current_stop, stop_to_improve)]

                    # Update label with new transfer trip, because the arrival time
                    # at other_stop is better with said trip
                    # NOTE: Each label contains the trip with which one arrives at the current stop
                    #   with k legs by boarding the trip at from_stop, along with the criteria
                    #   (i.e. boarding time, fares, number of legs)
                    transfer_arrival_time = label.arrival_time + transfer.transfer_time
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=stop_to_improve,
                        dep_time=label.arrival_time,
                        arr_time=transfer_arrival_time,
                        transport_type=transfer.transport_type
                    )

                    update_data = LabelUpdate(
                        boarding_stop=current_stop,
                        arrival_stop=stop_to_improve,
                        new_trip=transfer_trip,
                        boarding_stop_label=self.round_stop_bags[k][current_stop].get_label()
                    )
                    label = label.update(data=update_data)

                    # TODO old
                    # temp_bag.add(label)
                    temp_bag = temp_bag.merge(with_labels=[label])

                # Merge temp bag into B_k(p_j)
                self._update_stop(
                    k=k,
                    stop_to_update=stop_to_improve,
                    update_with=temp_bag.labels,
                    currently_marked_stops=marked_stops_transfers
                )

        logger.debug(f"{len(marked_stops_transfers)} transferable stops added")

        return list(marked_stops_transfers)
