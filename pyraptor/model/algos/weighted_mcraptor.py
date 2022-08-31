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
from collections import ChainMap
from collections.abc import Iterable, Mapping, MutableMapping
from copy import copy
from typing import List, Tuple, Dict, Set

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

        self.stop_forward_dependencies: Dict[Stop, List[Stop]] = {}
        """Dictionary that pairs each stop with the list of stops that depend on it. 
        For example, in a journey x1, x2, ..., xn, stop x2 depends on
        stop x1 because it comes later, hence the name 'forward' dependency"""

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

        # TODO debug - is stability important?
        initial_marked_stops = list(sorted(initial_marked_stops, key=lambda x: x.id))

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

        # TODO debug - is stability important?
        marked_stops = list(sorted(marked_stops, key=lambda x: x.id))

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

            # TODO debug - is stability important?
            route_marked_stops = list(sorted(route_marked_stops, key=lambda x: x[1].id))

            # Update stop arrival times calculated basing on reachable stops
            marked_trip_stops = self._traverse_routes(
                k=k,
                route_marked_stops=route_marked_stops
            )
            logger.debug(f"{len(marked_trip_stops)} reachable stops added")

            # TODO debug - is stability important?
            marked_trip_stops = list(sorted(marked_trip_stops, key=lambda x: x.id))

            # Add footpath transfers and update
            marked_transfer_stops = self._improve_with_transfers(
                k=k,
                marked_stops=marked_trip_stops,
                transfers=self.timetable.transfers
            )
            logger.debug(f"{len(marked_transfer_stops)} transferable stops added")

            # TODO debug - is stability important?
            marked_transfer_stops = list(sorted(marked_transfer_stops, key=lambda x: x.id))

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

            # TODO debug - is stability important?
            marked_stops = list(sorted(marked_stops, key=lambda x: x.id))

            logger.debug(f"{len(marked_stops)} stops to evaluate in next round")

            # TODO I don't know how to refactor code to avoid duplicating the 80 lines above this part.
            #   this method is in fact the copy of the base class method, except for this part below.
            #   However, the below step needs to be performed strictly on the stops marked
            #   by the transfer step, and is specific to the Weighted MC variant.
            #   ANSWER: since the convergence part is removed, i can remove the run method implementation:
            #       fwd dependency updates are handled in the overridden methods

            # TODO debug - print the journey for each round before converging
            if k == rounds:
                dest_name = 'MANTOVA'
                logger.warning("Printing best journey for each round")
                from pyraptor.model.output import get_journeys_to_destinations
                dest_stops = {
                    st.name: self.timetable.stations.get_stops(st.name) for st in self.timetable.stations
                    if st.name == dest_name  # only true destination stops
                }
                for k_j in range(1, rounds + 1):
                    logger.warning(f"------ Round {k_j}: ------\n")
                    journeys = get_journeys_to_destinations(
                        origin_stops=from_stops,
                        destination_stops=dest_stops,
                        best_labels=self.bag_round_stop[k_j]
                    )

                    if dest_name not in journeys:
                        logger.debug(f"No journeys at round {k_j}")
                        continue

                    for j in journeys[dest_name]:
                        j.print()
                        logger.debug("\n")

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

        self.stop_forward_dependencies: Dict[Stop, List[Stop]] = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = []

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        criteria_provider = CriteriaProvider(criteria_config_path=self.criteria_file_path)

        logger.debug("Criteria Configuration:")
        for c in criteria_provider.get_criteria():
            logger.debug(repr(c))

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
                arrival_stop=from_stop,  # Origin labels depart and arrive at the same stop: the origin
                trip=DEFAULT_ORIGIN_TRIP,
                criteria=with_departure_time
            )

            self.bag_round_stop[0][from_stop].add(mc_label)
            self.best_bag[from_stop] = mc_label

            # From stop has a dependency on itself since it's an origin stop
            self.stop_forward_dependencies[from_stop] = [from_stop]

        marked_stops = [s for s in from_stops]
        return marked_stops

    def _traverse_routes(
            self,
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        new_marked_stops = set()

        for marked_route, marked_stop in route_marked_stops:
            # Get all stops after current stop within the current route
            marked_stop_index = marked_route.stop_index(marked_stop)

            # Traversing through route from marked stop
            remaining_stops_in_route = marked_route.stops[marked_stop_index:]

            # The following steps refer to the three-part processing done for each stop,
            # described in the McRAPTOR section of the MSFT paper.
            route_bag = Bag()
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
                        boarding_stop=label.boarding_stop,  # Boarding stop stays the same
                        arrival_stop=current_stop,
                        old_trip=label.trip,
                        new_trip=label.trip,  # Trip stays the same
                        best_labels=self.best_bag,

                        # TODO debug
                        current_round=k
                    )
                    label = label.update(data=update_data)

                    updated_labels.append(label)

                # Note that bag B_{r} basically represents all the updated labels of
                #   the stops traversed until this point in the current route
                route_bag = Bag(labels=updated_labels)

                # Step 2: merge the route bag into the bag currently assigned to the stop
                self._update_bag(
                    k=k,
                    stop_to_update=current_stop,
                    update_with=route_bag,
                    currently_marked_stops=new_marked_stops
                )

                # Step 3: merge B_k(p) into B_r
                # In the non-weighted MC variant, it is B_{k-1}(p) that is merged into B_r,
                #   creating a bag where the best labels from the previous round,
                #   for the current stop, and the best from the current route are kept.
                # However, doing this in the Weighted MC variant breaks forward dependencies
                #   between stops (labels): it may happen that the labels for the current stop
                #   have been updated as part of a forward-dependency resolution in the current round,
                #   which would mean that the arrival times at round k-1 aren't valid anymore.
                #   This is why B_k(p) is used instead.
                current_round_stop_bag = self.bag_round_stop[k][current_stop]
                if len(current_round_stop_bag.labels) > 0:
                    route_bag = route_bag.merge(self.bag_round_stop[k][current_stop])
                else:
                    # This happens if the current stop hasn't been visited in the current round
                    # (it has an empty bag): in this case, we take the results of the prev round
                    route_bag = route_bag.merge(self.bag_round_stop[k - 1][current_stop])

                # Assign the earliest boardable trips to all the labels in the route_bag
                updated_labels = []
                for label in route_bag.labels:
                    earliest_trip = marked_route.earliest_trip(
                        label.get_earliest_arrival_time(), current_stop
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

                        # Update the current label with the new trip and
                        # associated boarding stop
                        update_data = LabelUpdate(
                            boarding_stop=boarding_stop,
                            arrival_stop=current_stop,
                            old_trip=label.trip,
                            new_trip=earliest_trip,
                            best_labels=self.best_bag,

                            # TODO debug
                            current_round=k
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

            for stop_to_improve in other_station_stops:
                # Create temp copy of B_k(p_i)
                temp_bag = Bag()
                for label in self.bag_round_stop[k][current_stop].labels:
                    transfer = transfers.stop_to_stop_idx[(current_stop, stop_to_improve)]

                    # Update label with new transfer trip, because the arrival time
                    # at other_stop is better with said trip
                    # NOTE: Each label contains the trip with which one arrives at the current stop
                    #   with k legs by boarding the trip at from_stop, along with the criteria
                    #   (i.e. boarding time, fares, number of legs)
                    transfer_arrival_time = label.get_earliest_arrival_time() + transfer.transfer_time
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=stop_to_improve,
                        dep_time=label.get_earliest_arrival_time(),
                        arr_time=transfer_arrival_time,
                        transport_type=transfer.transport_type
                    )

                    update_data = LabelUpdate(
                        boarding_stop=current_stop,
                        arrival_stop=stop_to_improve,
                        old_trip=label.trip,
                        new_trip=transfer_trip,
                        best_labels=self.best_bag,

                        # TODO debug
                        current_round=k
                    )
                    label = label.update(data=update_data)

                    temp_bag.add(label)

                # Merge temp bag into B_k(p_j)
                self._update_bag(
                    k=k,
                    stop_to_update=stop_to_improve,
                    update_with=temp_bag,
                    currently_marked_stops=marked_stops_transfers
                )

        logger.debug(f"{len(marked_stops_transfers)} transferable stops added")

        return list(marked_stops_transfers)

    def _update_bag(
            self,
            k: int,
            stop_to_update: Stop,
            update_with: Bag,
            currently_marked_stops: Set[Stop]
    ):
        """

        :param k:
        :param stop_to_update:
        :param update_with:
        :param currently_marked_stops:
        :return:
        """

        updated_bag = self.bag_round_stop[k][stop_to_update].merge(
            update_with
        )
        bag_update = updated_bag.updated

        # Mark the stop if bag is updated and update the best label(s) for that stop
        # Updated bag means that the current stop brought some improvements
        if bag_update:
            successful_fwd_updates, updated_fwd_dependencies = self._update_forward_dependencies(
                k=k,
                updated_stop=stop_to_update,
                updated_bag=updated_bag
            )

            if successful_fwd_updates:
                # Add the current stop to the forward dependencies of the new boarding stop
                for lbl in updated_bag.labels:
                    self.stop_forward_dependencies[lbl.boarding_stop].append(stop_to_update)

                # Remove the current stop from forward dependencies of the old boarding stop
                old_bag = self.bag_round_stop[k][stop_to_update]
                for old_lbl in old_bag.labels:
                    self.stop_forward_dependencies[old_lbl.boarding_stop].remove(stop_to_update)

                # Update the bags of the current stop to update
                self.bag_round_stop[k][stop_to_update] = updated_bag
                self.best_bag[stop_to_update] = copy(updated_bag.get_best_label())
                currently_marked_stops.add(stop_to_update)

                # Now update all the bags of the stops dependent on the current one
                # with what was previously calculated
                for dep_stop, dep_bag in updated_fwd_dependencies.items():
                    self.bag_round_stop[k][dep_stop] = dep_bag
                    self.best_bag[dep_stop] = copy(dep_bag.get_best_label())

    def _update_forward_dependencies(
            self,
            k: int,
            updated_stop: Stop,
            updated_bag: Bag
    ) -> Tuple[bool, MutableMapping[Stop, Bag]]:
        """
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
        at time K+1, but the best (minimum cost) path to Y has an arrival time of K+2.
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

        updated_fwd_deps: MutableMapping[Stop, Bag] = ChainMap()

        for updated_label in updated_bag.labels:
            fwd_dependent_stops: Iterable[Stop] = self.stop_forward_dependencies[updated_stop]

            for fwd_dep_stop in fwd_dependent_stops:
                # It is certain that a dependent stop has at least one label assigned to it:
                # since it is dependent, it means that it has been reached by the algorithm
                # TODO what if there are two or more labels with the boarding stop == updated_stop?
                #   i think it is fine, since it is not a problem if only one label is kept, updated
                #   and then be the only one assigned back to the original dep_stop: it would mean that
                #   some good paths would be overwritten, but it is an edge case that can be solved
                #   this way since the algorithm is greedy
                fwd_dep_label = next(
                    filter(
                        lambda lbl: lbl.boarding_stop == updated_stop,
                        self.bag_round_stop[k][fwd_dep_stop].labels,
                    )
                )

                # Create a temporary best bag, because the actual best bag doesn't contain
                # the updated labels generated during this dependency update.
                # Updated label is placed first in the chain map, so that it has the priority
                # when the value for the updated stop (which is the one the fwd-dependent
                # labels are dependent on) is retrieved.
                temp_best_bag: MutableMapping[Stop, MultiCriteriaLabel] = ChainMap(
                    {updated_stop: updated_label},
                    self.best_bag
                )

                # Consider the trip of the forward-dependent label: it is certain that
                # such trip contains both the updated stop (which acts as the boarding stop)
                # and the forward-dependent stop (which acts as the arrival stop).
                # The two cases where the trip is a transfer or not are handled differently
                if isinstance(fwd_dep_label.trip, TransferTrip):
                    # TODO what happens with shared mob? what transfers to use?
                    #   chiedere a Seba se uso di timetable.transfers potrebbe non aggiornare
                    #   correttamente shared-mob
                    transfers = self.timetable.transfers
                    transfer = transfers.stop_to_stop_idx[(updated_stop, fwd_dep_stop)]

                    transfer_arrival_time = updated_label.get_earliest_arrival_time() + transfer.transfer_time
                    transfer_trip = TransferTrip(
                        from_stop=updated_stop,
                        to_stop=fwd_dep_stop,
                        dep_time=updated_label.get_earliest_arrival_time(),
                        arr_time=transfer_arrival_time,
                        transport_type=transfer.transport_type
                    )

                    update_data = LabelUpdate(
                        boarding_stop=updated_stop,
                        arrival_stop=fwd_dep_stop,
                        old_trip=fwd_dep_label.trip,
                        new_trip=transfer_trip,
                        best_labels=temp_best_bag,

                        # TODO debug - minus sign to signal update happens in this method
                        current_round=-k
                    )
                    updated_fwd_dep_label = fwd_dep_label.update(data=update_data)
                    updated_fwd_dep_bag = Bag(labels=[updated_fwd_dep_label])
                    updated_fwd_deps[fwd_dep_stop] = updated_fwd_dep_bag
                else:
                    current_route = fwd_dep_label.trip.route_info.route
                    new_earliest_trip = current_route.earliest_trip(
                        dts_arr=updated_label.get_earliest_arrival_time(),
                        stop=updated_stop
                    )

                    if new_earliest_trip is None:
                        # No trip available means that the label can't be updated:
                        # returning false to roll back the update chain
                        return False, {}
                    else:
                        update_data = LabelUpdate(
                            boarding_stop=updated_stop,
                            arrival_stop=fwd_dep_stop,
                            old_trip=fwd_dep_label.trip,
                            new_trip=new_earliest_trip,
                            best_labels=temp_best_bag,

                            # TODO debug - minus sign to signal that update happens in this method
                            current_round=-k
                        )
                        updated_fwd_dep_label = fwd_dep_label.update(data=update_data)
                        updated_fwd_dep_bag = Bag(labels=[updated_fwd_dep_label])
                        updated_fwd_deps[fwd_dep_stop] = updated_fwd_dep_bag

                successful_update, rec_updated_fwd_deps = self._update_forward_dependencies(
                    k=k,
                    updated_stop=fwd_dep_stop,
                    updated_bag=updated_fwd_dep_bag
                )

                if not successful_update:
                    # Updates down the dependency chain failed, need to roll back
                    return False, {}
                else:
                    # It is noted that the merge between the current and the recursively created
                    # maps cannot lead to conflict (i.e. a key is present in more than one map),
                    # because there can't be cycles in an itinerary
                    updated_fwd_deps = ChainMap(updated_fwd_deps, rec_updated_fwd_deps)

        # The forward update correctly resolved, so newly calculated dependencies can be returned
        return True, updated_fwd_deps
