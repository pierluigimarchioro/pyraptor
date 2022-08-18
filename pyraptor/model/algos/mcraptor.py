"""
This is the original implementation of the McRAPTOR algorithm,
which closely follows what is described in the Microsoft paper.
TODO The old model classes have been locally copied to this file,
    since criteria.py now contains the new multi-criteria classes.
    Consider moving them to their own file (old_mc_structures.py?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable
from copy import copy
from time import perf_counter

import numpy as np
from loguru import logger
from itertools import compress

from pyraptor.model.timetable import (
    RaptorTimetable,
    Stop,
    Route,
    TransferTrip,
    TransportType,
    Trip,
)
from pyraptor.model.output import Journey


@dataclass
class Leg:
    """Leg"""

    from_stop: Stop
    to_stop: Stop
    trip: Trip
    earliest_arrival_time: int
    fare: int = 0
    n_trips: int = 0

    @property
    def criteria(self) -> Iterable:
        """Criteria"""
        return [self.earliest_arrival_time, self.fare, self.n_trips]

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
        arrival_time_compatible = (
                other_leg.earliest_arrival_time >= self.earliest_arrival_time
        )

        n_trips_compatible = (
            other_leg.n_trips >= self.n_trips
            if other_leg.is_same_station_transfer()
            else other_leg.n_trips > self.n_trips
        )

        criteria_compatible = np.all(
            np.array([c for c in other_leg.criteria])
            >= np.array([c for c in self.criteria])
        )

        return all([arrival_time_compatible, n_trips_compatible, criteria_compatible])

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
            fare=self.fare,
        )


@dataclass(frozen=True)
class Label:
    """Label"""

    earliest_arrival_time: int
    fare: int  # total fare
    trip: Trip | None  # trip to take to obtain travel_time and fare
    from_stop: Stop  # stop to hop-on the trip
    n_trips: int = 0
    infinite: bool = False

    @property
    def criteria(self):
        """Criteria"""
        return [self.earliest_arrival_time, self.fare, self.n_trips]

    def update(self, earliest_arrival_time=None, fare_addition=None, from_stop=None) -> Label:
        """
        Updates the current label with the provided earliest arrival time and fare_addition.
        Returns the updated label

        :param earliest_arrival_time: new earliest arrival time
        :param fare_addition: fare to add to the current
        :param from_stop: new stop that the label refers to
        :return: updated label
        """
        return copy(
            Label(
                earliest_arrival_time=earliest_arrival_time
                if earliest_arrival_time is not None
                else self.earliest_arrival_time,
                fare=self.fare + fare_addition
                if fare_addition is not None
                else self.fare,
                trip=self.trip,
                from_stop=from_stop if from_stop is not None else self.from_stop,
                n_trips=self.n_trips,
                infinite=self.infinite,
            )
        )

    def update_trip(self, trip: Trip, new_boarding_stop: Stop) -> Label:
        """
        Updates the trip and the boarding stop associated with the current label.
        Returns the updated label.

        :param trip: new trip to update the label with
        :param new_boarding_stop: new boarding stop to update the label with, only if the provided
            trip is different from the current one. Otherwise, the current stop is not updated.
        :return: updated label
        """

        return copy(
            Label(
                earliest_arrival_time=self.earliest_arrival_time,
                fare=self.fare,
                trip=trip,
                from_stop=new_boarding_stop if self.trip != trip else self.from_stop,
                n_trips=self.n_trips + 1 if self.trip != trip else self.n_trips,
                infinite=self.infinite,
            )
        )


@dataclass(frozen=True)
class Bag:
    """
    Bag B(k,p) or route bag B_r
    """

    labels: List[Label] = field(default_factory=list)
    updated: bool = False

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Bag({self.labels}, updated={self.updated})"

    def add(self, label: Label):
        """Add"""
        self.labels.append(label)

    def merge(self, other_bag: Bag) -> Bag:
        """Merge other bag in current bag and return updated Bag"""

        pareto_labels = self.labels + other_bag.labels

        if len(pareto_labels) == 0:
            return Bag(labels=[], updated=False)

        pareto_labels = pareto_set(pareto_labels)
        bag_update = True if pareto_labels != self.labels else False

        return Bag(labels=pareto_labels, updated=bag_update)

    def labels_with_trip(self):
        """All labels with trips, i.e. all labels that are reachable with a trip with given criterion"""
        return [l for l in self.labels if l.trip is not None]

    def earliest_arrival(self) -> int:
        """Earliest arrival"""
        return min([self.labels[i].earliest_arrival_time for i in range(len(self))])


def pareto_set(labels: List[Label], keep_equal=False):
    """
    Find the pareto-efficient points

    :param labels: list with labels
    :param keep_equal: return also labels with equal criteria
    :return: list with pairwise non-dominating labels
    """

    is_efficient = np.ones(len(labels), dtype=bool)
    labels_criteria = np.array([label.criteria for label in labels])
    for i, current_criteria in enumerate(labels_criteria):
        if is_efficient[i]:
            # Keep any point with a lower cost
            # This is the strict-domination approach: label A dominates label B
            #   if it is strictly better in at least one criterion
            if keep_equal:
                is_efficient[is_efficient] = np.any(
                    labels_criteria[is_efficient] < current_criteria, axis=1
                ) + np.all(labels_criteria[is_efficient] == current_criteria, axis=1)

            else:
                is_efficient[is_efficient] = np.any(
                    labels_criteria[is_efficient] < current_criteria, axis=1
                )

            is_efficient[i] = True  # And keep self

    return list(set(compress(labels, is_efficient)))


class McRaptorAlgorithm:
    """McRAPTOR Algorithm"""

    def __init__(self, timetable: RaptorTimetable):
        self.timetable = timetable

    def run(
            self, from_stops: List[Stop], dep_secs: int, rounds: int, previous_run: Dict[Stop, Bag] = None
    ) -> Tuple[Dict[int, Dict[Stop, Bag]], int]:
        """Run Round-Based Algorithm"""

        s = perf_counter()

        # Initialize empty bag, i.e. B_k(p) = [] for every k and p
        bag_round_stop: Dict[int, Dict[Stop, Bag]] = {}
        for k in range(0, rounds + 1):
            bag_round_stop[k] = {}
            for p in self.timetable.stops:
                bag_round_stop[k][p] = Bag()

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        # Initialize bag for round 0, i.e. add Labels with criterion 0 for all from stops
        if previous_run is not None:
            # For the range query
            bag_round_stop[0] = copy(previous_run)

        # Add origin stops to bag
        for from_stop in from_stops:
            bag_round_stop[0][from_stop].add(
                Label(
                    earliest_arrival_time=dep_secs,
                    fare=0,
                    trip=None,
                    from_stop=from_stop
                )
            )

        marked_stops = from_stops

        # Run rounds
        actual_rounds = 0
        for k in range(1, rounds + 1):
            logger.info(f"Analyzing possibilities round {k}")
            logger.debug(f"Stops to evaluate count: {len(marked_stops)}")

            # Copy bag from previous round
            bag_round_stop[k] = copy(bag_round_stop[k - 1])

            if len(marked_stops) > 0:
                actual_rounds = k

                # Accumulate routes serving marked stops from previous round
                # i.e. get the best (r,p) pairs where p is the first stop reachable in route r
                route_marked_stops = self.accumulate_routes(marked_stops)

                # Traverse each route
                # i.e. update the labels of each marked stop, assigning to each stop
                #   the earliest trip that can be caught and the earliest arrival
                #   time of that trip at that stop
                bag_round_stop, marked_stops_trips = self.traverse_route(
                    bag_round_stop, k, route_marked_stops
                )

                # Now add (footpath) transfers between stops and update
                # i.e. update the arrival time to each stop by considering (foot) transfers
                bag_round_stop, marked_stops_transfers = self.add_transfer_time(
                    bag_round_stop, k, marked_stops_trips
                )

                marked_stops = set(marked_stops_trips).union(marked_stops_transfers)
            else:
                break

        logger.info("Finish round-based algorithm to create bag with best labels")
        logger.info(f"Running time: {perf_counter() - s}")

        return bag_round_stop, actual_rounds

    def accumulate_routes(self, marked_stops) -> List[Tuple[Route, Stop]]:
        """Accumulate routes serving marked stops from previous round, i.e. Q"""

        route_marked_stops = {}  # i.e. Q
        for marked_stop in marked_stops:
            routes_serving_stop = self.timetable.routes.get_routes_of_stop(marked_stop)

            for route in routes_serving_stop:
                # Check if new_stop is before existing stop in Q
                current_stop_for_route = route_marked_stops.get(route, None)  # p'
                if (current_stop_for_route is None) or (
                        route.stop_index(current_stop_for_route) > route.stop_index(marked_stop)
                ):
                    route_marked_stops[route] = marked_stop
        route_marked_stops = [(r, p) for r, p in route_marked_stops.items()]

        logger.debug(f"Found {len(route_marked_stops)} routes serving marked stops")

        return route_marked_stops

    def traverse_route(
            self,
            bag_round_stop: Dict[int, Dict[Stop, Bag]],
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> Tuple[Dict[int, Dict[Stop, Bag]], List[Stop]]:
        """
        Traverse through all marked route-stops and update labels accordingly.

        :param bag_round_stop: Bag per round per stop
        :param k: current round
        :param route_marked_stops: list of marked (route, stop) for evaluation
        """

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
                    # Get the arrival time of the trip at the current stop
                    trip_stop_time = label.trip.get_stop_time(current_stop)

                    # Take fare of previous stop in trip as fare is defined on start
                    previous_stop = remaining_stops_in_route[current_stop_idx - 1]
                    from_fare = label.trip.get_fare(previous_stop)

                    label = label.update(
                        earliest_arrival_time=trip_stop_time.dts_arr,
                        fare_addition=from_fare,
                    )

                    updated_labels.append(label)

                # Route bag B_{r} basically represents all the updated labels of
                #   the stops in the current marked route. Notably, each updated label means
                #   that you can board a trip with better characteristics
                #   (i.e. earlier arrival time, better fares, lesser number of trips)
                route_bag = Bag(labels=updated_labels)

                # Step 2: merge bag_route into bag_round_stop and remove dominated labels
                # NOTE: merging the current stop bag with route bag basically means to keep
                #   the labels that allow to get to the current stop in the most efficient way
                bag_round_stop[k][current_stop] = bag_round_stop[k][current_stop].merge(
                    route_bag
                )
                bag_update = bag_round_stop[k][current_stop].updated

                # Mark stop if bag is updated.
                # Updated bag means that the current stop brought some improvements
                if bag_update:
                    new_marked_stops.add(current_stop)

                # Step 3: merge B_{k-1}(p) into B_r
                # Merging here creates a bag where the best labels from the previous round,
                #   for the current stop, and the best from the current route are kept
                route_bag = route_bag.merge(bag_round_stop[k - 1][current_stop])

                # Assign trips to all newly added labels in route_bag
                # This is the trip on which we board
                # This step is needed because the labels from the previous round need
                #   to be updated with the new earliest boardable trip
                updated_labels = []
                for label in route_bag.labels:
                    earliest_trip = marked_route.earliest_trip(
                        label.earliest_arrival_time, current_stop
                    )
                    if earliest_trip is not None:
                        # Update label with the earliest trip in route leaving from this station
                        # If trip is different, we board the trip at current_stop
                        label = label.update_trip(earliest_trip, current_stop)
                        updated_labels.append(label)
                route_bag = Bag(labels=updated_labels)

        logger.debug(f"{len(new_marked_stops)} reachable stops added")

        return bag_round_stop, list(new_marked_stops)

    def add_transfer_time(
            self,
            bag_round_stop: Dict[int, Dict[Stop, Bag]],
            k: int,
            marked_stops: List[Stop],
    ) -> Tuple[Dict[int, Dict[Stop, Bag]], List[Stop]]:
        """
        Updates each stop (label) earliest arrival time by also considering (footpath) transfers between stops.

        :param bag_round_stop: dictionary of stop bags keyed by round
        :param k: current round
        :param marked_stops: list of the currently marked stops
        :return:
        """

        marked_stops_transfers = set()

        # Add in transfers to other platforms (same station) and stops
        for current_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)
            other_station_stops = [t.to_stop for t in self.timetable.transfers if t.from_stop == current_stop]

            for other_stop in other_station_stops:
                # Create temp copy of B_k(p_i)
                temp_bag = Bag()
                for label in bag_round_stop[k][current_stop].labels:
                    # Add arrival time to each label
                    transfer_arrival_time = (
                            label.earliest_arrival_time
                            + self.get_transfer_time(current_stop, other_stop)
                    )
                    # Update label with new earliest arrival time at other_stop
                    # NOTE: Each label contains the trip with which one arrives at the current stop
                    #   with k legs by boarding the trip at from_stop, along with the criteria
                    #   (i.e. boarding time, fares, number of legs)
                    label = label.update(
                        earliest_arrival_time=transfer_arrival_time,
                        fare_addition=0,
                        from_stop=current_stop,
                    )

                    # Update the trip with which to arrive at other_stop with a transfer trip
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=other_stop,
                        dep_time=label.earliest_arrival_time,
                        arr_time=transfer_arrival_time,

                        # TODO add method or field `transfer_type` to Transfer class
                        #  such accesser is then overrode by shared mobility Transfer sub-classes
                        transport_type=TransportType.Walk
                    )
                    label = label.update_trip(
                        trip=transfer_trip,
                        new_boarding_stop=current_stop
                    )
                    temp_bag.add(label)

                # Merge temp bag into B_k(p_j)
                bag_round_stop[k][other_stop] = bag_round_stop[k][other_stop].merge(
                    temp_bag
                )
                bag_update = bag_round_stop[k][other_stop].updated

                # Mark stop if bag is updated
                if bag_update:
                    marked_stops_transfers.add(other_stop)

        logger.debug(f"{len(marked_stops_transfers)} transferable stops added")

        return bag_round_stop, list(marked_stops_transfers)

    def get_transfer_time(self, stop_from: Stop, stop_to: Stop) -> int:
        """
        Calculate the transfer time from a stop to another stop (usually at one station)
        """
        transfers = self.timetable.transfers
        return transfers.stop_to_stop_idx[(stop_from, stop_to)].transfer_time


def best_legs_to_destination_station(
        to_stops: List[Stop], last_round_bag: Dict[Stop, Bag]
) -> List[Leg]:
    """
    Find the last legs to destination station that are reached by non-dominated labels.
    """

    # Find all labels to target_stops
    best_labels = [
        (stop, label) for stop in to_stops for label in last_round_bag[stop].labels
    ]

    # Pareto optimal labels
    pareto_optimal_labels: List[Label] = pareto_set([label for (_, label) in best_labels])
    pareto_optimal_labels: List[Tuple[Stop, Label]] = [
        (stop, label) for (stop, label) in best_labels if label in pareto_optimal_labels
    ]

    # Label to leg, i.e. add to_stop
    legs = [
        Leg(
            label.from_stop,
            to_stop,
            label.trip,
            label.earliest_arrival_time,
            label.fare,
            label.n_trips,
        )
        for to_stop, label in pareto_optimal_labels
    ]
    return legs


def reconstruct_journeys(
        from_stops: List[Stop],
        destination_legs: List[Leg],
        bag_round_stop: Dict[int, Dict[Stop, Bag]],
        k: int,
) -> List[Journey]:
    """
    Construct Journeys for destinations from bags by recursively
    looping from destination to origin.
    """

    def loop(
            bag_round_stop: Dict[int, Dict[Stop, Bag]], k: int, journeys: List[Journey]
    ):
        """Create full journey by prepending legs recursively"""

        last_round_bags = bag_round_stop[k]

        for jrny in journeys:
            current_leg = jrny[0]

            # End of journey if we are at origin stop or journey is not feasible
            if current_leg.trip is None or current_leg.from_stop in from_stops:
                jrny = jrny.remove_empty_legs()

                # Journey is valid if leg k ends before the start of leg k+1
                if jrny.is_valid() is True:
                    yield jrny
                continue

            # Loop trough each new leg. These are the legs that come before the current and that lead to from_stop
            labels_to_from_stop = last_round_bags[current_leg.from_stop].labels
            for new_label in labels_to_from_stop:
                new_leg = Leg(
                    new_label.from_stop,
                    current_leg.from_stop,
                    new_label.trip,
                    new_label.earliest_arrival_time,
                    new_label.fare,
                    new_label.n_trips,
                )
                # Only prepend new_leg if compatible before current leg, e.g. earlier arrival time, etc.
                if new_leg.is_compatible_before(current_leg):
                    new_jrny = jrny.prepend_leg(new_leg)
                    for i in loop(bag_round_stop, k, [new_jrny]):
                        yield i

    journeys = [Journey(legs=[leg]) for leg in destination_legs]
    journeys = [jrny for jrny in loop(bag_round_stop, k, journeys)]

    return journeys
