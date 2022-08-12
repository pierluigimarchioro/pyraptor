"""RAPTOR algorithm"""
from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple, Dict
from copy import deepcopy

from loguru import logger

from pyraptor.dao.timetable import Timetable
from pyraptor.model.structures import Stop, Route, Leg, Journey, TransferTrip, TransportType, Label, LabelUpdate, \
    ArrivalTimeCriterion
from pyraptor.util import LARGE_NUMBER


class RaptorAlgorithm:
    """RAPTOR Algorithm"""

    def __init__(self, timetable: Timetable):
        self.timetable: Timetable = timetable
        self.bag_round_stop: Dict[int, Dict[Stop, Label]] = {}
        self.bag_star: Dict[Stop, Label] = {}

    def run(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> Dict[int, Dict[Stop, Label]]:
        """
        Run Round-Based Algorithm

        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param rounds: total number of rounds to execute
        :return:
        """

        # Initialize empty bag of labels, i.e. B_k(p) = Label() for every k and p
        # Dictionary is keyed by round and contains another dictionary, where, for each stop, there
        # is a label representing the earliest arrival time (initialized at +inf)
        for k in range(0, rounds + 1):
            self.bag_round_stop[k] = {}
            for p in self.timetable.stops:
                self.bag_round_stop[k][p] = Label()

        # Initialize bag with the earliest arrival times
        # This bag is used as a side-collection to efficiently retrieve
        # the earliest arrival time for each reachable stop at round k.
        # Look for "local pruning" in the Microsoft paper for a better description.
        self.bag_star = {}
        for p in self.timetable.stops:
            self.bag_star[p] = Label()

        # Initialize bags with starting stops taking dep_secs to reach
        # Remember that dep_secs is the departure_time expressed in seconds
        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")
        marked_stops = []
        for from_stop in from_stops:
            departure_label = Label(earliest_arrival_time=dep_secs)

            self.bag_round_stop[0][from_stop] = departure_label
            self.bag_star[from_stop] = departure_label

            marked_stops.append(from_stop)

        # Run rounds
        for k in range(1, rounds + 1):
            logger.info(f"Analyzing possibilities round {k}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            self.bag_round_stop[k] = deepcopy(self.bag_round_stop[k - 1])

            # Get list of stops to evaluate in the process
            logger.debug(f"Stops to evaluate count: {len(marked_stops)}")

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            route_marked_stops = self.accumulate_routes(marked_stops)

            # Update stop arrival times calculated basing on reachable stops
            marked_trip_stops = self.traverse_routes(
                k, route_marked_stops
            )
            logger.debug(f"{len(marked_trip_stops)} reachable stops added")

            # Add footpath transfers and update
            marked_transfer_stops = self.add_transfer_time(
                k, marked_trip_stops
            )
            logger.debug(f"{len(marked_transfer_stops)} transferable stops added")

            marked_stops = set(marked_trip_stops).union(marked_transfer_stops)
            logger.debug(f"{len(marked_stops)} stops to evaluate in next round")

        return self.bag_round_stop

    def accumulate_routes(self, marked_stops: List[Stop]) -> List[Tuple[Route, Stop]]:
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

    def traverse_routes(
            self,
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        """
        Iterate through the stops reachable and add all new reachable stops
        by following all trips from the reached stations. Trips are only followed
        in the direction of travel and beyond already added points.

        :param k: current round
        :param route_marked_stops: list of marked (route, stop) for evaluation
        """
        logger.debug(f"Traverse routes for round {k}")

        new_stops = []
        n_evaluations = 0
        n_improvements = 0

        # For each route
        for marked_route, marked_stop in route_marked_stops:

            # Current trip for this marked stop
            current_trip = None

            # Iterate over all stops after current stop within the current route
            current_stop_index = marked_route.stop_index(marked_stop)
            remaining_stops_in_route = marked_route.stops[current_stop_index:]
            boarding_stop = None

            for current_stop_index, current_stop in enumerate(remaining_stops_in_route):
                # Can the label be improved in this round?
                n_evaluations += 1

                # t != _|_
                if current_trip is not None:
                    # Arrival time at stop, i.e. arr(current_trip, next_stop)
                    new_arrival_time = current_trip.get_stop_time(current_stop).dts_arr
                    best_arrival_time = self.bag_star[
                        current_stop
                    ].earliest_arrival_time

                    if new_arrival_time < best_arrival_time:
                        # Update arrival by trip, i.e.
                        #   t_k(next_stop) = t_arr(t, pi)
                        #   t_star(p_i) = t_arr(t, pi)

                        arrival_label = self.bag_round_stop[k][current_stop]
                        update_data = LabelUpdate(
                            boarding_stop=boarding_stop,
                            arrival_stop=current_stop,
                            old_trip=arrival_label.trip,
                            new_trip=current_trip,
                            best_labels=self.bag_star
                        )
                        self.update_arrival_label(update_data=update_data, k=k)

                        # Logging
                        n_improvements += 1
                        new_stops.append(current_stop)

                # Can we catch an earlier trip at p_i
                # if tau_{k-1}(next_stop) <= tau_dep(t, next_stop)
                # TODO why bag[k] and not bag[k-1]? Try putting k-1 and see what happens
                previous_earliest_arrival_time = self.bag_round_stop[k][
                    current_stop
                ].earliest_arrival_time
                earliest_trip_stop_time = marked_route.earliest_trip_stop_time(
                    previous_earliest_arrival_time, current_stop
                )
                if (
                        earliest_trip_stop_time is not None
                        and previous_earliest_arrival_time
                        <= earliest_trip_stop_time.dts_dep
                ):
                    current_trip = earliest_trip_stop_time.trip
                    boarding_stop = current_stop

        logger.debug(f"- Evaluations    : {n_evaluations}")
        logger.debug(f"- Improvements   : {n_improvements}")

        return new_stops

    def add_transfer_time(
            self,
            k: int,
            marked_stops: List[Stop],
    ) -> List[Stop]:
        """
        Add transfers between platforms.

        :param k: current round
        :param marked_stops: list of marked stops for evaluation
        """

        new_stops: List[Stop] = []

        # Add in transfers from the transfers table
        for current_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)
            other_station_stops = [
                t.to_stop for t in self.timetable.transfers if t.from_stop == current_stop
            ]

            time_sofar = self.bag_round_stop[k][current_stop].earliest_arrival_time
            for arrival_stop in other_station_stops:
                arrival_time_with_transfer = time_sofar + self.get_transfer_time(
                    current_stop, arrival_stop
                )
                previous_earliest_arrival = self.bag_star[
                    arrival_stop
                ].earliest_arrival_time

                # Domination criteria
                if arrival_time_with_transfer < previous_earliest_arrival:
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=arrival_stop,
                        dep_time=time_sofar,
                        arr_time=arrival_time_with_transfer,

                        # TODO add method or field `transfer_type` to Transfer class
                        #  such accesser is then overrode by shared mobility Transfer sub-classes
                        transport_type=TransportType.Walk
                    )

                    # Update the label
                    arrival_label = self.bag_round_stop[k][arrival_stop]
                    update_data = LabelUpdate(
                        boarding_stop=current_stop,
                        arrival_stop=arrival_stop,
                        old_trip=arrival_label.trip,
                        new_trip=transfer_trip,
                        best_labels=self.bag_star
                    )
                    self.update_arrival_label(update_data=update_data, k=k)

                    new_stops.append(arrival_stop)

        return new_stops

    def get_transfer_time(self, stop_from: Stop, stop_to: Stop) -> int:
        """
        Calculate the transfer time from a stop to another stop (usually at one station)
        """
        transfers = self.timetable.transfers
        return transfers.stop_to_stop_idx[(stop_from, stop_to)].transfer_time

    def update_arrival_label(self, update_data: LabelUpdate, k: int):
        """
        Updates the label, along with the bag that store it, for the provided arrival stop
        :param update_data: data to update the label with
        :param k: current round of the algorithm
        :return:
        """

        arrival_label = self.bag_round_stop[k][update_data.arrival_stop]
        arrival_label = arrival_label.update(
            data=update_data
        )

        # Assign the updated label to the bags
        self.bag_round_stop[k][update_data.arrival_stop] = arrival_label
        self.bag_star[update_data.arrival_stop] = arrival_label


def best_stop_at_target_station(to_stops: List[Stop], bag: Dict[Stop, Label]) -> Stop:
    """
    Find the destination Stop with the shortest distance.
    Required in order to prevent adding travel time to the arrival time.
    """
    final_stop = 0
    distance = LARGE_NUMBER
    for stop in to_stops:
        if bag[stop].earliest_arrival_time < distance:
            distance = bag[stop].earliest_arrival_time
            final_stop = stop
    return final_stop


def reconstruct_journey(destination: Stop, bag: Dict[Stop, Label]) -> Journey:
    """Construct journey for destination from values in bag."""

    # Create journey with list of legs
    jrny = Journey()
    to_stop = destination
    while to_stop is not None:
        from_stop = bag[to_stop].boarding_stop
        to_stop_label = bag[to_stop]

        # The following are all default values except the raw value.
        # The important is giving the leg object information
        #   about arrival time, which allows it to check if legs are
        #   compatible and to store correct information about the journey
        # TODO using Criterion classes here feels out of place. what to do?
        arrival_time_crit = ArrivalTimeCriterion(
            name="arrival_time",
            weight=1,
            raw_value=to_stop_label.earliest_arrival_time,
            upper_bound=LARGE_NUMBER
        )
        leg = Leg(
            from_stop=from_stop,
            to_stop=to_stop,
            trip=to_stop_label.trip,
            criteria=[arrival_time_crit]
        )
        jrny = jrny.prepend_leg(leg)
        to_stop = from_stop

    jrny = jrny.remove_empty_and_same_station_legs()

    return jrny


def is_dominated(original_journey: List[Leg], new_journey: List[Leg]) -> bool:
    """Check if new journey is dominated by another journey"""
    # None if first journey
    if not original_journey:
        return False

    # No improvement
    if original_journey == new_journey:
        return True

    def depart(jrny: List[Leg]) -> int:
        depart_leg = jrny[0] if jrny[0].trip is not None else jrny[1]
        return depart_leg.dep

    def arrival(jrny: List[Leg]) -> int:
        return jrny[-1].arr

    original_depart = depart(original_journey)
    new_depart = depart(new_journey)

    original_arrival = arrival(original_journey)
    new_arrival = arrival(new_journey)

    # Is dominated, strictly better in one criterion and not worse in other
    return (
        True
        if (original_depart >= new_depart and original_arrival < new_arrival)
           or (original_depart > new_depart and original_arrival <= new_arrival)
        else False
    )
