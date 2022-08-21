from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import List, Tuple

from loguru import logger

from pyraptor.model.algos.base import BaseSMRaptor
from pyraptor.model.timetable import Transfer
from pyraptor.model.timetable import Stop, Route, TransferTrip
from pyraptor.model.criteria import BasicRaptorLabel, LabelUpdate, MultiCriteriaLabel
from pyraptor.model.output import Leg, Journey
from pyraptor.util import LARGE_NUMBER


class RaptorAlgorithm(BaseSMRaptor[BasicRaptorLabel, BasicRaptorLabel]):
    """
    Implementation of the basic RAPTOR algorithm, with some improvements:
    - transfers from the origin stops are evaluated immediately to widen
        the set of reachable stops before the first round is executed
    - it is possible to include shared mobility, real-time data in the computation
    """

    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> List[Stop]:
        # Initialize empty bag of labels for each stop
        for k in range(0, rounds + 1):
            self.bag_round_stop[k] = {}
            for s in self.timetable.stops:
                self.bag_round_stop[k][s] = BasicRaptorLabel()

        # Initialize bag with the earliest arrival times
        # This bag is used as a side-collection to efficiently retrieve
        # the earliest arrival time for each reachable stop at round k.
        # Look for "local pruning" in the Microsoft paper for a better description.
        self.bag_star = {}
        for s in self.timetable.stops:
            self.bag_star[s] = BasicRaptorLabel()

        # Initialize bags with starting stops taking dep_secs to reach
        # Remember that dep_secs is the departure_time expressed in seconds
        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")
        marked_stops = []
        for s in from_stops:
            departure_label = BasicRaptorLabel(earliest_arrival_time=dep_secs)

            self.bag_round_stop[0][s] = departure_label
            self.bag_star[s] = departure_label

            marked_stops.append(s)

        return marked_stops

    def _traverse_routes(
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
                        self._update_arrival_label(update_data=update_data, k=k)

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

    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Iterable[Transfer]
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
                t.to_stop for t in transfers if t.from_stop == current_stop
            ]

            time_sofar = self.bag_round_stop[k][current_stop].earliest_arrival_time
            for arrival_stop in other_station_stops:
                transfer = self._get_transfer(current_stop, arrival_stop)

                arrival_time_with_transfer = time_sofar + transfer.transfer_time
                previous_earliest_arrival = self.bag_star[
                    arrival_stop
                ].earliest_arrival_time

                # Domination criteria: update only if arrival time is improved
                if arrival_time_with_transfer < previous_earliest_arrival:
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=arrival_stop,
                        dep_time=time_sofar,
                        arr_time=arrival_time_with_transfer,
                        transport_type=transfer.transport_type
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
                    self._update_arrival_label(update_data=update_data, k=k)

                    new_stops.append(arrival_stop)

        return new_stops

    def _update_arrival_label(self, update_data: LabelUpdate, k: int):
        """
        Updates the label, along with the bag that stores it, for the provided arrival stop

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


# TODO remove because already in query.py or move to output.py?
def best_stop_at_target_station(to_stops: List[Stop], bag: Mapping[Stop, BasicRaptorLabel]) -> Stop:
    """
    Find the destination Stop with the shortest distance.
    Required in order to prevent adding travel time to the arrival time.
    """

    final_stop = None
    distance = LARGE_NUMBER
    for stop in to_stops:
        if bag[stop].earliest_arrival_time < distance:
            distance = bag[stop].earliest_arrival_time
            final_stop = stop

    return final_stop


# TODO remove because already in query.py or move to output.py?
def reconstruct_journey(destination: Stop, bag: Mapping[Stop, BasicRaptorLabel]) -> Journey:
    """Construct journey for destination from values in bag."""

    # Create journey with list of legs
    jrny = Journey()
    to_stop = destination
    while to_stop is not None:
        from_stop = bag[to_stop].boarding_stop
        to_stop_label = bag[to_stop]

        # Convert to multi-criteria label to create criteria instances from the base one
        mc_label = MultiCriteriaLabel.from_base_raptor_label(label=to_stop_label)
        leg = Leg(
            from_stop=from_stop,
            to_stop=to_stop,
            trip=mc_label.trip,
            criteria=mc_label.criteria
        )
        jrny = jrny.prepend_leg(leg)
        to_stop = from_stop

    jrny = jrny.remove_empty_legs()

    return jrny


# TODO move to output.py?
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
