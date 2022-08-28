from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple

from loguru import logger

from pyraptor.model.algos.base import BaseSharedMobRaptor
from pyraptor.model.timetable import Transfers
from pyraptor.model.timetable import Stop, Route, TransferTrip
from pyraptor.model.criteria import BasicRaptorLabel, LabelUpdate


class RaptorAlgorithm(BaseSharedMobRaptor[BasicRaptorLabel, BasicRaptorLabel]):
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
        self.best_bag = {}
        for s in self.timetable.stops:
            self.best_bag[s] = BasicRaptorLabel()

        # Initialize bags with starting stops taking dep_secs to reach
        # Remember that dep_secs is the departure_time expressed in seconds
        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")
        marked_stops = []
        for s in from_stops:
            departure_label = BasicRaptorLabel(earliest_arrival_time=dep_secs)

            self.bag_round_stop[0][s] = departure_label
            self.best_bag[s] = departure_label

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
                            best_labels=self.best_bag
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
                    # If the trip is different from the previous one, we board the trip
                    #   at current_stop, else the trip is still boarded at the old boarding stop.
                    # This basically results in consecutive stops of the same journey
                    #   only pointing to the first stop of each different trip in the journey,
                    #   as opposed to pointing to each intermediate stop of each different trip.
                    if earliest_trip_stop_time.trip != current_trip:
                        boarding_stop = current_stop

                    current_trip = earliest_trip_stop_time.trip

        logger.debug(f"- Evaluations    : {n_evaluations}")
        logger.debug(f"- Improvements   : {n_improvements}")

        return new_stops

    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Transfers
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
                transfer = transfers.stop_to_stop_idx[(current_stop, arrival_stop)]

                arrival_time_with_transfer = time_sofar + transfer.transfer_time
                previous_earliest_arrival = self.best_bag[
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
        self.best_bag[update_data.arrival_stop] = arrival_label
