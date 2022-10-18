from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple, Set, Dict

from loguru import logger

from pyraptor.model.algos.base import BaseRaptor
from pyraptor.model.timetable import Transfers
from pyraptor.model.timetable import Stop, Route, TransferTrip
from pyraptor.model.criteria import EarliestArrivalTimeLabel, EarliestArrivalTimeBag, LabelUpdate
from pyraptor.util import LARGE_NUMBER


class EarliestArrivalTimeRaptor(BaseRaptor[EarliestArrivalTimeLabel, EarliestArrivalTimeBag]):
    """
    Implementation of the Earliest Arrival Time RAPTOR algorithm.
    Just a single criterion is optimized, that is, arrival time.
    """

    def _initialization(self, from_stops: Iterable[Stop], dep_secs: int) -> List[Stop]:
        # Initialize Round 0 with default labels.
        # Following rounds are initialized by copying the previous one
        self.round_stop_bags[0] = {}
        for p in self.timetable.stops:
            self.round_stop_bags[0][p] = EarliestArrivalTimeBag(
                labels=[EarliestArrivalTimeLabel(arrival_time=LARGE_NUMBER)]
            )

        # Initialize bags with starting stops taking dep_secs to reach
        # Remember that dep_secs is the departure_time expressed in seconds
        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")
        marked_stops = []
        for s in from_stops:
            departure_label = EarliestArrivalTimeLabel(arrival_time=dep_secs)
            self.round_stop_bags[0][s] = EarliestArrivalTimeBag(labels=[departure_label])

            marked_stops.append(s)

        self.stop_forward_dependencies: Dict[Stop, List[Stop]] = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = []

        return marked_stops

    def _traverse_routes(
            self,
            k: int,
            marked_route_stops: List[Tuple[Route, Stop]],
    ) -> List[Stop]:
        logger.debug(f"Traverse routes for round {k}")

        new_stops = set()
        n_evaluations = 0
        n_improvements = 0

        # For each route
        for marked_route, marked_stop in marked_route_stops:
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
                    best_arrival_time = self.round_stop_bags[k][
                        current_stop
                    ].get_label().arrival_time

                    if new_arrival_time < best_arrival_time:
                        # Update arrival by trip, i.e.
                        #   t_k(next_stop) = t_arr(t, pi)
                        #   t_star(p_i) = t_arr(t, pi)

                        # Update arrival stop with new arrival time
                        update_data = LabelUpdate(
                            boarding_stop=boarding_stop,
                            arrival_stop=current_stop,
                            new_trip=current_trip,
                            boarding_stop_label=self.round_stop_bags[k][boarding_stop].get_label()
                        )
                        self._update_arrival_stop(
                            k=k,
                            update_data=update_data,
                            currently_marked_stops=new_stops
                        )

                        # Logging
                        n_improvements += 1
                        new_stops.add(current_stop)

                # Can we catch an earlier trip at p_i
                # if tau_{k-1}(next_stop) <= tau_dep(t, next_stop)
                # NOTE: bag[k] is used, and not bag[k-1], because at this point they have the same value.
                #   bag[k] is (iirc) initialized with bag[k-1] values
                previous_earliest_arrival_time = self.round_stop_bags[k][
                    current_stop
                ].get_label().arrival_time
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
                    # This basically means that the boarding_stop will always be the earliest one,
                    #   in terms of arrival order in the current route
                    if earliest_trip_stop_time.trip != current_trip:
                        boarding_stop = current_stop

                    current_trip = earliest_trip_stop_time.trip

        logger.debug(f"- Evaluations    : {n_evaluations}")
        logger.debug(f"- Improvements   : {n_improvements}")

        return list(new_stops)

    def _improve_with_transfers(
            self,
            k: int,
            marked_stops: Iterable[Stop],
            transfers: Transfers
    ) -> List[Stop]:
        new_stops: Set[Stop] = set()

        # Add in transfers from the transfers table
        for current_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)
            other_station_stops = [
                t.to_stop for t in transfers if t.from_stop == current_stop
            ]

            time_sofar = self.round_stop_bags[k][current_stop].get_label().arrival_time
            for arrival_stop in other_station_stops:
                transfer = transfers.stop_to_stop_idx[(current_stop, arrival_stop)]

                arrival_time_with_transfer = time_sofar + transfer.transfer_time

                previous_earliest_arrival = self.round_stop_bags[k][
                    arrival_stop
                ].get_label().arrival_time

                # Domination criteria: update only if arrival time is improved
                if arrival_time_with_transfer < previous_earliest_arrival:
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=arrival_stop,
                        dep_time=time_sofar,
                        arr_time=arrival_time_with_transfer,
                        transport_type=transfer.transport_type
                    )

                    # Update the arrival stop
                    update_data = LabelUpdate(
                        boarding_stop=current_stop,
                        arrival_stop=arrival_stop,
                        new_trip=transfer_trip,
                        boarding_stop_label=self.round_stop_bags[k][current_stop].get_label()
                    )
                    self._update_arrival_stop(
                        k=k,
                        update_data=update_data,
                        currently_marked_stops=new_stops
                    )

                    new_stops.add(arrival_stop)

        return list(new_stops)

    def _update_arrival_stop(
            self,
            k: int,
            update_data: LabelUpdate,
            currently_marked_stops: Set[Stop]
    ):
        """
        Updates the label, along with the bags that store it,
        associated to the provided arrival stop.

        :param update_data: data to update the label with;
            it also identifies the arrival stop
        :param k: current round of the algorithm
        :param currently_marked_stops: stops currently marked
        """

        arrival_label = self.round_stop_bags[k][update_data.arrival_stop].get_label()
        arrival_label = arrival_label.update(
            data=update_data
        )

        self._update_stop(
            k=k,
            stop_to_update=update_data.arrival_stop,
            update_with=[arrival_label],
            currently_marked_stops=currently_marked_stops,
        )
