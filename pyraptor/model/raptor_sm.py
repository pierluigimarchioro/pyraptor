"""RAPTOR algorithm"""
from __future__ import annotations

import itertools
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy

from loguru import logger
from numpy import argmin

from pyraptor.dao.timetable import Timetable
from pyraptor.model.structures import Stop, Trip, Route, Leg, Journey, SharedMobilityFeed, SharedMobilityVehicleType, \
    SharedMobilityPhysicalStation, SharedMobilityTransfer, Transfer
from pyraptor.util import LARGE_NUMBER


@dataclass
class Label:
    """Label"""

    earliest_arrival_time: int = LARGE_NUMBER
    trip: Trip = None  # trip to take to obtain earliest_arrival_time
    from_stop: Stop = None  # stop at which we hop-on trip with trip

    def update(self, earliest_arrival_time=None, trip=None, from_stop=None):
        """Update"""
        if earliest_arrival_time is not None:
            self.earliest_arrival_time = earliest_arrival_time
        if trip is not None:
            self.trip = trip
        if from_stop is not None:
            self.from_stop = from_stop

    def is_dominating(self, other: Label):
        """Dominates other label"""
        return self.earliest_arrival_time <= other.earliest_arrival_time

    def __repr__(self) -> str:
        return f"Label(earliest_arrival_time={self.earliest_arrival_time}, trip={self.trip}, from_stop={self.from_stop})"


class RaptorAlgorithmSharedMobility:
    """RAPTOR Algorithm Shared Mobility"""

    def __init__(self, timetable: Timetable, shared_mob: SharedMobilityFeed):
        self.timetable = timetable
        self.shared_mob = shared_mob
        self.bag_star = None
        self.vtype: SharedMobilityVehicleType = self.shared_mob.vtype
        self.no_source: List[Stop] = []
        self.no_dest: List[Stop] = []

    def _update_stop_info(self):
        """Updates object fields analyzing stops availability"""
        self.no_source: List = self.shared_mob.stops_no_source
        self.no_dest: List = self.shared_mob.stops_no_dest

    def _add_shared_mob_transfer(self, stop_a: SharedMobilityPhysicalStation, stop_b: SharedMobilityPhysicalStation):
        """ Given two stop add transfer to timetable depending on availability """

        t_dir, t_opp = SharedMobilityTransfer.get_shared_mob_transfer(stop_a, stop_b, self.vtype)

        if stop_a not in self.no_source and stop_b not in self.no_dest:
            self.timetable.transfers.add(t_dir)

        if stop_b not in self.no_source and stop_a not in self.no_dest:
            self.timetable.transfers.add(t_opp)

    def _append_shared_mob_stop(self, stop: SharedMobilityPhysicalStation,
                                shared_mob_stops: List[SharedMobilityPhysicalStation]) -> SharedMobilityPhysicalStation | None:
        """ If stop is not already computed, all transfers between it and all others share-mob stops are added; stop is returned as output.
            If stop was already computed, None is returned """
        if stop not in shared_mob_stops:  # skip stops already computed
            for marked in shared_mob_stops:
                self._add_shared_mob_transfer(stop, marked)
            shared_mob_stops.append(stop)
            return stop
        else:
            return None

    def _get_immediate_transferable(self, from_stops: List[Stop]) -> Dict[Stop, Transfer]:

        # immediate_transfers: transfers starting from_stop
        immediate_transfers: List[Transfer] = [t for t in self.timetable.transfers if
                                                t.from_stop in from_stops]

        # immediate_to_stop_time: dictionary keyed by all reachable stops, values are lists of associated transfers
        earliest_transfer: Dict[Stop, List[Transfer] | Transfer] = \
            {s: [t for t in immediate_transfers if t.to_stop == s]
             for s in set([t.to_stop for t in immediate_transfers])}

        # selecting earliest transfer
        for stop, transfers in earliest_transfer.items():
            min_idx = argmin([t.transfer_time for t in transfers])  # index of minimum transfer time
            earliest_transfer[stop] = transfers[min_idx]

        return earliest_transfer

    def run(self, from_stops, dep_secs, rounds) -> Dict[int, Dict[Stop, Label]]:
        """
        Run Round-Based Algorithm
        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param rounds: total number of rounds to execute
        :return:
        """

        # Downloading information about shared-mob stops availability
        self._update_stop_info()
        logger.debug(f"Shared mobility vehicle: {self.vtype.value}")
        logger.debug(f"No {len(self.no_source)} shared-mob stops available as source: {self.no_source} ")
        logger.debug(f"No {len(self.no_dest)} shared-mob stops available as destination: {self.no_dest} ")

        # Initialize empty bag of labels, i.e. B_k(p) = Label() for every k and p
        # Dictionary is keyed by round and contains another dictionary, where, for each stop, there
        # is a label representing the earliest arrival time (initialized at +inf)
        bag_round_stop: Dict[int, Dict[Stop, Label]] = {}
        for k in range(0, rounds + 1):
            bag_round_stop[k] = {}
            for p in self.timetable.stops:
                bag_round_stop[k][p] = Label()

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
        marked_stops: List[Stop] = []
        shared_mob_stops_known: List[SharedMobilityPhysicalStation] = []  # contains all known shared-mob stops
        for from_stop in from_stops:
            bag_round_stop[0][from_stop].update(dep_secs, None, None)
            self.bag_star[from_stop].update(dep_secs, None, None)
            marked_stops.append(from_stop)
            # Adding to know shared-mob if present in starting stops
            if isinstance(from_stop, SharedMobilityPhysicalStation):
                self._append_shared_mob_stop(from_stop, shared_mob_stops_known)

        s1 = len(marked_stops) # debugging
        sm1 = len(shared_mob_stops_known) # debugging
        logger.debug(f"Starting from {s1} stops ({sm1} are shared-mob)")

        # IMMEDIATE TRANSFERS: adding to initial stops transferable stops

        # immediate_transfers: transfers starting from_stop

        immediate = self._get_immediate_transferable(marked_stops)

        for stop, t in immediate.items():
            if stop not in marked_stops: # avoid re-add already existings stops
                bag_round_stop[0][stop].update(dep_secs + t.transfer_time, None, t.from_stop)  # TODO: which trip?
                self.bag_star[stop].update(dep_secs + t.transfer_time, None, t.from_stop)  # TODO: which trip?
                marked_stops.append(stop)
                # check if it is also a share-mob stop
                if isinstance(stop, SharedMobilityPhysicalStation):
                    self._append_shared_mob_stop(stop, shared_mob_stops_known)

        s2 = len(marked_stops)  # debugging
        sm2 = len(shared_mob_stops_known)  # debugging
        logger.debug(f"Added {s2-s1} immediate stops, ({sm2-sm1} are shared-mob)")

        # Run rounds
        for k in range(1, rounds + 1):
            logger.info(f"Analyzing possibilities round {k}")

            # Initialize round k (current) with the labels of round k-1 (previous)
            bag_round_stop[k] = deepcopy(bag_round_stop[k - 1])

            # Get list of stops to evaluate in the process
            logger.debug(f"Stops to evaluate count: {len(marked_stops)}")

            # Get (route, marked stop) pairs, where marked stop
            # is the first reachable stop of the route
            route_marked_stops = self.accumulate_routes(marked_stops)

            # Update stop arrival times calculated basing on reachable stops
            bag_round_stop, marked_trip_stops = self.traverse_routes(
                bag_round_stop, k, route_marked_stops
            )
            logger.debug(f"{len(marked_trip_stops)} reachable stops added")

            # Add footpath transfers and update
            bag_round_stop, marked_transfer_stops = self.add_transfer_time(
                bag_round_stop, k, marked_trip_stops
            )
            logger.debug(f"{len(marked_transfer_stops)} transferable stops added")

            print('here breakpoint')

            """
            Part 4
            In marked_transfer_stops there are possible SharedMobility Stations:
            we know we can reach a public stop with a trip and then a close share-mob stop with a transfer
            Those stops are filtered            
            """
            # filtering:
            shared_mob_reached_with_trips: List[SharedMobilityPhysicalStation] = [s for s in marked_transfer_stops if isinstance(s, SharedMobilityPhysicalStation)]
            logger.debug(f"{len(shared_mob_reached_with_trips)} shared-mob stops reachable ")

            # if new, adding to know shared mob stops list; also new transfers are added
            t1 = len(self.timetable.transfers)  # debugging
            shared_mob_reached_with_trips: List[SharedMobilityPhysicalStation] = \
                [s_ for s_ in [self._append_shared_mob_stop(s, shared_mob_stops_known)
                               for s in shared_mob_reached_with_trips] if s_ is not None]
            t2 = len(self.timetable.transfers)  # debugging
            logger.debug(f"New {len(shared_mob_reached_with_trips)} shared-mob stops reachable ")
            logger.debug(f"New {t2 - t1} shared-mob transfers created")

            """
            Know we need to update all transfers time in sub-shared mobility network:
                - just SharedMobilityTransfers
                - just Transfers which to_stop is in shared_mob_reached_with_trips
                - 
            """
            # List of share-mob transfers arriving to reachable share-mob stops
            transfers_sub: List[SharedMobilityTransfer] = \
                [t for t in self.timetable.transfers
                if isinstance(t, SharedMobilityTransfer) and t.to_stop in shared_mob_reached_with_trips]
            # List of shared-mob stops starting from a transfer_sub
            shared_mob_stops_known_sub: List[SharedMobilityPhysicalStation] = \
                list(set([t.from_stop for t in transfers_sub]))

            """
            Know we add these transfers
            """
            bag_round_stop, shared_mob_reached_with_trips_improved = self.add_transfer_time(
                bag_round_stop, k, shared_mob_stops_known_sub, transfers_sub
            )
            logger.debug(f"{len(shared_mob_reached_with_trips_improved)} transferable share-mob stops upgraded")

            """
            Part 5
            Know we have some share-mob stop updated, so starting from these we update all transferable public stops
            """
            bag_round_stop, marked_shared_mob_stops = self.add_transfer_time(
                bag_round_stop, k, shared_mob_reached_with_trips_improved
            )
            logger.debug(f"{len(marked_shared_mob_stops)} using shared-mobility stops upgraded")

            marked_stops = set(marked_trip_stops).union(marked_transfer_stops, marked_shared_mob_stops)
            logger.debug(f"{len(marked_stops)} stops to evaluate in next round")

        return bag_round_stop

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
            bag_round_stop: Dict[int, Dict[Stop, Label]],
            k: int,
            route_marked_stops: List[Tuple[Route, Stop]],
    ) -> Tuple:
        """
        Iterate through the stops reachable and add all new reachable stops
        by following all trips from the reached stations. Trips are only followed
        in the direction of travel and beyond already added points.
        :param bag_round_stop: Bag per round per stop
        :param k: current round
        :param route_marked_stops: list of marked (route, stop) for evaluation
        """
        logger.debug(f"Traverse routes for round {k}")

        bag_round_stop = deepcopy(bag_round_stop)
        new_stops = []
        n_evaluations = 0
        n_improvements = 0

        # For each route
        for (marked_route, marked_stop) in route_marked_stops:

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
                    new_arrival_time = current_trip.get_stop(current_stop).dts_arr
                    best_arrival_time = self.bag_star[
                        current_stop
                    ].earliest_arrival_time

                    if new_arrival_time < best_arrival_time:
                        # Update arrival by trip, i.e.
                        #   t_k(next_stop) = t_arr(t, pi)
                        #   t_star(p_i) = t_arr(t, pi)

                        bag_round_stop[k][current_stop].update(
                            new_arrival_time, current_trip, boarding_stop
                        )
                        self.bag_star[current_stop].update(
                            new_arrival_time, current_trip, boarding_stop
                        )

                        # Logging
                        n_improvements += 1
                        new_stops.append(current_stop)

                # Can we catch an earlier trip at p_i
                # if tau_{k-1}(next_stop) <= tau_dep(t, next_stop)
                # TODO why bag[k] and not bag[k-1]? Try putting k-1 and see what happens
                previous_earliest_arrival_time = bag_round_stop[k][
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

        return bag_round_stop, new_stops

    def add_transfer_time(
            self,
            bag_round_stop: Dict[int, Dict[Stop, Label]],
            k: int,
            marked_stops: List[Stop],
            transfer_sub: List[Transfer] | None = None
    ) -> Tuple:
        """
        Add transfers between platforms.
        :param bag_round_stop: Label per round per stop
        :param k: current round
        :param marked_stops: list of marked stops for evaluation
        :param transfer_sub: used for reduce transfer sample
        """

        new_stops = []

        # Add in transfers from the transfers table
        for current_stop in marked_stops:
            # Note: transfers are transitive, which means that for each reachable stops (a, b) there
            # is transfer (a, b) as well as (b, a)

            transfers = self.timetable.transfers if transfer_sub is None else transfer_sub
            other_station_stops = [
                t.to_stop for t in transfers if t.from_stop == current_stop
            ]

            time_sofar = bag_round_stop[k][current_stop].earliest_arrival_time
            for arrive_stop in other_station_stops:
                arrival_time_with_transfer = time_sofar + self.get_transfer_time(
                    current_stop, arrive_stop
                )
                previous_earliest_arrival = self.bag_star[
                    arrive_stop
                ].earliest_arrival_time

                # Domination criteria
                if arrival_time_with_transfer < previous_earliest_arrival:
                    transfer_trip = Trip.get_transfer_trip(from_stop=current_stop,
                                                           to_stop=arrive_stop,
                                                           dep_time=time_sofar,
                                                           arr_time=arrival_time_with_transfer)

                    bag_round_stop[k][arrive_stop].update(
                        arrival_time_with_transfer,
                        transfer_trip,
                        current_stop,
                    )
                    self.bag_star[arrive_stop].update(
                        arrival_time_with_transfer, transfer_trip, current_stop
                    )
                    new_stops.append(arrive_stop)

        return bag_round_stop, new_stops

    def get_transfer_time(self, stop_from: Stop, stop_to: Stop) -> int:
        """
        Calculate the transfer time from a stop to another stop (usually at one station)
        """
        transfers = self.timetable.transfers
        return transfers.stop_to_stop_idx[(stop_from, stop_to)].transfer_time


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
        from_stop = bag[to_stop].from_stop
        bag_to_stop = bag[to_stop]

        leg = Leg(
            from_stop, to_stop, bag_to_stop.trip, bag_to_stop.earliest_arrival_time
        )
        jrny = jrny.prepend_leg(leg)
        to_stop = from_stop

    jrny = jrny.remove_empty_legs()

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

    # Is dominated, strictly better in one criteria and not worse in other
    return (
        True
        if (original_depart >= new_depart and original_arrival < new_arrival)
           or (original_depart > new_depart and original_arrival <= new_arrival)
        else False
    )