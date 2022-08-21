"""RAPTOR algorithm"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict

from loguru import logger
from numpy import argmin

import pyraptor.model.algos.raptor as raptor
from pyraptor.dao.timetable import RaptorTimetable
from pyraptor.model.timetable import Stop, Trip, Route, Transfer, TransferTrip, TransportType
from pyraptor.model.shared_mobility import (
    SharedMobilityFeed,
    RentingStation,
    VehicleTransfer,
    VehicleTransfers,
    filter_shared_mobility, VEHICLE_SPEED
)
from pyraptor.model.output import Leg, Journey
from pyraptor.util import LARGE_NUMBER


# TODO use Label from criteria.py module
#   fast way might be to just move the required code from here to the base RAPTOR class
#   that already uses criteria.py data structures, and then delete this module
@dataclass
class Label:
    """Label"""

    earliest_arrival_time: int = LARGE_NUMBER
    trip: Trip = None  # trip to take to obtain earliest_arrival_time
    boarding_stop: Stop = None  # stop at which we hop-on trip with trip

    def update(self, earliest_arrival_time=None, trip=None, from_stop=None):
        """Update"""
        if earliest_arrival_time is not None:
            self.earliest_arrival_time = earliest_arrival_time
        if trip is not None:
            self.trip = trip
        if from_stop is not None:
            self.boarding_stop = from_stop

    def is_dominating(self, other: Label):
        """Dominates other label"""
        return self.earliest_arrival_time <= other.earliest_arrival_time

    def __repr__(self) -> str:
        return f"Label(earliest_arrival_time={self.earliest_arrival_time}, trip={self.trip}, from_stop={self.boarding_stop})"


class RaptorAlgorithmSharedMobility:
    """RAPTOR Algorithm Shared Mobility"""

    def __init__(self,
                 timetable: RaptorTimetable,
                 shared_mobility_feeds: Iterable[SharedMobilityFeed],
                 preferred_vehicle: TransportType,
                 use_car: bool):
        self.timetable: RaptorTimetable = timetable
        self.shared_mobility_feeds: Iterable[SharedMobilityFeed] = shared_mobility_feeds
        self.preferred_vehicle: TransportType = preferred_vehicle
        self.use_car = use_car

        self.bag_star: Dict[Stop, Label] = {}

        self.no_source: List[RentingStation] = []
        """list of renting stations not available as source"""

        self.no_dest: List[RentingStation] = []
        """list of renting stations not available as destination"""

        self.v_transfers: VehicleTransfers = VehicleTransfers()
        """list of vehicle transfers build while computing"""

    def _add_vehicle_transfer(self, stop_a: RentingStation, stop_b: RentingStation):
        """ Given two stop adds associated outdoor vehicle-transfer
            to a vehicles transfers depending on availability and system belongings

            If stops have common available multiple vehicles:
             * uses preferred vehicle if present,
             * otherwise uses an other vehicle TODO more option criteria
        """

        # 1) they are part of same system
        if stop_a.system_id == stop_b.system_id:
            # 2.a) evaluating common transport type
            common_ttypes: List[TransportType] = list(set(stop_a.transport_types).intersection(stop_b.transport_types))
            # 2.b) removed car transfer if disabled
            if not self.use_car:
                car_ttype: TransportType.Car = TransportType(9001)  # Car
                common_ttypes = list(set(common_ttypes).difference([car_ttype]))
            # 2.c) create a vehicle transfer if at least one common transport type found
            if len(common_ttypes) > 0:
                # 3.a) if preferred vehicle is present, transfer is generated
                if self.preferred_vehicle in common_ttypes:
                    ttype = self.preferred_vehicle
                # 3.b) other fatest one is choshen # TODO different possible criteria
                else:
                    ind = argmin([VEHICLE_SPEED[ttype] for ttype in common_ttypes])
                    ttype = common_ttypes[ind]
                # 4) creating  transfer
                t_out, _ = VehicleTransfer.get_vehicle_transfer(stop_a, stop_b, ttype)

                # 5) compatibility to real-time availability
                if stop_a not in self.no_source and stop_b not in self.no_dest:
                    self.v_transfers.add(t_out)

    def _update_availability_info(self):
        """ Updates stops availability based on real-time query
            Also clears all vehicle transfers computed """

        for feed in self.shared_mobility_feeds:
            feed.renting_stations.update()

        no_source_: List[List[RentingStation]] = [feed.renting_stations.no_source for feed in
                                                  self.shared_mobility_feeds]
        no_dest_: List[List[RentingStation]] = [feed.renting_stations.no_destination for feed in
                                                self.shared_mobility_feeds]

        self.no_source: List[RentingStation] = [i for sub in no_source_ for i in sub]  # flatten
        self.no_dest: List[RentingStation] = [i for sub in no_dest_ for i in sub]  # flatten

        self.v_transfers = VehicleTransfers()

    def _get_immediate_transferable(self, from_stops: List[Stop]) -> Dict[Stop, Transfer]:
        """ Given a list of stops returns all transferable stops associated with transfer.
            If multiple transfers to a destination are available, the earliest one is chosen """

        # List of immediate transfers (starting from from_stops)
        immediate_transfers: List[Transfer] = [t for t in self.timetable.transfers if t.from_stop in from_stops]

        # Dictionary {reachable stop : list of transfers to reach it}
        earliest_transfer: Dict[Stop, List[Transfer] | Transfer] = \
            {s: [t for t in immediate_transfers if t.to_stop == s]
             for s in set([t.to_stop for t in immediate_transfers])}

        # Identifying the earliest transfer from the list
        for stop, transfers in earliest_transfer.items():
            min_idx = argmin([t.transfer_time for t in transfers])  # index of the earliest transfer time
            earliest_transfer[stop] = transfers[min_idx]

        return earliest_transfer

    def run(self, from_stops: Iterable[Stop], dep_secs: int, rounds: int) -> Dict[int, Dict[Stop, Label]]:
        """
        Run Round-Based Algorithm with 2 optimization for shared-mobility integration
        :param from_stops: collection of stops to depart from
        :param dep_secs: departure time in seconds from midnight
        :param rounds: total number of rounds to execute
        :return:
        """

        # Downloading information about shared-mob stops availability
        self._update_availability_info()
        sm_feeds_info = [f'{feed.system_id} ({[t.name for t in feed.transport_types]})' for feed in self.shared_mobility_feeds]
        logger.debug(f"Shared mobility feeds: {sm_feeds_info} ")
        logger.debug(f"{len(self.no_source)} shared-mob stops not available as source: {self.no_source} ")
        logger.debug(f"{len(self.no_dest)} shared-mob stops not available as destination: {self.no_dest} ")

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
        renting_stations_known: List[RentingStation] = []
        for from_stop in from_stops:
            bag_round_stop[0][from_stop].update(dep_secs, None, None)
            self.bag_star[from_stop].update(dep_secs, None, None)
            marked_stops.append(from_stop)
            # If renting station, adding to known shared-mob
            if (isinstance(from_stop, RentingStation)
                    and from_stop not in renting_stations_known):
                renting_stations_known.append(from_stop)

        s1, sm1 = len(marked_stops), len(renting_stations_known)  # debugging
        logger.debug(f"Starting from {s1} stops ({sm1} are renting stations)")

        # TODO why can't all this section be replaced by a single call to add_transfer_time()?
        # Adding to initial stops immediately transferable stops
        immediate: Dict[Stop, Transfer] = self._get_immediate_transferable(marked_stops)

        for stop, t in immediate.items():
            if stop not in marked_stops:  # skipping already marked stops
                arr_time = dep_secs + t.transfer_time
                from_stop = t.from_stop
                to_stop = t.to_stop
                transfer_trip: Trip = TransferTrip(
                    from_stop=from_stop,
                    to_stop=to_stop,
                    dep_time=dep_secs,
                    arr_time=arr_time,

                    # TODO
                    transport_type=TransportType.Walk
                )
                bag_round_stop[0][stop].update(arr_time, transfer_trip, from_stop)
                self.bag_star[stop].update(arr_time, transfer_trip, from_stop)
                marked_stops.append(stop)

                # If renting station, adding to known shared-mob
                if (isinstance(stop, RentingStation)
                        and stop not in renting_stations_known):
                    renting_stations_known.append(stop)

        s2, sm2 = len(marked_stops), len(renting_stations_known)  # debugging
        logger.debug(f"Added {s2 - s1} immediate stops, ({sm2 - sm1} are renting stations)")

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

            # Part 4
            # There may be some renting stations in  `marked_transfer_stops`:
            # indeed we can reach a public stop with a trip
            # and then use a footpath (a transfer) and walk to a renting station
            #
            # We filter these renting station in `renting_station_from_trip`
            renting_stations_from_trip: List[RentingStation] = filter_shared_mobility(marked_transfer_stops)
            logger.debug(f"{len(renting_stations_from_trip)} renting stations reachable ")

            # then we keep only new renting stations
            new_renting_stations: List[RentingStation] = list(
                set(renting_stations_from_trip).difference(renting_stations_known)
            )
            logger.debug(f"New {len(new_renting_stations)} renting station reachable ")

            # We add a VehicleTransfer foreach (old, new) renting station
            # (according to system_id and availability)
            t1 = len(self.v_transfers)  # debugging

            for old in renting_stations_known:
                for new in new_renting_stations:
                    self._add_vehicle_transfer(old, new)

            t2 = len(self.v_transfers)  # debugging

            logger.debug(f"New {t2 - t1} vehicle transfers created")

            # We can try to improve the best arrival-time taking advantage of shared-mobility network:
            #
            # We consider only:
            #     - vehicle-transfers
            #     - just Transfers which to_stop is in `renting_stations_from_trip`
            #     -
            # List of vehicle-transfers arriving to reachable renting stations
            vtransfers_sub: List[List[VehicleTransfer]] = [self.v_transfers.with_to_stop(s) for s in
                                                           new_renting_stations]
            vtransfers_sub: List[VehicleTransfer] = [i for sub in vtransfers_sub for i in sub]  # flatten

            # List of departing renting stations from previous filtered vehicle-transfers
            renting_stations_known_sub: List[RentingStation] = list(set([t.from_stop for t in vtransfers_sub]))

            # We can get compute transfer-time from these selected renting stations using only filtered transfers
            bag_round_stop, renting_stations_from_trip_improved = self.add_transfer_time(
                bag_round_stop, k, renting_stations_known_sub, vtransfers_sub
            )
            logger.debug(f"{len(renting_stations_from_trip_improved)} transferable renting stations upgraded")

            # Finally, we can update known renting stops including new ones too
            renting_stations_known = list(set(renting_stations_known).union(new_renting_stations))

            # Part 5
            # `renting_stations_from_trip_improved` contains all improved renting-stations
            # These improvements must reflect to public-transport network, so we compute footpaths (Transfers)
            # between improved renting stations and associated transferable public stops
            bag_round_stop, marked_shared_mob_stops = self.add_transfer_time(
                bag_round_stop, k, renting_stations_from_trip_improved
            )
            logger.debug(f"{len(marked_shared_mob_stops)} using shared-mobility stops upgraded")

            marked_stops = list(set(marked_trip_stops).union(marked_transfer_stops, marked_shared_mob_stops))

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
            other_station_stops_trip = [
                (t.to_stop, t) for t in transfers if t.from_stop == current_stop
            ]

            time_sofar = bag_round_stop[k][current_stop].earliest_arrival_time
            for arrive_stop, transfer in other_station_stops_trip:
                arrival_time_with_transfer = time_sofar + self.get_transfer_time(
                    current_stop, arrive_stop
                )
                previous_earliest_arrival = self.bag_star[
                    arrive_stop
                ].earliest_arrival_time

                # Domination criteria
                if arrival_time_with_transfer < previous_earliest_arrival:
                    transport_type = transfer.transport_type \
                                      if isinstance(transfer, VehicleTransfer) \
                                      else TransportType.Walk
                    transfer_trip = TransferTrip(
                        from_stop=current_stop,
                        to_stop=arrive_stop,
                        dep_time=time_sofar,
                        arr_time=arrival_time_with_transfer,

                        # TODO add method or field `transfer_type` to base Transfer class
                        #  such accesser is then overrode by shared mobility Transfer sub-classes
                        transport_type=transport_type
                    )

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
        try:
            return transfers.stop_to_stop_idx[(stop_from, stop_to)].transfer_time
        except:  # search on vehicle transfers
            return self.v_transfers.stop_to_stop_idx[(stop_from, stop_to)].transfer_time


def best_stop_at_target_station(to_stops: List[Stop], bag: Dict[Stop, Label]) -> Stop:
    """
    Find the destination Stop with the shortest distance.
    Required in order to prevent adding travel time to the arrival time.
    """

    # TODO use BaseRaptorLabel
    return raptor.best_stop_at_target_station(to_stops=to_stops, bag=bag)


def reconstruct_journey(destination: Stop, bag: Dict[Stop, Label]) -> Journey:
    """Construct journey for destination from values in bag."""

    # TODO use BaseRaptorLabel
    return raptor.reconstruct_journey(destination=destination, bag=bag)


def is_dominated(original_journey: List[Leg], new_journey: List[Leg]) -> bool:
    """Check if new journey is dominated by another journey"""

    return raptor.is_dominated(original_journey=original_journey, new_journey=new_journey)
