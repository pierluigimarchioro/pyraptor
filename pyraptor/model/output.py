from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Callable, Tuple, Any

import joblib
import numpy as np
from loguru import logger

from pyraptor.model.criteria import Criterion, Bag, pareto_set, MultiCriteriaLabel
from pyraptor.model.timetable import Stop, Trip, TimetableInfo
from pyraptor.util import sec2str, mkdir_if_not_exists


@dataclass
class Leg:
    """Leg"""

    from_stop: Stop
    to_stop: Stop
    trip: Trip
    criteria: Iterable[Criterion]

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

    @property
    def total_cost(self) -> float:
        return sum(self.criteria, start=0.0)

    def is_same_station_transfer(self) -> bool:
        """
        Returns true if the current instance is a transfer leg between stops
        belonging to the same station (i.e. platforms)
        :return:
        """

        return self.from_stop.station == self.to_stop.station

    def is_compatible_before(self, other_leg: Leg) -> bool:
        """
        Check if Leg is allowed before another leg, that is if the accumulated value of
        the criteria of the current leg is less or equal to the accumulated value of
        those of the other leg (current leg is instance of this class).
        E.g. Leg X+1 criteria must be >= their counter-parts in Leg X, because
            Leg X+1 comes later.
        """

        criteria_compatible = np.all(
            np.array([c.raw_value for c in other_leg.criteria])
            >= np.array([c.raw_value for c in self.criteria])
        )

        return all([criteria_compatible])

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
            criteria=self.criteria
        )

    def __str__(self):
        return f"From {self.from_stop} to {self.to_stop} | Criteria: {self.criteria}"


@dataclass(frozen=True)
class Journey:
    """
    Journey from origin to destination specified as Legs
    """

    legs: List[Leg] = field(default_factory=list)

    def __len__(self):
        return len(self.legs)

    def __repr__(self):
        return f"Journey(n_legs={len(self.legs)})"

    def __getitem__(self, index):
        return self.legs[index]

    def __iter__(self):
        return iter(self.legs)

    def __lt__(self, other):
        return self.dep() < other.dep()

    def __str__(self):
        def update_(s: str):
            nonlocal out_str
            out_str += f"{s}\n"

        out_str = ''

        update_("Journey")

        if len(self) == 0:
            update_("No journey available ")
            return out_str

        # Print all legs in journey
        first_trip = self.legs[0].trip
        prev_trip = first_trip if first_trip is not None else None
        n_changes = 1
        for leg in self:
            current_trip = leg.trip
            if current_trip is not None:
                hint = current_trip.hint

                if current_trip != prev_trip:
                    trip_change = f"-- Trip Change #{n_changes} -- "
                    update_(trip_change)
                    n_changes += 1

                prev_trip = current_trip
            else:
                raise Exception(f"Leg trip cannot be {None}. Value: {current_trip}")

            msg = (
                    str(sec2str(leg.dep))
                    + " "
                    + leg.from_stop.station.name.ljust(20)
                    + " TO "
                    + str(sec2str(leg.arr))
                    + " "
                    + leg.to_stop.station.name.ljust(20)
                    + " WITH "
                    + str(hint)
            )
            update_(msg)

            # TODO debug
            update_(f"[Leg] Last updated at: {leg.last_updated_round}")
            for c in leg.criteria:
                update_(f"[Leg] {str(c)}")

        update_("")
        for c in self.criteria():
            update_(str(c))

        msg = f"Duration: {sec2str(self.travel_time())}"
        update_(msg)
        update_("")

        return out_str

    def number_of_trips(self):
        """Return number of distinct trips"""
        trips = set([lbl.trip for lbl in self.legs])
        return len(trips)

    def prepend_leg(self, leg: Leg) -> Journey:
        """Add leg to journey"""
        legs = self.legs
        legs.insert(0, leg)
        jrny = Journey(legs=legs)
        return jrny

    def remove_empty_legs(self) -> Journey:
        """
        Removes all empty legs (where the trip is not set)
        and transfer legs between stops of the same station.

        :return: updated journey
        """

        legs = [
            leg
            for leg in self.legs
            if (leg.trip is not None)
        ]
        jrny = Journey(legs=legs)

        return jrny

    def is_valid(self) -> bool:
        """
        Returns true if the journey is considered valid.
        Notably, a journey is valid if, for each leg, leg k arrival time
        is not greater than leg k+1 departure time.

        :return: True if journey is valid, False otherwise
        """

        for index in range(len(self.legs) - 1):
            if not self.legs[index].is_compatible_before(self.legs[index + 1]):
                return False

        return True

    def from_stop(self) -> Stop:
        """Origin stop of Journey"""
        return self.legs[0].from_stop

    def to_stop(self) -> Stop:
        """Destination stop of Journey"""
        return self.legs[-1].to_stop

    def dep(self) -> int:
        """Departure time"""
        return self.legs[0].dep

    def arr(self) -> int:
        """Arrival time"""
        return self.legs[-1].arr

    def travel_time(self) -> int:
        """Travel time in seconds"""
        return self.arr() - self.dep()

    def criteria(self) -> Iterable[Criterion]:
        """
        Returns the final criteria for the journey, which correspond to
        the criteria values of the final leg.
        :return:
        """

        return self.legs[-1].criteria

    def total_cost(self) -> float:
        """
        Returns the total cost of the journey
        :return:
        """

        return sum(self.criteria(), start=0.0)

    def dominates(self, jrny: Journey):
        """Dominates other Journey"""
        return (
            True
            if (self.total_cost() <= jrny.total_cost())
            and (self != jrny)
            else False
        )

    def print(self, dep_secs: int = None, logger_: Callable[[str], None] = logger.info):
        """Prints the current journey instance on the provided logger"""

        logger_(str(self))

        if dep_secs:
            logger_(f" ({sec2str(self.arr() - dep_secs)} from request time {sec2str(dep_secs)})")

    def to_list(self) -> List[Dict]:
        """Convert journey to list of legs as dict"""
        return [leg.to_dict(leg_index=idx) for idx, leg in enumerate(self.legs)]


def get_journeys_to_destinations(
        origin_stops: Iterable[Stop],
        destination_stops: Dict[Any, Iterable[Stop]],
        best_labels: Mapping[Stop, Bag]
) -> Mapping[Any, Sequence[Journey]]:
    # Calculate journeys to all destinations
    logger.info("Calculating journeys to all destinations")
    s = perf_counter()

    journeys_to_destinations = {}
    for destination_station_name, to_stops in destination_stops.items():
        destination_legs = _best_legs_to_destination_station(to_stops, best_labels)

        if len(destination_legs) == 0:
            logger.debug(f"Destination '{destination_station_name}' unreachable with given parameters."
                         f"Station stops: {to_stops}")
            continue

        journeys = _reconstruct_journeys(
            origin_stops, destination_legs, best_labels
        )
        journeys_to_destinations[destination_station_name] = journeys

    logger.info(f"Journey calculation time: {perf_counter() - s}")

    return journeys_to_destinations


def _best_legs_to_destination_station(
        to_stops: Iterable[Stop],
        last_round_bag: Mapping[Stop, Bag]
) -> Sequence[Leg]:
    """
    Find the last legs to destination station that are reached by non-dominated labels.
    """

    # Find all labels to target_stops
    best_labels = [
        (stop, label) for stop in to_stops for label in last_round_bag[stop].labels
    ]

    # TODO Use merge function on Bag
    # Pareto optimal labels
    pareto_optimal_labels = pareto_set([label for (_, label) in best_labels])
    pareto_optimal_labels: List[Tuple[Stop, MultiCriteriaLabel]] = [
        (stop, label) for (stop, label) in best_labels if label in pareto_optimal_labels
    ]

    # Label to leg, i.e. add to_stop
    legs = [
        Leg(
            from_stop=label.boarding_stop,
            to_stop=to_stop,
            trip=label.trip,
            criteria=label.criteria
        )
        for to_stop, label in pareto_optimal_labels
    ]
    return legs


def _reconstruct_journeys(
        from_stops: Iterable[Stop],
        destination_legs: Iterable[Leg],
        best_labels: Mapping[Stop, Bag],
        add_intermediate_legs: bool = True
) -> List[Journey]:
    """
    Construct Journeys for destinations from bags by recursively
    looping from destination to origin.
    """

    def loop(best_labels: Mapping[Stop, Bag], journeys: Iterable[Journey]):
        """Create full journey by prepending legs recursively"""

        for jrny in journeys:
            later_leg = jrny[0]

            # End of journey if we are at origin stop or journey is not feasible
            if later_leg.trip is None or later_leg.from_stop in from_stops:
                jrny = jrny.remove_empty_legs()

                # Journey is valid if leg k ends before the start of leg k+1
                if jrny.is_valid() is True:
                    yield jrny

                continue

            # Loop trough each new leg. These are the legs that come before the current and that lead to from_stop
            labels_to_from_stop = best_labels[later_leg.from_stop].labels
            for new_label in labels_to_from_stop:
                full_earlier_leg = Leg(
                    from_stop=new_label.boarding_stop,
                    to_stop=later_leg.from_stop,
                    trip=new_label.trip,
                    criteria=new_label.criteria
                )

                # Only add the new leg if compatible before current leg, e.g. earlier arrival time, etc.
                if full_earlier_leg.is_compatible_before(later_leg):
                    # Generate and prepend the intermediate legs to the provided journey,
                    # starting from the full earlier leg
                    if add_intermediate_legs:
                        intermediate_legs = _generate_intermediate_legs(
                            full_leg=full_earlier_leg
                        )

                        new_jrny = jrny
                        for leg in intermediate_legs:
                            new_jrny = new_jrny.prepend_leg(leg)
                    else:
                        # Only add the full leg if intermediate legs are not added
                        new_jrny = jrny.prepend_leg(full_earlier_leg)

                    for i in loop(best_labels, [new_jrny]):
                        yield i

    if add_intermediate_legs:
        journeys = [Journey(legs=list(reversed(_generate_intermediate_legs(leg))))
                    for leg in destination_legs]
    else:
        journeys = [Journey(legs=[leg]) for leg in destination_legs]
    journeys = [jrny for jrny in loop(best_labels, journeys)]

    return journeys


def _generate_intermediate_legs(full_leg: Leg) -> List[Leg]:
    """
    Generates the intermediate legs from the provided full leg,
    i.e. all the sub-legs involving each stop of the trip that the leg represents.
    Such intermediate legs are then returned, with the first leg of the collection
    being the later one.

    :param full_leg: full leg to generate intermediate legs from
    :return: intermediate legs, ordered by most late
    """

    # Start from `to_stop` of the full leg (which is `from_stop` of the later leg)
    # and go backwards until `from_stop` of the full earlier leg is reached:
    # these are the intermediate legs
    prev_dep_stop: Stop = full_leg.to_stop
    first_stop_reached: bool = False
    legs = []
    while not first_stop_reached:
        # The first intermediate stop is the one that comes before
        # the last stop of the previous intermediate leg
        intermediate_stop_idx = (full_leg.trip.stop_times_index[prev_dep_stop] - 1)
        intermediate_stop = full_leg.trip[intermediate_stop_idx].stop
        intermediate_leg = Leg(
            from_stop=intermediate_stop,
            to_stop=prev_dep_stop,
            trip=full_leg.trip,

            # Setting the criteria to be the same ones of the leg that is
            # currently being reconstructed (the earlier one)
            # and not specifically retrieving them from best labels is fine,
            # since the compatibility between consecutive legs is maintained
            criteria=full_leg.criteria
        )
        legs.append(intermediate_leg)

        prev_dep_stop = intermediate_leg.from_stop

        # Stop adding intermediate legs when the first stop
        # of the full leg is reached
        if intermediate_leg.from_stop == full_leg.from_stop:
            first_stop_reached = True

    return legs


@dataclass
class AlgorithmOutput(TimetableInfo):
    """
    Class that represents the data output of a Raptor algorithm execution.
    Contains the best journey found by the algorithm, the departure date and time of said journey
    and the path to the directory of the GTFS feed originally used to build the timetable
    provided to the algorithm.
    """

    _DEFAULT_FILENAME = "algo-output"

    journeys: Iterable[Journey] = None
    """Best journey found by the algorithm"""

    departure_time: str = None
    """string in the format %H:%M:%S"""

    @staticmethod
    def read_from_file(filepath: str | bytes | os.PathLike) -> AlgorithmOutput:
        """
        Returns the AlgorithmOutput instance read from the provided folder
        :param filepath: path to an AlgorithmOutput .pcl file
        :return: AlgorithmOutput instance
        """

        def load_joblib() -> AlgorithmOutput:
            logger.debug(f"Loading '{filepath}'")
            with open(Path(filepath), "rb") as handle:
                return joblib.load(handle)

        if not os.path.exists(filepath):
            raise IOError(
                "PyRaptor AlgorithmOutput not found. Run `python pyraptor/query_raptor`"
                " first to generate an algorithm output .pcl file."
            )

        logger.debug("Using cached datastructures")

        algo_output: AlgorithmOutput = load_joblib()

        return algo_output

    @staticmethod
    def save(output_dir: str | bytes | os.PathLike,
             algo_output: AlgorithmOutput,
             filename: str = _DEFAULT_FILENAME):
        """
        Write the algorithm output to the provided directory

        :param output_dir: path to the directory to write the serialized output file
        :param algo_output: instance to serialize
        :param filename: name of the serialized output file
        """

        def write_joblib(to_serialize: object, filename: str):
            with open(Path(output_dir, f"{filename}.pcl"), "wb") as handle:
                joblib.dump(to_serialize, handle)

        logger.info(f"Writing PyRaptor output to {output_dir}")

        mkdir_if_not_exists(output_dir)
        write_joblib(algo_output, filename)
