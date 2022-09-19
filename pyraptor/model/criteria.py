from __future__ import annotations

import json
import os
from abc import abstractmethod, ABC
from collections.abc import Sequence, MutableMapping
from dataclasses import dataclass, field
from itertools import compress
from typing import List, Type, Dict, TypeVar, Generic

import attr
import numpy as np
from loguru import logger

from pyraptor.model.timetable import TransportType, Trip, Stop
from pyraptor.util import sec2str, LARGE_NUMBER


@dataclass(frozen=True)
class BaseLabel(ABC):
    """
    Abstract class representing the base characteristics that a RAPTOR label
    needs to have. Depending on the algorithm version, there are different types of
    labels.

    Generally speaking, each label contains the trip with which one arrives at the label's associated stop
    with k legs by boarding the trip at the boarding stop. It also contains the criteria with which each
    stop is evaluated by the algorithm.

    Reference (RAPTOR paper):
    https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf
    """

    trip: Trip | None = None
    """Trip to take to arrive at the destination stop at `earliest_arrival_time`"""

    boarding_stop: Stop = None
    """Stop at which the trip is boarded"""

    @abstractmethod
    def get_earliest_arrival_time(self):
        """
        Returns the earliest arrival time associated to this label

        :return: arrival time in seconds past the midnight of the departure day
        """
        pass

    @abstractmethod
    def update(self, data: LabelUpdate) -> BaseLabel:
        """
        Returns a new label with updated attributes.
        If the provided values are None, the corresponding attributes are not updated.

        :param data: label update data
        :return: new updated label
        """
        pass

    @abstractmethod
    def is_dominating(self, other: BaseLabel) -> bool:
        """
        Returns true if the current label is dominating the provided label,
        meaning that it is not worse in any of the valuation criteria.

        :param other: other label to compare
        :return:
        """
        pass


@dataclass(frozen=True)
class BasicRaptorLabel(BaseLabel):
    """
    Class that represents a label used in the basic RAPTOR variant
    described in the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf).
    """

    earliest_arrival_time: int = LARGE_NUMBER
    """Earliest time to get to the destination stop by boarding the current trip"""

    def get_earliest_arrival_time(self):
        return self.earliest_arrival_time

    def update(self, data: LabelUpdate) -> BasicRaptorLabel:
        trip = data.new_trip if self.trip != data.new_trip else self.trip
        boarding_stop = data.boarding_stop if data.boarding_stop is not None else self.boarding_stop

        # Earliest arrival time to the arrival stop on the updated trip
        earliest_arrival_time = trip.get_stop_time(data.arrival_stop).dts_arr

        return BasicRaptorLabel(
            earliest_arrival_time=earliest_arrival_time,
            boarding_stop=boarding_stop,
            trip=trip
        )

    def is_dominating(self, other: BasicRaptorLabel) -> bool:
        return self.earliest_arrival_time <= other.earliest_arrival_time

    def __repr__(self) -> str:
        return f"{BasicRaptorLabel.__name__}(earliest_arrival_time={self.earliest_arrival_time}, " \
               f"trip={self.trip}, boarding_stop={self.boarding_stop})"


@dataclass(frozen=True)
class MultiCriteriaLabel(BaseLabel):
    """
    Class that represents a multi-criteria label.

    The concept this is class is modeled after is that of the multi-label in the
    `McRAPTOR` section of the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf)
    """

    arrival_stop: Stop = attr.ib(default=None)
    """Stop that this label is associated to"""

    criteria: Sequence[Criterion] = attr.ib(default=list)
    """Collection of criteria used to compare labels"""

    # TODO since arrival time information must always be considered for correctness purposes,
    #   consider defining an arrival_time attribute in the BaseLabel class
    #   (possibly of type ArrivalTimeCriterion?)

    @staticmethod
    def from_base_raptor_label(label: BasicRaptorLabel) -> MultiCriteriaLabel:
        """
        Creates a multi-criteria label from a base RAPTOR label instance.
        The new multi-criteria label has a total cost of 1.

        :param label: base RAPTOR label to convert to multi-criteria
        :return: converted multi-criteria label
        """

        # Args except raw_value are not important
        arr_criterion = ArrivalTimeCriterion(
            raw_value=label.earliest_arrival_time
        )
        mc_lbl = MultiCriteriaLabel(
            boarding_stop=label.boarding_stop,
            trip=label.trip,
            criteria=[arr_criterion]
        )

        return mc_lbl

    @property
    def total_cost(self) -> float:
        """
        Returns the total cost assigned to the label, which corresponds to
        the weighted sum of its criteria.
        :return: float instance representing the total cost
        """

        if len(self.criteria) == 0:
            raise Exception("No criteria to calculate cost with")

        return sum(self.criteria, start=0.0)

    def get_earliest_arrival_time(self) -> int:
        """
        Returns the earliest arrival time associated to this label

        :return: arrival time in seconds past the midnight of the departure day
        """

        arrival_time_crit = next(
            filter(lambda c: isinstance(c, ArrivalTimeCriterion), self.criteria),
            None
        )

        if arrival_time_crit is None:
            raise ValueError(f"No {ArrivalTimeCriterion.__name__} is defined for this label")
        else:

            return int(arrival_time_crit.raw_value)

    def update(self, data: LabelUpdate[MultiCriteriaLabel]) -> MultiCriteriaLabel:
        if len(self.criteria) == 0:
            raise Exception("Trying to update an instance with no criteria set")

        updated_criteria = []
        for c in self.criteria:
            updated_c = c.update(data=data)
            updated_criteria.append(updated_c)

        updated_trip = data.new_trip if data.new_trip is not None else self.trip
        updated_dep_stop = data.boarding_stop if data.boarding_stop is not None else self.boarding_stop
        updated_arr_stop = data.arrival_stop if data.arrival_stop is not None else self.arrival_stop

        return MultiCriteriaLabel(
            boarding_stop=updated_dep_stop,
            arrival_stop=updated_arr_stop,
            criteria=updated_criteria,
            trip=updated_trip,
        )

    def is_dominating(self, other: MultiCriteriaLabel) -> bool:
        return self.total_cost <= other.total_cost


class Criterion(ABC):
    """
    Base class for a RAPTOR label criterion
    """

    def __init__(
            self, 
            name: str = "", 
            weight: float = 1.0, 
            raw_value: float = 0.0, 
            upper_bound: float = LARGE_NUMBER
    ):
        self.name: str = name
        """Name of the criterion"""

        self.weight: float = weight
        """Weight used to determine the cost of this criterion"""

        self.raw_value: float = raw_value
        """
        Raw value of the criterion, that is before any weight is applied.
        This value maintains is expressed in the original unit of measurement.
        """

        self.upper_bound: float = upper_bound
        """
        Maximum value allowed for this criterion.
        Such threshold has two main purposes:

        - to scale the raw value into the [0,1] range;
        - to represent the maximum accepted raw value for the criteria, over which
            the label is considered very unfavorable (the cost becomes artificially very large)
            or discarded completely.
        """

    @property
    def cost(self) -> float:
        """
        Returns the weighted cost of this criterion.
        The raw cost is scaled on the range [0, `upper_bound`] and is then
        multiplied by the provided weight.

        :return: weighted scaled cost
        """

        if self.raw_value > self.upper_bound:
            return LARGE_NUMBER
        else:
            return self.weight * (self.raw_value / self.upper_bound)  # lower bound is always 0

    def __add__(self, other: object) -> Criterion | float:
        """
        Returns the sum between two criteria, which is:
        - a Criterion instance if the two objects are of type Criterion
            and have the same name and weight;
        - a float, which is the weighted sum of their values,
            if the two objects are of type Criterion but have different names,
            or if the other object is of type float (which is assumed to be a cost);
        - an exception if the two objects are not of type Criterion or float
            or if they have the same name but differ in the other characteristics
            (weight, upper bound)
        :param other: second addend of the sum operation
        :return: Criterion or float instance
        """

        if isinstance(other, Criterion):
            if other.name == self.name:
                if other.weight != self.weight or other.upper_bound != self.upper_bound:
                    raise Exception(f"Cannot add criteria with the same name but different characteristics"
                                    f"(weight, upper bound).\n"
                                    f"First addend: {self} - Second addend: {other}")
                else:
                    return Criterion(
                        name=self.name,
                        weight=self.weight,
                        raw_value=(self.raw_value + other.raw_value),
                        upper_bound=self.upper_bound
                    )
            else:
                return self.cost + other.cost
        elif isinstance(other, float):
            return self.cost + other
        else:
            raise TypeError(f"Cannot add type {Criterion.__name__} with type {other.__class__.__name__}.\n"
                            f"Second addend: {other}")

    def __radd__(self, other) -> Criterion | float:
        return self.__add__(other)

    def __lt__(self, other):
        return self.__cmp__(other) == -1

    def __le__(self, other):
        cmp = self.__cmp__(other)
        return cmp <= 0

    def __gt__(self, other):
        return self.__cmp__(other) == 1

    def __ge__(self, other):
        cmp = self.__cmp__(other)
        return cmp >= 0

    def __cmp__(self, other) -> int:
        # 0 if equal, -1 if < other, +1 if > other
        if self.cost < other.cost:
            return -1
        elif self.cost == other.cost:
            return 0
        else:
            return 1

    def __float__(self) -> float:
        return self.cost

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name};weight={self.weight};" \
               f"raw_value={self.raw_value};upper_bound={self.upper_bound})"

    @abstractmethod
    def update(self, data: LabelUpdate) -> Criterion:
        pass


# Generic type vars for Criterion and Label subclasses
_C = TypeVar('_C', bound=Criterion)
_LabelType = TypeVar("_LabelType", bound=BaseLabel)


# NOTE: what if `stop` is associated with two labels with an equal cost?
# In that case, `best_label` would contain just one of the two, with the choice
# being non-deterministic. This could create errors when calculating the criteria
# and reconstructing the journey. In this implementation, however, it is certain that at most
# one label is associated with each stop at any given time, because pareto_set() and Bag.merge()
# implementations just keep the best one (keep_equal arg is False), and non-deterministically choose
# one between labels with the same cost. It is however worthy to point out this potential breaking point.
# A possible solution would be to remove the concept of Bag and keeping just a single label,
# therefore forcing the above situation to happen (instead of relying on the implementation details
# of the Bag.merge() method)
def _get_best_stop_criterion(
        criterion_class: Type[_C],
        stop: Stop,
        best_labels: MutableMapping[Stop, MultiCriteriaLabel]
) -> _C:
    """
    Returns the instance of the specified type of criterion, which is retrieved from the
    best label associated with the provided stop.
    An exception is raised if the criterion instance couldn't be found.

    :param criterion_class: type of the criterion to retrieve
    :param stop: stop to retrieve the best criterion for
    :param best_labels: collection that pairs the best labels with their associated stop
    :return: instance of the specified criterion type
    """

    # Criteria can be retrieved only from MultiCriteriaLabel instances
    best_label = best_labels[stop]
    criterion = next(
        filter(lambda c: isinstance(c, criterion_class), best_label.criteria),
        None
    )
    if criterion is None:
        raise ValueError(f"The provided best labels do not include "
                         f"a criterion of type {criterion_class.__name__}")

    return criterion


class DistanceCriterion(Criterion):
    """
    Class that represents and handles calculations for the distance criterion.
    The value represents the total number of km travelled.
    """
    
    def __init__(self, weight: float = 1.0, raw_value: float = 0.0, upper_bound: float = LARGE_NUMBER):
        super(DistanceCriterion, self).__init__(
            name="Total Distance",
            weight=weight,
            raw_value=raw_value,
            upper_bound=upper_bound
        )

    def __str__(self):
        return f"Travelled Distance: {self.raw_value} [Km]"

    def update(self, data: LabelUpdate[MultiCriteriaLabel]) -> DistanceCriterion:
        arrival_distance = self._get_total_arrival_distance(data=data)

        return DistanceCriterion(
            weight=self.weight,
            raw_value=arrival_distance,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_distance(data: LabelUpdate[MultiCriteriaLabel]) -> float:
        """
        Returns the updated distance (in km) for the criterion instance based on the
        new provided boarding and arrival stop. Such value represents the total travelled
        distance between the origin stop and the provided arrival stop.

        :param data: update data
        :return: distance in km
        """

        # The formula is the following:
        # total_distance(arrival) = total_distance(boarding) + [trip_distance(arrival) - trip_distance(boarding)]
        # where trip_distance(x) is the cumulative distance of the trip T that leads to x, starting from
        # the beginning of T
        same_trip_distance = _get_same_trip_distance(
            trip=data.new_trip,
            from_stop=data.boarding_stop,
            to_stop=data.arrival_stop
        )

        # Extract the total distance of the previous stop (boarding stop) in the journey
        # from its distance criterion instance
        prev_stop_dist_criterion = _get_best_stop_criterion(
            criterion_class=DistanceCriterion,
            stop=data.boarding_stop,
            best_labels=data.best_labels
        )

        return prev_stop_dist_criterion.raw_value + same_trip_distance


def _get_same_trip_distance(trip: Trip, from_stop: Stop, to_stop: Stop) -> float:
    """
    Returns the distance between the two provided stops in the specified trip.

    :param trip: trip covering the two stops
    :param from_stop: first stop
    :param to_stop: second stop
    :return: distance between the first and second stop, in km
    """

    from_stop_time = trip.get_stop_time(from_stop)
    to_stop_time = trip.get_stop_time(to_stop)

    return to_stop_time.travelled_distance - from_stop_time.travelled_distance


class EmissionsCriterion(Criterion):
    """
    Class that represents and handles calculations for the co2 emissions criterion
    """
    
    def __init__(self, weight: float = 1.0, raw_value: float = 0.0, upper_bound: float = LARGE_NUMBER):
        super(EmissionsCriterion, self).__init__(
            name="Total Emissions",
            weight=weight,
            raw_value=raw_value,
            upper_bound=upper_bound
        )

    def __str__(self):
        return f"Total Emissions: {self.raw_value} [CO2 grams / passenger Km]"

    def update(self, data: LabelUpdate[MultiCriteriaLabel]) -> EmissionsCriterion:
        arrival_emissions = self._get_total_arrival_emissions(data=data)

        return EmissionsCriterion(
            weight=self.weight,
            raw_value=arrival_emissions,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_emissions(data: LabelUpdate[MultiCriteriaLabel]) -> float:
        """
        Returns the updated total emissions (in co2 grams / passenger km) for
        this criterion instance, based on the new provided boarding and arrival stop.
        Such value represents the total emissions between the origin stop
        and the provided arrival stop.

        :param data: update data
        :return: emissions in co2 grams / passenger km
        """

        same_trip_distance = _get_same_trip_distance(
            trip=data.new_trip,
            from_stop=data.boarding_stop,
            to_stop=data.arrival_stop
        )

        co2_multiplier = EmissionsCriterion.get_emission_multiplier(
            transport_type=data.new_trip.route_info.transport_type
        )
        same_trip_emissions = same_trip_distance * co2_multiplier

        prev_stop_emissions_crit = _get_best_stop_criterion(
            criterion_class=EmissionsCriterion,
            stop=data.boarding_stop,
            best_labels=data.best_labels
        )

        return prev_stop_emissions_crit.raw_value + same_trip_emissions

    @staticmethod
    def get_emission_multiplier(transport_type: TransportType) -> float:
        """
        Returns the emission multiplier for the provided transport type,
        expressed in `co2 grams / passenger km`
        :return:
        """

        # Sources (values expressed in co2 grams/passenger km):
        # - https://ourworldindata.org/travel-carbon-footprint
        # - Ferry https://www.thrustcarbon.com/insights/how-to-calculate-emissions-from-a-ferry-journey
        # - Electric Bike https://www.bosch-ebike.com/us/service/sustainability
        co2_grams_per_passenger_km: Dict[TransportType, float] = {
            TransportType.Walk: 0,
            TransportType.Bike: 0,
            TransportType.ElectricBike: 14,
            TransportType.Car: (192 + 172) / 2,  # Avg between petrol and diesel

            # It is assumed that all rail vehicles have the same impact,
            # since, even if different sources point to different numbers,
            # the average emissions per passenger km between the different
            # rail transports are approximately equal
            TransportType.Rail: 41,
            TransportType.LightRail: 35,

            # Monorail and cable cars are all assumed to have
            # the same impact of light rail transport, since they are
            # all usually electrically powered (couldn't find specific data)
            TransportType.Monorail: 35,
            TransportType.CableTram: 35,
            TransportType.Funicular: 35,
            TransportType.AerialLift: 35,
            TransportType.Metro: 31,

            # Since trolleybus are very similar to trams, except they have wheels,
            # it is assumed that their emissions are equal. I couldn't find
            # recent data about trolleybus co2 emissions per passenger/km
            TransportType.TrolleyBus: 35,
            TransportType.Bus: 105,
        }

        return co2_grams_per_passenger_km[transport_type]


class ArrivalTimeCriterion(Criterion):
    """
    Class that represents and handles calculations for the arrival time criterion
    """

    def __init__(self, weight: float = 1.0, raw_value: float = 0.0, upper_bound: float = LARGE_NUMBER):
        super(ArrivalTimeCriterion, self).__init__(
            name="Arrival Time",
            weight=weight,
            raw_value=raw_value,
            upper_bound=upper_bound
        )

    def __str__(self):
        return f"Arrival Time: {sec2str(seconds=int(self.raw_value))}"

    def update(self, data: LabelUpdate) -> ArrivalTimeCriterion:
        new_arrival_time = data.new_trip.get_stop_time(data.arrival_stop).dts_arr

        if new_arrival_time is None or np.isnan(new_arrival_time):
            logger.error(f"Arrival time for stop {data.arrival_stop} is None\n"
                         f"Stop time object: {data.new_trip.get_stop_time(data.arrival_stop)}")

            raise ValueError(f"Arrival time not found for stop {data.arrival_stop}")

        return ArrivalTimeCriterion(
            weight=self.weight,
            raw_value=new_arrival_time,
            upper_bound=self.upper_bound
        )


DEFAULT_ORIGIN_TRIP = None
"""Trip initially assigned to the origin stops of a journey"""


class TransfersCriterion(Criterion):
    """
    Class that represents and handles calculations for the number of transfers criterion.
    A transfer is defined as a change of trip, excluding the initial change that happens
    at the origin stops to board the first trip.
    """
    
    def __init__(self, weight: float = 1.0, raw_value: float = 0.0, upper_bound: float = LARGE_NUMBER):
        super(TransfersCriterion, self).__init__(
            name="Transfers Number",
            weight=weight,
            raw_value=raw_value,
            upper_bound=upper_bound
        )

    def __str__(self):
        return f"Total Transfers: {self.raw_value}"

    def update(self, data: LabelUpdate) -> TransfersCriterion:
        # The transfer counter is updated only if:
        # - there is a trip change (new != old) and
        #       the old isn't the initial trip (origin trip)
        best_boarding_label = data.best_labels[data.boarding_stop]
        add_new_transfer = data.new_trip != best_boarding_label.trip and best_boarding_label.trip != DEFAULT_ORIGIN_TRIP

        transfers_at_boarding = _get_best_stop_criterion(
            criterion_class=TransfersCriterion,
            stop=data.boarding_stop,
            best_labels=data.best_labels
        )

        return TransfersCriterion(
            weight=self.weight,
            raw_value=transfers_at_boarding.raw_value if not add_new_transfer else transfers_at_boarding.raw_value + 1,
            upper_bound=self.upper_bound
        )


@dataclass
class CriterionConfiguration:
    """
    Class that represents the configuration for some criterion.
    It defines weight and upper_bound parameters
    """

    weight: float
    upper_bound: float


class CriteriaProvider:
    """
    Class that represents a criteria factory whose goal is to create properly
    parameterized criterion instances based on some specified configuration
    """

    _criteria_config: Dict[Type[Criterion], CriterionConfiguration]

    def __init__(self, criteria_config: Dict[Type[Criterion], CriterionConfiguration]):
        self._criteria_config = criteria_config

    @property
    def criteria_config(self) -> Dict[Type[Criterion], CriterionConfiguration]:
        """
        Returns the currently loaded criteria configuration
        :return:
        """

        return self._criteria_config

    def get_criteria(self, defaults: Dict[Type[Criterion], float] = None) -> Sequence[Criterion]:
        """
        Returns a collection of criteria objects that are based the configuration provided
        to this instance

        :param: dictionary containing the default values for each criterion type.
            The default value for an unspecified criterion is `0`.
        :return: criteria objects
        """

        if defaults is None:
            defaults = {}

        criteria = []
        for criterion_class, criterion_cfg in self._criteria_config.items():
            c = criterion_class(
                weight=criterion_cfg.weight,
                raw_value=defaults.get(criterion_class, 0.0),
                upper_bound=criterion_cfg.upper_bound
            )
            criteria.append(c)

        return criteria


class FileCriteriaProvider(CriteriaProvider):
    """
    Class that provides parameterized criteria instances based on a .json configuration file.

    Such file has the following format::

        {
            "criterion_name1": {
                "weight": float
                "upper_bound": float
            },
            "criterion_name2": {
                "weight": float
                "upper_bound": float
            },
            ...
        }
    """

    def __init__(self, criteria_config_path: str | bytes | os.PathLike):
        """
        :param criteria_config_path: path to the criteria configuration file,
            containing the weights of each supported criteria
        """

        if not os.path.exists(criteria_config_path):
            raise FileNotFoundError(f"'{criteria_config_path}' is not a valid path to a criteria configuration file.")

        self._criteria_config_path: str | bytes | os.PathLike = criteria_config_path
        cfg = self._read_config_from_file()

        super(FileCriteriaProvider, self).__init__(criteria_config=cfg)
        
    def _read_config_from_file(self) -> Dict[Type[Criterion], CriterionConfiguration]:
        with open(self._criteria_config_path) as f:
            cfg_json = json.load(f)

            # Pair criteria names with their class (and constructor)
            criterion_classes = {
                "distance": DistanceCriterion,
                "arrival_time": ArrivalTimeCriterion,
                "co2": EmissionsCriterion,
                "transfers": TransfersCriterion,
            }

            criteria_cfg: Dict[Type[Criterion], CriterionConfiguration] = {}
            for name, criteria_info in cfg_json.items():
                weight = criteria_info["weight"]
                upper_bound = criteria_info["max"]
                
                criteria_cfg[criterion_classes[name]] = CriterionConfiguration(
                    weight=weight,
                    upper_bound=upper_bound
                )
                
            return criteria_cfg


@dataclass(frozen=True)
class LabelUpdate(Generic[_LabelType]):
    """
    Class that represents all the necessary data to update a label
    """

    boarding_stop: Stop
    """Stop at which the trip is boarded"""

    arrival_stop: Stop
    """Stop at which the trip is hopped off"""

    new_trip: Trip
    """New trip to board to get from `boarding_stop` to `arrival_stop`."""

    best_labels: MutableMapping[Stop, _LabelType]
    """
    Reference to the best labels for each stop, independent from the number of rounds.
    This data is needed by criteria that have a dependency on other labels to calculate their cost.
    (e.g. the distance cost of label x+1 depends on the distance cost of label x)
    """

# TODO this can be removed: WMC RAPTOR only keeps the best label each time a merge is done,
#   which means that the concept of bag is useless. We can directly compare the cost of the
#   old and the potentially new best label.
@dataclass(frozen=True)
class Bag:
    """
    Class that represents a container of label used in the MC RAPTOR algorithm
    """

    labels: List[MultiCriteriaLabel] = field(default_factory=list)
    updated: bool = False

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Bag({self.labels}, updated={self.updated})"

    def add(self, label: MultiCriteriaLabel):
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

    def get_best_label(self) -> MultiCriteriaLabel:
        """
        Returns the label with the best (lowest) cost in the bag
        :return:
        """

        if len(self.labels) == 0:
            raise Exception("There are no labels to retrieve the best from")

        by_cost_asc = list(sorted(self.labels, key=lambda l: l.total_cost))
        return by_cost_asc[0]

    def labels_with_trip(self):
        """All labels with trips, i.e. all labels that are reachable with a trip with given criterion"""
        return [lbl for lbl in self.labels if lbl.trip is not None]


def pareto_set(labels: List[MultiCriteriaLabel], keep_equal=False):
    """
    Find the pareto-efficient points

    :param labels: list with labels
    :param keep_equal: return also labels with equal criteria
    :return: list with pairwise non-dominating labels
    """

    is_efficient = np.ones(len(labels), dtype=bool)
    label_costs = np.array([label.total_cost for label in labels])
    for i, cost in enumerate(label_costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            if keep_equal:
                # keep point with all labels equal or one lower
                # Note: list1 < list2 determines if list1 is smaller than list2
                #   based on lexicographic ordering
                #   (i.e. the smaller list is the one with the smaller leftmost element)
                is_efficient[is_efficient] = np.any(
                    label_costs[is_efficient] < cost, axis=0
                ) + np.all(label_costs[is_efficient] == cost, axis=0)

            else:
                is_efficient[is_efficient] = np.any(
                    label_costs[is_efficient] < cost, axis=0
                )

            is_efficient[i] = True  # And keep self

    return list(compress(labels, is_efficient))
