from __future__ import annotations

import json
import os
from abc import abstractmethod, ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import compress
from typing import List, Type, Dict, TypeVar, Generic

import numpy as np
from loguru import logger

from pyraptor.model.timetable import TransportType, Trip, Stop
from pyraptor.util import sec2str, LARGE_NUMBER


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

    trip: Trip
    """Trip to take to arrive at the destination stop at `earliest_arrival_time`"""

    boarding_stop: Stop
    """Stop at which the trip is boarded"""

    arrival_stop: Stop
    """Stop at which the trip is hopped off"""

    arrival_time: int
    """Earliest time to get to the destination stop by boarding the current trip"""

    # TODO label_update_log?

    def __init__(
            self,
            trip: Trip = None,
            boarding_stop: Stop = None,
            arrival_stop: Stop = None,
            arrival_time: int = LARGE_NUMBER
    ):
        """
        :param trip: trip associated to this label
        :param boarding_stop: stop that the associated trip is boarded at
        :param arrival_stop: stop that the associated trip is hopped off at
        :param arrival_time: time of arrival at the arrival stop
        """

        self.trip = trip
        self.boarding_stop = boarding_stop
        self.arrival_stop = arrival_stop
        self.arrival_time = arrival_time

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
    def is_strictly_dominating(self, other: BaseLabel) -> bool:
        """
        Returns true if the current label is *strictly dominating* the provided label,
        meaning that it is *strictly better* in at least one criterion.

        :param other: other label to compare
        :return: True if the current label is strictly dominating
        """
        pass


_LabelType = TypeVar("_LabelType", bound=BaseLabel)


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

    boarding_stop_label: _LabelType
    """Label associated to the boarding stop. Used to calculate criteria that rely
    on previous stops information (e.g. current_cost = previous_cost + K)"""


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
    def update(
            self,
            data: LabelUpdate,
            accumulated_criteria_provider: CriterionProvider
    ) -> Criterion:
        """
        Update the current criterion instance with the provided data

        :param data: data to update the criterion with
        :param accumulated_criteria_provider: instance that allows to retrieve the values
            accumulated for each optimized criteria up until this point in the journey
        :return: updated criterion instance
        """
        pass


# Forward Reference to Criterion class, defined afterwards
_C = TypeVar('_C', bound="Criterion")


class CriterionProvider(ABC):
    """
    Abstract class that defines an interface that allows to retrieve
    criterion instances of a specific type
    """

    @abstractmethod
    def get_criterion(
            self,
            criterion_class: Type[_C],
    ) -> _C:
        """
        Retrieve the criterion instance of the specified type
        :param criterion_class: type of the criterion to retrieve
        :return: criterion instance
        """
        pass


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

    def update(
            self,
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> DistanceCriterion:
        arrival_distance = self._get_total_arrival_distance(
            data=data,
            accumulated_criteria_provider=accumulated_criteria_provider
        )

        return DistanceCriterion(
            weight=self.weight,
            raw_value=arrival_distance,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_distance(
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> float:
        """
        Returns the updated distance (in km) for the criterion instance based on the
        new provided boarding and arrival stop. Such value represents the total travelled
        distance between the origin stop and the provided arrival stop.

        :param data: update data
        :param accumulated_criteria_provider: instance that allows to retrieve the values
            accumulated for each optimized criteria up until this point in the journey
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
        prev_stop_dist_criterion = accumulated_criteria_provider.get_criterion(
            criterion_class=DistanceCriterion
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

    def update(
            self,
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> EmissionsCriterion:
        arrival_emissions = self._get_total_arrival_emissions(
            data=data,
            accumulated_criteria_provider=accumulated_criteria_provider
        )

        return EmissionsCriterion(
            weight=self.weight,
            raw_value=arrival_emissions,
            upper_bound=self.upper_bound
        )

    @staticmethod
    def _get_total_arrival_emissions(
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> float:
        """
        Returns the updated total emissions (in co2 grams / passenger km) for
        this criterion instance, based on the new provided boarding and arrival stop.
        Such value represents the total emissions between the origin stop
        and the provided arrival stop.

        :param data: update data
        :param accumulated_criteria_provider: instance that allows to retrieve the values
            accumulated for each optimized criteria up until this point in the journey
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

        prev_stop_emissions_crit = accumulated_criteria_provider.get_criterion(
            criterion_class=EmissionsCriterion
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

    def update(
            self,
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider = None
    ) -> ArrivalTimeCriterion:
        new_arrival_time = data.new_trip.get_stop_time(data.arrival_stop).dts_arr

        # This error might happen if the timetable was generated with incorrect data
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
    This means that to K trips equal K-1 transfers.
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

    def update(
            self,
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> TransfersCriterion:
        # The transfer counter is updated only if:
        # - there is a trip change (new != old) and
        #       the old isn't the initial trip (origin trip)
        best_boarding_label = data.boarding_stop_label
        add_new_transfer = (data.new_trip != best_boarding_label.trip
                            and best_boarding_label.trip != DEFAULT_ORIGIN_TRIP)

        transfers_at_boarding = accumulated_criteria_provider.get_criterion(
            criterion_class=TransfersCriterion
        )

        return TransfersCriterion(
            weight=self.weight,
            raw_value=transfers_at_boarding.raw_value if not add_new_transfer else transfers_at_boarding.raw_value + 1,
            upper_bound=self.upper_bound
        )


class GeneralizedCostCriterion(Criterion, CriterionProvider):
    criteria: Sequence[Criterion]
    """
    Set of criteria used to calculate the generalized cost
    """

    def __init__(
            self,
            criteria: Sequence[Criterion],
            weight: float = 1.0,
            upper_bound: float = LARGE_NUMBER,
    ):
        super(GeneralizedCostCriterion, self).__init__(
            name="Generalized Cost",
            weight=weight,
            raw_value=0.0,
            upper_bound=upper_bound
        )

        self.criteria = criteria

        # Calculate and set generalized cost on init
        self.raw_value = sum(self.criteria, start=0.0)

    def update(
            self,
            data: LabelUpdate[MultiCriteriaLabel],
            accumulated_criteria_provider: CriterionProvider
    ) -> GeneralizedCostCriterion:
        assert len(self.criteria) != 0, "Trying to update an instance with no criteria set"

        # The accumulated criteria MUST be retrieved from a GeneralizedCostCriterion instance,
        #   because the criteria used to calculate the generalized cost may differ
        #   from the whole set of optimized criteria
        # E.g. MultiCriteriaLabel.criteria == { A, B, GeneralizedCost } and
        #   GeneralizedCost.criteria == { C, D, E }
        if not isinstance(accumulated_criteria_provider, GeneralizedCostCriterion):
            accumulated_criteria_provider = data.boarding_stop_label.get_criterion(GeneralizedCostCriterion)

        updated_criteria = []
        for c in self.criteria:
            updated_c = c.update(
                data=data,
                accumulated_criteria_provider=accumulated_criteria_provider
            )
            updated_criteria.append(updated_c)

        return GeneralizedCostCriterion(
            criteria=updated_criteria,
            weight=self.weight,
            upper_bound=self.upper_bound
        )

    def get_criterion(
            self,
            criterion_class: Type[_C],
    ) -> _C:
        """
        Returns the instance of the specified type of criterion from the current
        Generalized Cost Criterion instance.

        :param criterion_class: type of the criterion to retrieve
        :return: instance of the specified criterion type
        """

        criterion = next(
            filter(lambda c: isinstance(c, criterion_class), self.criteria),
            None
        )
        assert criterion is not None, (f"The current {GeneralizedCostCriterion.__name__} instance does "
                                       f"not include a criterion of type {criterion_class.__name__}")

        return criterion


class MultiCriteriaLabel(BaseLabel, CriterionProvider):
    """
    Class that represents a multi-criteria label.

    The concept this is class is modeled after is that of the multi-label in the
    `McRAPTOR` section of the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf)
    """

    arrival_stop: Stop
    """Stop that this label is associated to"""

    # TODO what if it was a mapping for easier retrieval?
    criteria: Sequence[Criterion]
    """Set of criteria to optimize"""

    def __init__(
            self,
            trip: Trip = None,
            boarding_stop: Stop = None,
            arrival_stop: Stop = None,
            arrival_time: int = LARGE_NUMBER,
            criteria: Sequence[Criterion] = None
    ):
        """
        :param trip: trip associated to this label
        :param boarding_stop: stop that the associated trip is boarded at
        :param arrival_stop: stop that the associated trip is hopped off at
        :param arrival_time: time of arrival at the arrival stop
        :param criteria: set of criteria to optimize
        """

        super(MultiCriteriaLabel, self).__init__(
            trip=trip,
            boarding_stop=boarding_stop,
            arrival_stop=arrival_stop,
            arrival_time=arrival_time
        )

        self.criteria = criteria

    def update(self, data: LabelUpdate[MultiCriteriaLabel]) -> MultiCriteriaLabel:
        assert len(self.criteria) != 0, "Trying to update an instance with no criteria set"

        updated_criteria = []
        for c in self.criteria:
            updated_c = c.update(
                data=data,
                accumulated_criteria_provider=data.boarding_stop_label
            )
            updated_criteria.append(updated_c)

        updated_trip = data.new_trip if data.new_trip is not None else self.trip
        updated_dep_stop = data.boarding_stop if data.boarding_stop is not None else self.boarding_stop
        updated_arr_stop = data.arrival_stop if data.arrival_stop is not None else self.arrival_stop

        # Earliest arrival time to the arrival stop on the updated trip
        updated_arr_time = updated_trip.get_stop_time(updated_arr_stop).dts_arr

        return MultiCriteriaLabel(
            arrival_time=updated_arr_time,
            boarding_stop=updated_dep_stop,
            arrival_stop=updated_arr_stop,
            criteria=updated_criteria,
            trip=updated_trip,
        )

    def get_criterion(
            self,
            criterion_class: Type[_C],
    ) -> _C:
        """
        Returns the instance of the specified type of criterion from the current label.

        :param criterion_class: type of the criterion to retrieve
        :return: instance of the specified criterion type
        """

        criterion = next(
            filter(lambda c: isinstance(c, criterion_class), self.criteria),
            None
        )
        assert criterion is not None, (f"The current label does not include "
                                       f"a criterion of type {criterion_class.__name__}")

        return criterion

    def is_strictly_dominating(self, other: MultiCriteriaLabel) -> bool:
        # TODO strict domination here? is this method even needed?
        #    after implementing McRAPTOR, consider removing if unused
        return self.criteria < other.criteria


class EarliestArrivalTimeLabel(MultiCriteriaLabel):
    """
    Class that represents a label used in the Earliest Arrival Time RAPTOR variant
    described in the RAPTOR paper
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf).

    Arrival time is the only optimization criterion.
    """

    at_criterion: ArrivalTimeCriterion
    """Arrival Time Criterion instance used to calculate updated arrival times.
    This is the only criterion optimized in labels of this type."""

    def __init__(
            self,
            trip: Trip = None,
            boarding_stop: Stop = None,
            arrival_stop: Stop = None,
            arrival_time: int = LARGE_NUMBER
    ):
        """
        :param trip: trip associated to this label
        :param boarding_stop: stop that the associated trip is boarded at
        :param arrival_stop: stop that the associated trip is hopped off at
        :param arrival_time: time of arrival at the arrival stop
        """

        at_criterion = ArrivalTimeCriterion(raw_value=arrival_time)
        super(EarliestArrivalTimeLabel, self).__init__(
            trip=trip,
            boarding_stop=boarding_stop,
            arrival_stop=arrival_stop,
            arrival_time=arrival_time,
            criteria=[at_criterion]  # Arrival Time is the only optimized criterion here
        )

        self.at_criterion = at_criterion

    def is_strictly_dominating(self, other: EarliestArrivalTimeLabel) -> bool:
        return self.arrival_time < other.arrival_time

    def __repr__(self) -> str:
        return (f"{EarliestArrivalTimeLabel.__name__}(earliest_arrival_time={self.arrival_time}, "
                f"trip={self.trip}, boarding_stop={self.boarding_stop})")


class GeneralizedCostLabel(MultiCriteriaLabel):
    """
    Label that considers only generalized cost as optimization criterion
    """

    gc_criterion: GeneralizedCostCriterion
    """Generalized cost criterion instance used to calculate generalized cost.
    This is the only criterion optimized in labels of this type."""

    def __init__(
            self,
            trip: Trip = None,
            boarding_stop: Stop = None,
            arrival_stop: Stop = None,
            arrival_time: int = LARGE_NUMBER,
            criteria: Sequence[Criterion] = None
    ):
        """
        :param trip: trip associated to this label
        :param boarding_stop: stop that the associated trip is boarded at
        :param arrival_stop: stop that the associated trip is hopped off at
        :param arrival_time: time of arrival at the arrival stop
        :param criteria: criteria used to calculate the generalized cost
        """

        gc_criterion = GeneralizedCostCriterion(criteria=criteria)
        super(GeneralizedCostLabel, self).__init__(
            trip=trip,
            boarding_stop=boarding_stop,
            arrival_stop=arrival_stop,
            arrival_time=arrival_time,
            criteria=[gc_criterion]  # GC is the only optimized criterion here
        )

        self.gc_criterion = gc_criterion

    @property
    def generalized_cost(self) -> float:
        """
        Returns the total cost assigned to the label, which corresponds to
        the weighted sum of its criteria.
        :return: float instance representing the total cost
        """

        return self.gc_criterion.cost

    def update(self, data: LabelUpdate[GeneralizedCostLabel]) -> GeneralizedCostLabel:
        updated_trip = data.new_trip if data.new_trip is not None else self.trip
        updated_dep_stop = data.boarding_stop if data.boarding_stop is not None else self.boarding_stop
        updated_gc_criterion = self.gc_criterion.update(
            data=data,
            accumulated_criteria_provider=data.boarding_stop_label.gc_criterion
        )

        # Earliest arrival time to the arrival stop on the updated trip
        updated_arr_time = updated_trip.get_stop_time(data.arrival_stop).dts_arr

        return GeneralizedCostLabel(
            arrival_time=updated_arr_time,
            boarding_stop=updated_dep_stop,
            arrival_stop=data.arrival_stop,
            trip=updated_trip,
            criteria=updated_gc_criterion.criteria
        )

    def is_strictly_dominating(self, other: GeneralizedCostLabel) -> bool:
        return self.generalized_cost < other.generalized_cost


@dataclass(frozen=True)
class Bag(ABC, Generic[_LabelType]):
    """
    Abstract class that defines a bag (collection) of labels associated to some stop.
    """

    labels: List[_LabelType] = field(default_factory=list)
    """Collection of labels associated to this bag"""

    improved: bool = False
    """True if this bag was created after a merging that brought some improvements"""

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Bag({self.labels})"

    @abstractmethod
    def merge(self, with_labels: List[_LabelType]) -> Bag:
        """
        Returns a new bag instance updated with the provided labels.

        :param with_labels: labels to merge with
        :return: bag that contains only the best labels between the
            ones provided and the ones in the current instance
        """
        pass


# Suppress warning that SingleLabelBag must implement ABC
# noinspection PyAbstractClass


class SingleLabelBag(Bag[_LabelType]):
    """
    Abstract class that represents a bag that always contains at most one label.
    """

    def get_label(self) -> _LabelType | None:
        """
        Returns the only label contained in this bag, or None if there aren't any.
        :return: label instance or None if there are no labels
        """

        assert len(self.labels) <= 1, "There should never be more than one label"

        return self.labels[0] if len(self.labels) > 0 else None


class EarliestArrivalTimeBag(SingleLabelBag[EarliestArrivalTimeLabel]):
    """
    Label Bag used in Earliest Arrival Time RAPTOR
    """

    def merge(self, with_labels: List[EarliestArrivalTimeLabel]) -> EarliestArrivalTimeBag:
        """
        Returns an updated bag containing the label with the best arrival time

        :param with_labels: labels to update the bag with
        :return:
        """

        if len(with_labels) == 0 and len(self.labels) == 0:
            return EarliestArrivalTimeBag(improved=False)

        prev_best = (
            None
            if len(self.labels) == 0
            else min(self.labels, key=lambda lbl: lbl.arrival_time)
        )
        best_label = min(self.labels + with_labels, key=lambda lbl: lbl.arrival_time)

        was_improved = prev_best != best_label
        return EarliestArrivalTimeBag(labels=[best_label], improved=was_improved)


class GeneralizedCostBag(SingleLabelBag[GeneralizedCostLabel]):
    """
    Label Bag used in Generalized Cost RAPTOR
    """

    def merge(self, with_labels: List[GeneralizedCostLabel]) -> GeneralizedCostBag:
        """
        Returns an updated bag containing the label with the lowest generalized cost

        :param with_labels: labels to update the bag with
        :return:
        """

        if len(with_labels) == 0 and len(self.labels) == 0:
            return GeneralizedCostBag(improved=False)

        prev_best = (
            None
            if len(self.labels) == 0
            else min(self.labels, key=lambda lbl: lbl.generalized_cost)
        )
        best_label = min(self.labels + with_labels, key=lambda lbl: lbl.generalized_cost)

        was_improved = prev_best != best_label
        return GeneralizedCostBag(labels=[best_label], improved=was_improved)


@dataclass(frozen=True)
class ParetoBag(Bag[MultiCriteriaLabel]):
    """
    Class that represents a Pareto-set of labels, i.e. a set of pair-wise non-dominating labels,
    where no label is worse than any other in at least one criterion
    """

    # TODO this class is currently not used, but will be when McRaptor will be implemented again.
    #   Remember to check if all the methods are useful/properly implemented

    def __repr__(self):
        return f"ParetoBag({self.labels}, updated={self.improved})"

    def add(self, label: MultiCriteriaLabel):
        """Add"""

        self.labels.append(label)

    def merge(self, with_labels: List[MultiCriteriaLabel]) -> ParetoBag:
        pareto_labels = self.labels + with_labels

        if len(pareto_labels) == 0:
            return ParetoBag(labels=[], improved=False)

        pareto_labels = pareto_set(pareto_labels)
        bag_update = True if pareto_labels != self.labels else False

        return ParetoBag(labels=pareto_labels, improved=bag_update)

    def labels_with_trip(self):
        """
        All labels with trips, i.e. all labels that are reachable with a trip with given criterion
        """

        return [lbl for lbl in self.labels if lbl.trip is not None]


def pareto_set(labels: List[MultiCriteriaLabel], keep_equal=False):
    """
    Find the pareto-efficient points

    :param labels: list with labels
    :param keep_equal: return also labels with equal criteria
    :return: list with pairwise non-dominating labels
    """

    best_mask = np.ones(len(labels), dtype=bool)
    all_labels_criteria = np.array([label.criteria for label in labels])
    for current_lbl_idx, criteria in enumerate(all_labels_criteria):
        if best_mask[current_lbl_idx]:
            # Here the sets of criteria of each label are compared with all the others
            # This is strict domination, where at least one criterion must be better for
            #   a label to be considered dominating (not the `np.any` and the strict `<`)
            # `keep_equal` option also keeps labels with an equal set of criteria
            if keep_equal:
                best_mask[best_mask] = np.any(
                    all_labels_criteria[best_mask] < criteria, axis=1
                ) + np.all(all_labels_criteria[best_mask] == criteria, axis=1)

            else:
                best_mask[best_mask] = np.any(
                    all_labels_criteria[best_mask] < criteria, axis=1
                )

            # Keep the label that has been compared with all the others
            best_mask[current_lbl_idx] = True

    return list(compress(labels, best_mask))


@dataclass
class CriterionConfiguration:
    """
    Class that represents the configuration for some criterion.
    It defines weight and upper_bound parameters.
    """

    weight: float
    upper_bound: float


class CriteriaFactory:
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

    def create_criteria(self, defaults: Dict[Type[Criterion], float] = None) -> Sequence[Criterion]:
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


class FileCriteriaProvider(CriteriaFactory):
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
