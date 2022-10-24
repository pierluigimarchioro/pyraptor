from __future__ import annotations

from collections.abc import Iterable
from typing import List

from loguru import logger

from pyraptor.model.algos.base import BaseMultiCriteriaRaptor
from pyraptor.model.timetable import (
    Stop,
)
from pyraptor.model.criteria import (
    MultiCriteriaLabel,
    ParetoBag,
    ArrivalTimeCriterion,
    DEFAULT_ORIGIN_TRIP
)
from pyraptor.util import LARGE_NUMBER


class MultiCriteriaRaptor(BaseMultiCriteriaRaptor[MultiCriteriaLabel, ParetoBag]):
    """
    Implementation of the MC RAPTOR algorithm described in the RAPTOR paper.
    """

    def _initialization(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int
    ) -> List[Stop]:
        criterion_types = self.criteria_factory.criteria_config.keys()

        # Initialize Round 0 with empty bags.
        # Following rounds are initialized by copying the previous one
        self.round_stop_bags[0] = {}
        for s in self.timetable.stops:
            # Initialize every criterion to infinite (the highest possible) cost
            with_infinite_cost = self.criteria_factory.create_criteria(
                defaults={
                    c_type: LARGE_NUMBER
                    for c_type in criterion_types
                }
            )
            self.round_stop_bags[0][s] = ParetoBag(
                labels=[MultiCriteriaLabel(criteria=with_infinite_cost)]
            )

        self.stop_forward_dependencies = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = set()

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        # Initialize origin stops labels, bags and dependencies
        for from_stop in from_stops:
            # Default arrival time for origin stops is the departure time
            # Only add default for arrival time criterion if defined
            with_departure_time = self.criteria_factory.create_criteria(
                defaults={ArrivalTimeCriterion: dep_secs} if ArrivalTimeCriterion in criterion_types else None
            )
            mc_label = MultiCriteriaLabel(
                arrival_time=dep_secs,
                boarding_stop=from_stop,
                trip=DEFAULT_ORIGIN_TRIP,
                criteria=with_departure_time
            )

            self.round_stop_bags[0][from_stop] = ParetoBag(labels=[mc_label])

            # From stop has a dependency on itself since it's an origin stop
            self.stop_forward_dependencies[from_stop] = {from_stop}

        marked_stops = [s for s in from_stops]
        return marked_stops
