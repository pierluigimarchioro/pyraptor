from __future__ import annotations

from collections.abc import Iterable
from typing import List

from loguru import logger

from pyraptor.model.algos.base import SingleCriterionRaptor
from pyraptor.model.timetable import Stop
from pyraptor.model.criteria import EarliestArrivalTimeLabel, EarliestArrivalTimeBag


class EarliestArrivalTimeRaptor(SingleCriterionRaptor[EarliestArrivalTimeLabel, EarliestArrivalTimeBag]):
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
                labels=[EarliestArrivalTimeLabel()]
            )

        self.stop_forward_dependencies = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = set()

        # Initialize bags with starting stops taking dep_secs to reach
        # Remember that dep_secs is the departure_time expressed in seconds
        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")
        marked_stops = []
        for s in from_stops:
            departure_label = EarliestArrivalTimeLabel(arrival_time=dep_secs)
            self.round_stop_bags[0][s] = EarliestArrivalTimeBag(labels=[departure_label])

            # From stop has a dependency on itself since it's an origin stop
            self.stop_forward_dependencies[s] = {s}

            marked_stops.append(s)

        return marked_stops
