"""
# TODO aggiorna
Weighted implementation of the McRAPTOR algorithm described in the Microsoft paper.
In this implementation, each label has a cost, which is defined as the weighted sum
of all the criteria (i.e. distance, emissions, arrival time, etc.).
This means that the `dominates` relationship changes as follows:
X1 dominates X2 if X1 hasn't got a worse total cost than X2, where both
X1 and X2 are either Labels or Journeys.
This differs from the original implementation of McRAPTOR described in the paper,
which instead says that a label L1 dominates a label L2 if L1 is not worse
than L2 in any criterion.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple, Dict

from loguru import logger

from pyraptor.model.algos.base import SingleCriterionRaptor, SharedMobilityConfig
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import (
    Stop,
    RaptorTimetable
)
from pyraptor.model.criteria import (
    GeneralizedCostLabel,
    GeneralizedCostBag,
    CriteriaFactory,
    ArrivalTimeCriterion,
    DEFAULT_ORIGIN_TRIP
)
from pyraptor.util import LARGE_NUMBER


class GeneralizedCostRaptor(SingleCriterionRaptor[GeneralizedCostLabel, GeneralizedCostBag]):
    """
    Implementation of the Generalized Cost RAPTOR algorithm.
    Just a single criterion is optimized, that is, generalized cost, which is obtained
    as the weighted and normalized sum of a set of criteria.
    """

    def __init__(
            self,
            timetable: RaptorTimetable | RaptorTimetableSM,
            enable_fwd_deps_heuristic: bool,
            enable_sm: bool,
            sm_config: SharedMobilityConfig,
            criteria_provider: CriteriaFactory,
    ):
        """
        :param timetable: object containing the data that will be used by the algorithm
        :param criteria_provider: object that provides properly parameterized criteria for
            the algorithm to use
        """

        super(GeneralizedCostRaptor, self).__init__(
            timetable=timetable,
            enable_sm=enable_sm,
            sm_config=sm_config,
            enable_fwd_deps_heuristic=enable_fwd_deps_heuristic
        )

        self.criteria_provider: CriteriaFactory = criteria_provider
        """Object that provides properly parameterized criteria for
            the algorithm to use"""

    def _initialization(
            self,
            from_stops: Iterable[Stop],
            dep_secs: int
    ) -> List[Stop]:
        criterion_types = self.criteria_provider.criteria_config.keys()

        # Initialize Round 0 with empty bags.
        # Following rounds are initialized by copying the previous one
        self.round_stop_bags[0] = {}
        for p in self.timetable.stops:
            # Initialize every criterion to infinite (the highest possible) cost
            with_infinite_cost = self.criteria_provider.create_criteria(
                defaults={
                    c_type: LARGE_NUMBER
                    for c_type in criterion_types
                }
            )
            self.round_stop_bags[0][p] = GeneralizedCostBag(
                labels=[GeneralizedCostLabel(criteria=with_infinite_cost)]
            )

        self.stop_forward_dependencies: Dict[Stop, List[Stop]] = {}
        for s in self.timetable.stops:
            self.stop_forward_dependencies[s] = []

        logger.debug(f"Starting from Stop IDs: {str(from_stops)}")

        # Initialize origin stops labels, bags and dependencies
        for from_stop in from_stops:
            # Only add default for arrival time crit. if defined
            with_departure_time = self.criteria_provider.create_criteria(
                # Default arrival time for origin stops is the departure time
                defaults={ArrivalTimeCriterion: dep_secs} if ArrivalTimeCriterion in criterion_types else None
            )

            gc_label = GeneralizedCostLabel(
                arrival_time=dep_secs,
                boarding_stop=from_stop,
                trip=DEFAULT_ORIGIN_TRIP,
                criteria=with_departure_time
            )

            self.round_stop_bags[0][from_stop] = GeneralizedCostBag(labels=[gc_label])

            # From stop has a dependency on itself since it's an origin stop
            self.stop_forward_dependencies[from_stop] = [from_stop]

        marked_stops = [s for s in from_stops]
        return marked_stops
