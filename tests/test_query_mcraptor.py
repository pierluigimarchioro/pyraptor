"""Test Query McRaptor"""
from bdb import set_trace
from pyraptor import query_mcraptor
from pyraptor.model.mcraptor import pareto_set, Label
from pyraptor.model.timetable import Stop, Trip


def test_run_mcraptor_with_transfers_and_fares(timetable_with_transfers_and_fares):
    """
    Test run mcraptor with transfers and fares.

    Query from s2 to s4, starting at 00:20.
    This should yield 3 non-discriminating options for the timetable_with_fares:
        * s2-s7 with 201 + s7-s4 with 301, arrival time = 3.5, n_transfers = 1, fare = 0
        * s2-s4 with 401, arrival time = 3, n_transfers = 0, fare = 7
        * s2-s4 with 101, arrival time = 4, n_transfers = 0, fare = 0
    """

    origin_station = "8400002"
    destination_station = "8400004"
    departure_time = 1200
    rounds = 4

    journeys_to_destinations = query_mcraptor.run_mcraptor(
        timetable_with_transfers_and_fares,
        origin_station,
        departure_time,
        rounds,
        is_weighted_mc=False,
        criteria_file_path=""
    )

    journeys = journeys_to_destinations[destination_station]
    for jrny in journeys:
        jrny.print(departure_time)

    assert len(journeys) == 3, "should have 3 journeys"


def test_run_mcraptor_many_transfers(timetable_with_many_transfers):
    """Test run mcraptor"""

    origin_station = "8400004"
    destination_station = "8400014"
    departure_time = 0
    rounds = 4

    # Find route between two stations
    journeys_to_destinations = query_mcraptor.run_mcraptor(
        timetable_with_many_transfers,
        origin_station,
        departure_time,
        rounds,
        is_weighted_mc=False,
        criteria_file_path=""
    )
    journeys = journeys_to_destinations[destination_station]
    for jrny in journeys:
        jrny.to_list()
        jrny.print(departure_time)

    assert len(journeys) == 1, "should have 1 journey"


def test_pareto_set():
    """test creating pareto set"""

    stop = Stop(1, 1, "UT", "13")
    stop2 = Stop(1, 1, "UT", "14")

    label_0 = Label(earliest_arrival_time=1, n_trips=6, trip=Trip(id_=6), fare=0, from_stop=stop)
    label_1 = Label(earliest_arrival_time=1, n_trips=6, trip=Trip(id_=6), fare=0, from_stop=stop2)
    label_2 = Label(earliest_arrival_time=3, n_trips=4, trip=Trip(id_=4), fare=0, from_stop=stop)
    label_3 = Label(earliest_arrival_time=5, n_trips=1, trip=Trip(id_=1), fare=0, from_stop=stop)
    label_4 = Label(earliest_arrival_time=3, n_trips=5, trip=Trip(id_=5), fare=0, from_stop=stop)
    label_5 = Label(earliest_arrival_time=5, n_trips=3, trip=Trip(id_=3), fare=0, from_stop=stop)
    label_6 = Label(earliest_arrival_time=6, n_trips=1, trip=Trip(id_=1), fare=0, from_stop=stop)
    labels1 = pareto_set(
        [
            label_0,
            label_1,
            label_2,
            label_3,
            label_4,
            label_5,
            label_6,
        ]
    )
    labels2 = pareto_set(
        [
            label_0,
            label_1,
            label_2,
            label_3,
            label_4,
            label_5,
            label_6,
        ],
        keep_equal=True,
    )
    expected1 = [label_0, label_2, label_3]
    expected2 = [label_0, label_1, label_2, label_3]

    assert labels1 == expected1 and labels2 == expected2
