"""Utility functions"""
import os
import numpy as np


TRANSFER_COST = 2 * 60  # Default transfer time is 2 minutes
LARGE_NUMBER = 2147483647  # Earliest arrival time at start of algorithm
TRANSFER_TRIP = None  # TODO remove when Trip.get_transfer_trip is implemented


def mkdir_if_not_exists(name: str) -> None:
    """Create directory if not exists"""
    if not os.path.exists(name):
        os.makedirs(name)


def str2sec(time_str: str) -> int:
    """
    Convert hh:mm:ss to seconds since midnight
    :param time_str: String in format hh:mm:ss
    """
    split_time = time_str.strip().split(":")
    if len(split_time) == 3:
        # Has seconds
        hours, minutes, seconds = split_time
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    hour, minutes = split_time
    return int(hour) * 3600 + int(minutes) * 60


def sec2str(scnds: int, show_sec: bool = False) -> str:
    """
    Convert hh:mm:ss to seconds since midnight

    :param show_sec: only show :ss if True
    :param scnds: Seconds to translate to hh:mm:ss
    """
    scnds = np.round(scnds)
    hours = int(scnds / 3600)
    minutes = int((scnds % 3600) / 60)
    seconds = int(scnds % 60)
    return (
        "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        if show_sec
        else "{:02d}:{:02d}".format(hours, minutes)
    )


WALK_TRANSPORT_TYPE = -1


def get_transport_type_description(transport_type: int) -> str:
    """
    Returns a description for the provided transport type,
    which is the route_type attribute of the routes.txt GTFS table.

    :param transport_type: integer code for a transport type
    :return: transport type description
    """

    # TODO maybe refactor transport_type to enum?
    transport_descriptions = {
        WALK_TRANSPORT_TYPE: "Walk",
        0: "Light Rail (e.g. Tram)",
        1: "Metro",
        2: "Rail",
        3: "Bus",
        4: "Ferry",
        5: "Cable Tram",
        6: "Aerial Lift",
        7: "Funicular",
        11: "Trolleybus",
        12: "Monorail",
    }

    return transport_descriptions[transport_type]
