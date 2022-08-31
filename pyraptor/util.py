"""Utility functions"""
from __future__ import annotations

import os
from math import floor

import numpy as np

DEFAULT_TRANSFER_COST: int = 2 * 60  # Default transfer between stop in same station time is 2 minutes
LARGE_NUMBER: int = 2147483647  # Earliest arrival time at start of algorithm

MIN_DIST: float = 0.3  # Minimum distance in kilometers to consider transfer

# Average speed for some transport types [Km/h]
MEAN_FOOT_SPEED: float = 4.0
MEAN_BIKE_SPEED: float = 10.0
MEAN_ELECTRIC_BIKE_SPEED: float = 15.0
MEAN_CAR_SPEED: float = 50.0


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


def sec2str(seconds: int, show_sec: bool = False) -> str:
    """
    Convert hh:mm:ss to seconds since midnight

    :param show_sec: only show :ss if True
    :param seconds: Seconds to translate to hh:mm:ss
    """
    seconds = np.round(seconds)
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = int(seconds % 60)
    return (
        "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        if show_sec
        else "{:02d}:{:02d}".format(hours, minutes)
    )


def sec2minutes(seconds: float) -> str:
    """
    Returns a string
    :param seconds: number of seconds
    :return: minutes if not equal to zero and seconds
    """

    seconds: int = round(seconds)
    min_: int = floor(seconds / 60)
    sec: int = seconds % 60
    return f"{f'{min_} minutes and ' if min_>0 else ''}{sec} seconds"
