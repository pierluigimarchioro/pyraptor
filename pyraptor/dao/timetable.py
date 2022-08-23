"""Data access object for timetable"""
import os
from pathlib import Path

from loguru import logger
import joblib

from pyraptor.model.timetable import RaptorTimetable
from pyraptor.util import mkdir_if_not_exists


def read_timetable(input_folder: str) -> RaptorTimetable:
    """
    Read the timetable data from the cache directory
    """

    def load_joblib(name):
        logger.debug(f"Loading '{name}'")
        with open(Path(input_folder, f"{name}.pcl"), "rb") as handle:
            return joblib.load(handle)

    if not os.path.exists(input_folder):
        raise IOError(
            "PyRaptor timetable not found. Run `python pyraptor/gtfs/timetable.py`"
            " first to create timetable from GTFS files."
        )

    logger.debug("Using cached datastructures")

    timetable: RaptorTimetable = load_joblib("timetable")

    timetable.counts()

    return timetable


def write_timetable(output_folder: str, timetable: RaptorTimetable) -> None:
    """
    Write the timetable to output directory
    """

    def write_joblib(state, name):
        with open(Path(output_folder, f"{name}.pcl"), "wb") as handle:
            joblib.dump(state, handle)

    logger.info("Writing PyRaptor timetable to output directory")

    mkdir_if_not_exists(output_folder)
    write_joblib(timetable, "timetable")
