""" Following script save into the same directory of timetable.pcl a json containing all station names.
    This file is used in the Demo to avoid heavy process of reloading timetable during debugging """
import argparse
import json
from os import path
from loguru import logger

from pyraptor.dao import read_timetable

NAMES_FILE = 'names.json'


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Input directory containing timetable.pcl",
    )

    arguments = parser.parse_args()
    return arguments


def main(timetable_folder: str):
    logger.debug("Input folder     : {}", timetable_folder)

    timetable = read_timetable(timetable_folder)
    names = [st.name.strip() for st in timetable.stations]
    names = sorted(names, key=lambda s: s.lower())

    out_file = path.join(timetable_folder, NAMES_FILE)
    names_json = {'names': names}

    logger.debug(f"Saving {len(names_json['names'])} to {out_file}")
    with open(out_file, 'w') as fp:
        json.dump(names_json, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(timetable_folder=args.input)
