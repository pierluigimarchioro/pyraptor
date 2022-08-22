from __future__ import annotations

import argparse
from os import path

from loguru import logger

from pyraptor.dao import read_timetable
from pyraptor.model.criteria import Criterion, DistanceCriterion
from pyraptor.model.output import Leg
from pyraptor.model.timetable import Stop, Trip, Coordinates
from pyraptor.visualization.folium_visualizer import StopVisualization, TripVisualization

OUT_FILE = "all_stops.html"


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Input directory containing timetable.pcl"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help=f"Path to save {OUT_FILE} file",
    )
    parser.add_argument(
        "-b",
        "--browser",
        type=bool,
        default=True,
        help="If True opens html in browser",
    )

    arguments = parser.parse_args()
    return arguments


def main(
        input: str,
        output: str,
        open_: bool
):
    logger.debug("Input timetable     : {}", input)
    logger.debug("Output directory    : {}", output)
    logger.debug("Open in browser     : {}", open_)

    timetable = read_timetable('data/output/milan')

    visualizer = TripVisualization(legs=[
        Leg(
            from_stop=Stop(geo=Coordinates(0, 0)),
            to_stop=Stop(geo=Coordinates(0, 0)),
            trip=Trip(),
            criteria=[DistanceCriterion(name='', weight=0, raw_value=0, upper_bound=0)]
        )
    ])

    viss = [StopVisualization(stop) for stop in timetable.stops]

    for vis in viss:
        visualizer.put_marker(
            vis.geo,
            vis.info,
            vis.setting
        )

    out_file_path = path.join(output, OUT_FILE)
    visualizer.save(path_=out_file_path, open_browser=open_)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        input=args.input,
        output=args.output,
        open_=args.browser
    )
