from __future__ import annotations

import argparse
import webbrowser

import folium

from loguru import logger
from os import path

from pyraptor.model.structures import AlgorithmOutput


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo_output",
        type=str,
        default="data/output/algo-output.pcl",
        help=".pcl file of an AlgorithmOutput instance",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data/output",
        help="Path to directory to save algo.html file",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    out_file = path.join(args.output_dir, 'algo_output.html')
    logger.info(f"Loading Raptor algorithm output from {args.algo_output}")
    output: AlgorithmOutput = AlgorithmOutput.read_from_file(filepath=args.algo_output)

    first_stop = output.journey.legs[0].from_stop
    m = folium.Map(location=list(first_stop.geo), zoom_start=12)

    folium.Marker(list(first_stop.geo), popup=first_stop.name).add_to(m)

    for leg in output.journey.legs:
        from_ = list(leg.from_stop.geo)
        to = list(leg.to_stop.geo)
        folium.Marker(to, popup=leg.to_stop.name).add_to(m)
        folium.PolyLine([from_, to], color="red", weight=2.5, opacity=1).add_to(m)


    logger.info(f"Saving to {out_file}")
    m.save(out_file)

    webbrowser.open(path.abspath(out_file), new=2)
