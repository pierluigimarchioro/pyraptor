from __future__ import annotations

import argparse
import webbrowser
from enum import Enum
from statistics import mean
from typing import List, Tuple, Mapping, Callable, Any

import attr
import folium
from folium import Map, Marker, PolyLine

from loguru import logger
from os import path

from pyraptor.model.structures import AlgorithmOutput, Leg, Stop, Coordinates

FILE_NAME = 'algo_output.html'


class MarkerType(Enum):
    PublicStop = 'public'
    RentingStation = 'renting'


class LineType(Enum):
    Public = 'public'
    ShareMob = 'renting'
    Walk = 'walk'


class MarkerSetting:
    def __init__(self, icon: folium.Icon = None):
        if icon is not None:
            self.icon: folium.Icon = icon
        else:
            self.icon = folium.Icon()


class LineTypeSetting:

    def __init__(self, color: str='black', weight: float=1,
                 opacity: float=1, dash_array: str='1'):

        self.color = color
        self.weight = weight
        self.opacity = opacity
        self.dash_array = dash_array


MARKER_SETTINGS: Mapping[MarkerType, Callable[[], MarkerSetting]] = {
    MarkerType.PublicStop:
        lambda: MarkerSetting(icon=folium.Icon(color='red', icon_color='white', icon='bus', prefix='fa')),
    MarkerType.RentingStation:
        lambda: MarkerSetting(icon=folium.Icon(color='green', icon_color='white', icon='bicycle', prefix='fa'))
}


LINE_TYPE_SETTINGS = {
    LineType.Public: LineTypeSetting(color='red', weight=2.5, opacity=1, dash_array='1'),
    LineType.ShareMob: LineTypeSetting(color='green', weight=2, opacity=1, dash_array='8'),
    LineType.Walk: LineTypeSetting(color='blue', weight=2, opacity=0.8, dash_array='15')
}


class MapVisualizer:
    DEFAULT_ZOOM_START = 12

    def __init__(self, legs: List[Leg]):
        self.legs: List[Leg] = legs
        self.map_: Map = Map(location=list(self._mean_point), zoom_start=self.DEFAULT_ZOOM_START)

    @property
    def stops(self) -> List[Stop]:
        return [next(iter(self.legs)).from_stop] + [leg.to_stop for leg in self.legs]

    @property
    def _mean_point(self) -> Tuple[float, float]:
        lats = [stop.geo.lat for stop in self.stops]
        lons = [stop.geo.lon for stop in self.stops]
        return mean(lats), mean(lons)

    def put_marker(self, coord: Coordinates,
                   text: str | None = None, marker_setting: MarkerSetting = None):
        if marker_setting is None:
            marker_setting = MarkerSetting()
        marker = Marker(
            location=coord.to_list,
            popup=text,
            icon=marker_setting.icon
            #icon=folium.Icon(color="red", icon="info-sign")
        )
        marker.add_to(self.map_)

    def draw_line(self, coord1: Coordinates, coord2: Coordinates,
                  text: str | None = None, line_setting: LineTypeSetting = LineTypeSetting()):
        line = PolyLine(
            locations=[coord1.to_list, coord2.to_list],
            tooltip=text,
            color=line_setting.color,
            weight=line_setting.weight,
            opacity=line_setting.opacity,
            dash_array=line_setting.dash_array
        )
        line.add_to(self.map_)

    def save(self, path_: str, open_: bool = False):
        self.map_.save(path_)
        if open_:
            webbrowser.open(url=path_, new=2)


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

    logger.info(f"Loading Raptor algorithm output from {args.algo_output}")
    output: AlgorithmOutput = AlgorithmOutput.read_from_file(filepath=args.algo_output)

    visualizer = MapVisualizer(legs=output.journey.legs)

    c1 = Coordinates(45.455409820801286, 9.112903140348216)
    c2 = Coordinates(45.45941360104322, 9.129259707905655)
    c3 = Coordinates(45.451189635387, 9.12167044518899)

    visualizer.put_marker(c1, 'prima', MARKER_SETTINGS[MarkerType.PublicStop]())
    visualizer.put_marker(c2, 'seconda', MARKER_SETTINGS[MarkerType.PublicStop]())
    visualizer.put_marker(c3, 'terza')

    visualizer.draw_line(c1, c2, 'aaa', LINE_TYPE_SETTINGS[LineType.Public])
    visualizer.draw_line(c1, c3, 'bbb', LINE_TYPE_SETTINGS[LineType.ShareMob])
    visualizer.draw_line(c2, c3, 'ccc', LINE_TYPE_SETTINGS[LineType.Walk])

    out_file_path = path.join(args.output_dir, 'algo_output.html')
    visualizer.save(path_=out_file_path, open_=True)
