from __future__ import annotations

import argparse
import webbrowser
from datetime import timedelta
from enum import Enum
from statistics import mean
from typing import List, Tuple, Mapping, Callable, Any

import attr
import folium
from folium import Map, Marker, PolyLine

from loguru import logger
from os import path

from pyraptor.model.structures import AlgorithmOutput, Leg, Stop, Coordinates
from pyraptor.util import TRANSFER_TYPE, get_transport_type_description

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


COLOR_DEPARTURE = 'blue'
COLOR_ARRIVAL = 'darkblue'


class LineTypeSetting:

    def __init__(self, color: str = 'black', weight: float = 1,
                 opacity: float = 1, dash_array: str = '1'):
        self.color = color
        self.weight = weight
        self.opacity = opacity
        self.dash_array = dash_array


# Value is a lambda producing a new MarkerSetting,
# this is why each icon must be referenced just one time
MARKER_SETTINGS: Mapping[MarkerType, Callable[[], MarkerSetting]] = {
    MarkerType.PublicStop:
        lambda: MarkerSetting(icon=folium.Icon(color='red', icon_color='white', icon='bus', prefix='fa')),
    MarkerType.RentingStation:
        lambda: MarkerSetting(icon=folium.Icon(color='green', icon_color='white', icon='bicycle', prefix='fa'))
}

LINE_TYPE_SETTINGS: Mapping[LineType, LineTypeSetting] = {
    LineType.Public: LineTypeSetting(color='red', weight=2.5, opacity=1, dash_array='1'),
    LineType.ShareMob: LineTypeSetting(color='green', weight=2, opacity=1, dash_array='8'),
    LineType.Walk: LineTypeSetting(color='blue', weight=2, opacity=0.8, dash_array='15')
}

class StopVisualizer(object):

    def __init__(self, stop: Stop):
        self.stop: Stop = stop
        self.arr: str | None = None
        self.dep: str | None = None

    @property
    def geo(self) -> Coordinates:
        return self.stop.geo

    @property
    def name(self) -> str:
        return f"{self.stop.name} "

    @property
    def is_start(self) -> bool:
        return self.arr is None and \
               self.dep is not None

    @property
    def is_end(self) -> bool:
        return self.arr is not None and \
               self.dep is None

    @property
    def arrival_departure_info(self) -> str:
        if self.is_start:
            return f"DEP: {self.dep} "
        elif self.is_end:
            return f"ARR: {self.arr} "
        elif self.arr == self.dep:
            return f"AT {self.dep} "
        else:
            return f"ARR: {self.arr}, DEP: {self.dep} "

    @property
    def info(self) -> str:
        return f'''{self.name}<br>{self.arrival_departure_info}'''

    @property
    def setting(self) -> MarkerSetting:
        mtype: MarkerType = MarkerType.PublicStop if type(self.stop) == Stop else MarkerType.RentingStation
        msetting: MarkerSetting = MARKER_SETTINGS[mtype]()
        if self.is_start:
            msetting.icon.options['markerColor'] = COLOR_DEPARTURE
        if self.is_end:
            msetting.icon.options['markerColor'] = COLOR_ARRIVAL
        return msetting

    @property
    def dep(self):
        return self._dep

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, seconds: int):
        self._arr = seconds_to_hour(seconds)

    @dep.setter
    def dep(self, seconds: int):
        self._dep = seconds_to_hour(seconds)


def seconds_to_hour(seconds: int) -> str:
    if seconds is not None:
        return str(timedelta(seconds=seconds))


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

    def add_stops(self):
        visualizers: Mapping[Stop, StopVisualizer] = {stop: StopVisualizer(stop) for stop in self.stops}
        for leg in self.legs:
            visualizers[leg.from_stop].dep = leg.dep
            visualizers[leg.to_stop].arr = leg.arr
        for vis in visualizers.values():
            self.put_marker(
                coord=vis.geo,
                text=vis.info,
                marker_setting=vis.setting
            )

    def add_moves(self):
        for leg in self.legs:
            # TODO create LineVisualizer class
            c1 = leg.from_stop.geo
            c2 = leg.to_stop.geo
            tt = leg.trip.route_info.transport_type
            if tt != TRANSFER_TYPE:
                desc = get_transport_type_description(tt)
                linet = LineType.Public
            else:
                desc = leg.trip.route_info.name
                linet = LineType.Walk if desc == 'walk path' else LineType.ShareMob
            self.draw_line(
                coord1=c1,
                coord2=c2,
                text=desc,
                line_setting=LINE_TYPE_SETTINGS[linet]
            )

    def put_marker(self, coord: Coordinates,
                   text: str | None = None, marker_setting: MarkerSetting = None):
        if marker_setting is None:
            marker_setting = MarkerSetting()
        marker = Marker(
            location=coord.to_list,
            tooltip=text,
            icon=marker_setting.icon
            # icon=folium.Icon(color="red", icon="info-sign")
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
    visualizer.add_stops()
    visualizer.add_moves()

    out_file_path = path.join(args.output_dir, 'algo_output.html')
    visualizer.save(path_=out_file_path, open_=True)
