from __future__ import annotations

import argparse
import webbrowser
from datetime import timedelta
from enum import Enum
from statistics import mean
from typing import List, Tuple, Mapping, Callable

import folium
from folium import Map, Marker, PolyLine

from loguru import logger
from os import path

from pyraptor.model.timetable import Stop, Coordinates, TransportType, SHARED_MOBILITY_TYPES, PUBLIC_TRANSPORT_TYPES
from pyraptor.model.output import AlgorithmOutput, Leg

FILE_NAME = 'algo_output.html'


def seconds_to_hour(seconds: int) -> str:
    """ Convert number of seconds to string hour"""
    if seconds is not None:
        return str(timedelta(seconds=seconds))


""" Marker and Line Types"""


class MarkerType(Enum):
    """ This class represent all possible types
        of a point on the map """
    PublicStop = 'public_transport'
    RentingStation = 'shared_mobility'


class LineType(Enum):
    """ This class represent all possible types
        of a conjunction between two points on the map """
    PublicTransport = 'public_transport'
    ShareMobility = 'shared_mobility'
    Walk = 'walk'


""" Marker and Line Setting """

COLOR_DEPARTURE = 'blue'  # default color for departure
COLOR_ARRIVAL = 'darkblue'  # default color for arrival


class MarkerSetting:
    """ Class representing style settings for a Marker"""

    def __init__(self, icon: folium.Icon = None):
        self.icon: folium.Icon = icon if icon is not None else folium.Icon()


class LineSetting:
    """ Class representing style settings for a Line """

    def __init__(self, color: str = 'black', weight: float = 1,
                 opacity: float = 1, dash_array: str = '1'):
        self.color = color
        self.weight = weight
        self.opacity = opacity
        self.dash_array = dash_array


""" Marker and Line Default Settings """

# Value is a lambda producing a new MarkerSetting,
# this is why each icon must be referenced just once
MARKER_SETTINGS: Mapping[MarkerType, Callable[[], MarkerSetting]] = {
    MarkerType.PublicStop:
        lambda: MarkerSetting(icon=folium.Icon(color='red', icon_color='white', icon='bus', prefix='fa')),
    MarkerType.RentingStation:
        lambda: MarkerSetting(icon=folium.Icon(color='green', icon_color='white', icon='bicycle', prefix='fa'))
}

LINE_SETTINGS: Mapping[LineType, LineSetting] = {
    LineType.PublicTransport: LineSetting(color='red', weight=2.5, opacity=1, dash_array='1'),
    LineType.ShareMobility: LineSetting(color='green', weight=2, opacity=1, dash_array='8'),
    LineType.Walk: LineSetting(color='blue', weight=2, opacity=0.8, dash_array='15')
}

""" Visualizers """


class StopVisualizer(object):
    """  This class represents a Stop on the map """

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
        """ Is the stop the first stop of the journey """
        return self.arr is None and \
               self.dep is not None

    @property
    def is_end(self) -> bool:
        """ Is the stop the last stop of the journey """
        return self.arr is not None and \
               self.dep is None

    @property
    def arrival_departure_info(self) -> str:
        """ Returns arrival and departure infos as a string"""
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
        """ Combines both name and arrival-departure infos """
        return f'''{self.name}<br>{self.arrival_departure_info}'''

    @property
    def setting(self) -> MarkerSetting:
        """ Returns setting basing on stop infos """
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
        """ Convert seconds number to hour """
        self._arr = seconds_to_hour(seconds)

    @dep.setter
    def dep(self, seconds: int):
        """ Convert seconds number to hour """
        self._dep = seconds_to_hour(seconds)

    def add_to(self, map_visualizer: MapVisualizer):
        map_visualizer.put_marker(
            coord=self.geo,
            text=self.info,
            marker_setting=self.setting
        )


class MovementVisualizer:
    """  This class represents a Movement on the map between two stops """

    def __init__(self, leg: Leg):
        self.leg: Leg = leg

    @property
    def from_coord(self) -> Coordinates:
        return self.leg.from_stop.geo

    @property
    def to_coord(self) -> Coordinates:
        return self.leg.to_stop.geo

    @property
    def transport_type(self) -> TransportType:
        return self.leg.trip.route_info.transport_type

    @property
    def transport_info(self) -> str:
        return self.leg.trip.route_info.name

    @property
    def info(self) -> str:
        if self.transport_type in PUBLIC_TRANSPORT_TYPES:
            return f'''{str(self.transport_type)}<br>{self.transport_info}'''
        else:
            return str(self.transport_type)

    @property
    def line_type(self) -> LineType:
        tt = self.transport_type
        if tt == TransportType.Walk:
            return LineType.Walk
        elif tt in SHARED_MOBILITY_TYPES:
            return LineType.ShareMobility
        elif tt in PUBLIC_TRANSPORT_TYPES:
            return LineType.PublicTransport
        else:
            raise ValueError(f"No valid {tt} transport")

    @property
    def setting(self) -> LineSetting:
        return LINE_SETTINGS[self.line_type]

    def add_to(self, map_visualizer: MapVisualizer):
        map_visualizer.draw_line(
            coord1=self.from_coord,
            coord2=self.to_coord,
            text=self.info,
            line_setting=self.setting
        )


class MapVisualizer:
    """ Map to visualize the trip """

    def __init__(self, legs: List[Leg]):
        self.legs: List[Leg] = legs
        self.map_: Map = Map(location=list(self._mean_point))
        self.map_.fit_bounds(self.bounds)  # to visualize

    @property
    def stops(self) -> List[Stop]:
        """ Returns all journeys stops """
        return [next(iter(self.legs)).from_stop] + [leg.to_stop for leg in self.legs]

    @property
    def bounds(self) -> [[float, float], [float, float]]:
        """ Returns stops latitude and longitude bounds as [[min_lat, min_lon], [max_lat, max_lon]]"""
        geos = [(stop.geo.lat, stop.geo.lon) for stop in self.stops]
        return [list(min(geos)), list(max(geos))]

    @property
    def _mean_point(self) -> Tuple[float, float]:
        """ Returns mean latitude and longitude of journey stops"""
        lats = [stop.geo.lat for stop in self.stops]
        lons = [stop.geo.lon for stop in self.stops]
        return mean(lats), mean(lons)

    def add_stops(self):
        """ Adds journey stops to map """
        visualizers: Mapping[Stop, StopVisualizer] = {stop: StopVisualizer(stop) for stop in self.stops}
        for leg in self.legs:
            visualizers[leg.from_stop].dep = leg.dep
            visualizers[leg.to_stop].arr = leg.arr
        for vis in visualizers.values():
            vis.add_to(map_visualizer=self)

    def add_moves(self):
        """ Adds movement between stops to map """
        visualizers: List[MovementVisualizer] = [MovementVisualizer(leg) for leg in self.legs]
        for vis in visualizers:
            vis.add_to(self)

    def put_marker(self, coord: Coordinates,
                    text: str | None = None, marker_setting: MarkerSetting = None):
        """ Creates a marker on the map """
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
                   text: str | None = None, line_setting: LineSetting = LineSetting()):
        """ Creates a line on the map """
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
            path_url = 'file:///'+path.abspath(path_)
            webbrowser.open(url=path_url, new=1)


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
