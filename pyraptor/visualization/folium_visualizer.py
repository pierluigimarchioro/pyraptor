from __future__ import annotations

import argparse
import webbrowser
from enum import Enum
from statistics import mean
from typing import List, Tuple, Mapping, Callable

import folium
from folium import Map, Marker, PolyLine

from loguru import logger
from os import path

from pyraptor.model.timetable import Stop, Coordinates, TransportType, SHARED_MOBILITY_TYPES, PUBLIC_TRANSPORT_TYPES
from pyraptor.model.output import AlgorithmOutput, Leg
from pyraptor.util import sec2str


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
    Rail = 'rail'
    Metro = 'metro'
    Bus = 'bus'
    ShareMobility = 'shared_mobility'
    Walk = 'walk'


""" Marker and Line Settings """

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
    LineType.Rail: LineSetting(color='#FF5100', weight=2.5, opacity=1, dash_array='1'),
    LineType.Bus: LineSetting(color='#181F62', weight=2.5, opacity=1, dash_array='1'),
    LineType.Metro: LineSetting(color='#F700FF', weight=2.5, opacity=1, dash_array='1'),
    LineType.ShareMobility: LineSetting(color='green', weight=2, opacity=1, dash_array='8'),
    LineType.Walk: LineSetting(color='blue', weight=2, opacity=0.8, dash_array='15')
}

""" Visualizers """


class StopVisualization(object):
    """
    This class represents the visualization of a Stop on the map
    """

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
        return (self.arr is None and
                self.dep is not None)

    @property
    def is_end(self) -> bool:
        """ Is the stop the last stop of the journey """
        return (self.arr is not None and
                self.dep is None)

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
        m_type: MarkerType = MarkerType.PublicStop if type(self.stop) == Stop else MarkerType.RentingStation
        m_setting: MarkerSetting = MARKER_SETTINGS[m_type]()
        if self.is_start:
            m_setting.icon.options['markerColor'] = COLOR_DEPARTURE
        if self.is_end:
            m_setting.icon.options['markerColor'] = COLOR_ARRIVAL
        return m_setting

    @property
    def dep(self):
        return self._dep

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, seconds: int):
        """ Convert seconds number to hour """
        self._arr = sec2str(seconds) if seconds is not None else None

    @dep.setter
    def dep(self, seconds: int):
        """ Convert seconds number to hour """
        self._dep = sec2str(seconds) if seconds is not None else None

    def add_to(self, trip_visualization: TripVisualization):
        trip_visualization.put_marker(
            coord=self.geo,
            text=self.info,
            marker_setting=self.setting
        )


class MovementVisualization:
    """
    This class represents the map visualization of a movement between two stops
    """

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
        t_type = self.transport_type
        if t_type == TransportType.Walk:
            return LineType.Walk
        elif t_type in SHARED_MOBILITY_TYPES:
            return LineType.ShareMobility
        elif t_type == TransportType.Metro:
            return LineType.Metro
        elif t_type == TransportType.Bus:
            return LineType.Bus
        elif t_type == TransportType.Rail:
            return LineType.Rail
        elif t_type in PUBLIC_TRANSPORT_TYPES:
            return LineType.PublicTransport
        else:
            raise ValueError(f"`{t_type}` is not a valid transport type")

    @property
    def setting(self) -> LineSetting:
        return LINE_SETTINGS[self.line_type]

    def add_to(self, trip_visualization: TripVisualization):
        trip_visualization.draw_line(
            coord1=self.from_coord,
            coord2=self.to_coord,
            text=self.info,
            line_setting=self.setting
        )


class TripVisualization:
    """
    Class that represents a trip visualization on a map
    """

    def __init__(self, legs: List[Leg]):
        self.legs: List[Leg] = legs
        self.map_: Map = Map(location=list(self._mean_point))
        self.map_.fit_bounds(self.bounds)  # to visualize
        self._add_stops()
        self._add_moves()

    @property
    def stops(self) -> List[Stop]:
        """ Returns all journeys stops """
        return list(set([leg.from_stop for leg in self.legs]).union([leg.to_stop for leg in self.legs]))

    @property
    def bounds(self) -> [[float, float], [float, float]]:
        """ Returns stops latitude and longitude bounds as [[min_lat, min_lon], [max_lat, max_lon]]"""
        geos = [(stop.geo.lat, stop.geo.lon) for stop in self.stops]
        return [list(min(geos)), list(max(geos))]

    @property
    def _mean_point(self) -> Tuple[float, float]:
        """ Returns mean latitude and longitude of journey stops"""
        lats = [stop.geo.lat for stop in self.stops]
        longs = [stop.geo.lon for stop in self.stops]
        return mean(lats), mean(longs)

    def _add_stops(self):
        """ Adds journey stops to map """
        visualizers: Mapping[Stop, StopVisualization] = {stop: StopVisualization(stop) for stop in self.stops}

        for leg in self.legs:
            visualizers[leg.from_stop].dep = leg.dep
            visualizers[leg.to_stop].arr = leg.arr

        for vis in visualizers.values():
            vis.add_to(trip_visualization=self)

    def _add_moves(self):
        """ Adds movement between stops to map """
        visualizers: List[MovementVisualization] = [MovementVisualization(leg) for leg in self.legs]
        for vis in visualizers:
            vis.add_to(self)

    def put_marker(
            self,
            coord: Coordinates,
            text: str | None = None,
            marker_setting: MarkerSetting = None
    ):
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

    def draw_line(
            self, coord1: Coordinates,
            coord2: Coordinates,
            text: str | None = None,
            line_setting: LineSetting = LineSetting()
    ):
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

    def save(self, path_: str, open_browser: bool = False):
        folium.TileLayer('openstreetmap').add_to(self.map_)
        folium.TileLayer('cartodbpositron').add_to(self.map_)
        folium.LayerControl().add_to(self.map_)

        logger.debug(f"Saving visualization to {path_}")
        self.map_.save(path_)
        if open_browser:
            logger.debug("Opening visualization in the browser")
            path_url = 'file:///' + path.abspath(path_)
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
    parser.add_argument(
        "-b",
        "--browser",
        type=bool,
        default=True,
        help="If True opens html in browser",
    )

    arguments = parser.parse_args()
    return arguments


def visualize_output(
        algo_output_path: str,
        visualization_dir: str,
        open_browser: bool
):
    """
    Saves the visualization of the provided output.

    :param algo_output_path: output to visualize
    :param visualization_dir: directory to save the visualizations in
    :param open_browser: if True, the visualization are opened on the browser
    """

    logger.debug("Algorithm path      : {}", algo_output_path)
    logger.debug("Output directory    : {}", visualization_dir)
    logger.debug("Open in browser     : {}", open_browser)

    logger.info(f"Visualizing {algo_output_path}")

    try:
        output: AlgorithmOutput = AlgorithmOutput.read_from_file(filepath=algo_output_path)
    except IOError as ex:
        raise Exception(f"An error occurred while trying to read {algo_output_path}: {ex}")

    for i, jrny in enumerate(output.journeys):
        visualization = TripVisualization(legs=jrny.legs)
        dep = jrny.legs[0].from_stop.name
        arr = jrny.legs[-1].to_stop.name

        out_file_path = path.join(visualization_dir, f"{dep}-{arr}_{i}.html")
        visualization.save(path_=out_file_path, open_browser=open_browser)


if __name__ == "__main__":
    args = parse_arguments()
    visualize_output(
        algo_output_path=args.algo_output,
        visualization_dir=args.output_dir,
        open_browser=args.browser
    )
