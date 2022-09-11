from __future__ import annotations

import json
from abc import abstractmethod, ABC
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import List, Dict, Tuple
from urllib.request import urlopen

import attr
from loguru import logger

from pyraptor.model.timetable import Stop, Stops, TransportType, Transfers, Transfer, Coordinates, Station, \
    RaptorTimetable, TRANSPORT_TYPE_SPEEDS


@dataclass
class RaptorTimetableSM(RaptorTimetable):
    """Timetable data"""

    shared_mobility_feeds: List[SharedMobilityFeed] = None
    vehicle_transfers: VehicleTransfers = None

    def counts(self) -> None:
        """Prints timetable counts"""
        super().counts()
        logger.debug("VTransfers : {}", len(self.vehicle_transfers))
        logger.debug([str(smf) for smf in self.shared_mobility_feeds])


@attr.s(cmp=False, repr=False)
class RentingStation(Stop, ABC):
    """
    Interface representing a Renting Station used
    This class represents a Physical Renting Station used in urban network for shared mobility
    """
    system_id: str = attr.ib(default=None)
    """Shared mobility system identifier"""

    transport_types: List[TransportType] = attr.ib(default=None)
    """Types of vehicle rentable in the station"""

    @property
    @abstractmethod
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        pass

    @property
    @abstractmethod
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        pass


@attr.s(cmp=False, repr=False)
class RentingStations(Stops, ABC):
    """
    Interface representing a set of renting stations
    """

    system_id: str = attr.ib(default=None)
    system_transport_types: List[TransportType] = attr.ib(default=None)

    @property
    def no_source(self) -> List[RentingStation]:
        """ Returns all renting stations with no available vehicles for departure """
        return [s for s in self if not s.valid_source]

    @property
    def no_destination(self) -> List[RentingStation]:
        """ Returns all renting stations with no available docks for arrival """
        return [s for s in self if not s.valid_destination]

    @abstractmethod
    def init_download(self):
        """ Downloads static datas """
        pass

    @abstractmethod
    def update(self):
        """ Update datas using real-time feeds  """
        pass


@attr.s(cmp=False, repr=False)
class PhysicalRentingStation(RentingStation):
    capacity: int = attr.ib(default=0)
    vehicles_available: int = attr.ib(default=0)  # Available vehicles number (real-time)
    docks_available: int = attr.ib(default=0)  # Available docks number (real-time)
    is_installed: bool = attr.ib(default=False)  # Station currently on the street (real-time)
    is_renting: bool = attr.ib(default=False)  # Station renting vehicles (real-time)
    is_returning: bool = attr.ib(default=False)  # Station accepting vehicles returns (real-time)

    @property
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        valid = (self.vehicles_available > 0 and
                 self.is_installed and
                 self.is_renting)
        return valid

    @property
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        valid = (self.vehicles_available < self.capacity and
                 self.docks_available > 0 and
                 self.is_returning)
        return valid


@attr.s(cmp=False, repr=False)
class PhysicalRentingStations(RentingStations):
    # New dictionaries types
    set_idx: Dict[str, RentingStation] = dict()
    set_index: Dict[int, RentingStation] = dict()
    last_index: int = 1

    station_info_url: str = attr.ib(default=None)
    station_status_url: str = attr.ib(default=None)

    """ Override superclass methods with stub, subsuming to RentingStations """

    def get_stop(self, stop_id) -> PhysicalRentingStation:
        return super(PhysicalRentingStations, self).get_stop(stop_id)

    def get_by_index(self, stop_index) -> PhysicalRentingStation:
        """Get stop by index"""
        return super(PhysicalRentingStations, self).get_by_index(stop_index)

    def add_stop(self, stop: PhysicalRentingStation) -> PhysicalRentingStation:
        return super(PhysicalRentingStations, self).add_stop(stop)

    """ Override abstract methods """

    def init_download(self):
        """ Downloads static datas """

        stations: List[Dict] = SharedMobilityFeed.open_json(self.station_info_url)['data']['stations']
        for station in stations:
            new_station: Station = Station(id=station['name'], name=station['name'])
            new_: PhysicalRentingStation = PhysicalRentingStation(
                id=station['station_id'],
                name=station['name'],
                station=new_station,
                platform_code=-1,
                index=None,
                geo=Coordinates(station['lat'], station['lon']),
                system_id=self.system_id,
                transport_types=self.system_transport_types,
                capacity=station['capacity']
            )
            new_station.add_stop(new_)
            self.add_stop(new_)

    def update(self):
        """ Update datas using real-time feeds  """
        status: List[Dict] = SharedMobilityFeed.open_json(self.station_status_url)['data']['stations']
        for state in status:
            station: PhysicalRentingStation = self.get_stop(state['station_id'])
            station.is_installed = state['is_installed']
            station.is_renting = state['is_renting']
            station.is_returning = state['is_returning']
            station.docks_available = state['num_docks_available']

            # This is specific for bikemi gbfs, because it uses bikes.
            # For other gbfs feeds, this code needs to be extended to handle other vehicles
            v_name = 'bike'
            station.vehicles_available = state[f'num_{v_name}s_available']


@attr.s(cmp=False, repr=False)
class GeofenceArea(RentingStation):

    @property
    def valid_source(self) -> bool:
        """ Returns true if the renting station is able to rent a vehicle, false otherwise """
        return False

    @property
    def valid_destination(self) -> bool:
        """ Returns true if the renting station is able to accept a returning vehicle, false otherwise """
        return False


@attr.s(cmp=False, repr=False)
class GeofenceAreas(RentingStations):
    # New dictionaries types
    set_idx: Dict[str, GeofenceArea] = dict()
    set_index: Dict[int, GeofenceArea] = dict()
    last_index: int = 1

    geofencing_zones_url: str = attr.ib(default=None)
    free_bike_status_url: str = attr.ib(default=None)

    """ Override superclass methods with stub, subsuming to RentingStations """

    def get_stop(self, stop_id) -> GeofenceArea:
        return self.set_idx[stop_id]

    def get_by_index(self, stop_index) -> GeofenceArea:
        """Get stop by index"""
        return self.set_index[stop_index]

    def add_stop(self, stop: GeofenceArea) -> GeofenceArea:
        return super(GeofenceAreas, self).add_stop(stop)

    """ Override abstract methods """

    def init_download(self):
        """ Downloads static datas """
        pass

    def update(self):
        """ Update datas using real-time feeds  """
        pass


@attr.s
class VehicleTransfer(Transfer):
    """
    This class represents a generic Transfer between two
    """

    @staticmethod
    def get_vehicle_transfer(
            sa: RentingStation,
            sb: RentingStation,
            transport_type: TransportType,
            speed: float | None = None
    ) -> Tuple[VehicleTransfer, VehicleTransfer]:
        """
        Given two renting stations, compute both inbound and outbound transfers.
        Transfer time is approximated dividing computed distance by transport type constant speed.
        """

        dist: float = Stop.stop_distance(sa, sb)

        if speed is None:
            if transport_type not in TRANSPORT_TYPE_SPEEDS.keys():
                raise ValueError(f"Unhandled transport type `{transport_type}`: average speed is not defined")

            speed: float = TRANSPORT_TYPE_SPEEDS[transport_type]

        time: int = int(dist * 3600 / speed)

        return (
            VehicleTransfer(from_stop=sa, to_stop=sb, transfer_time=time, transport_type=transport_type),
            VehicleTransfer(from_stop=sb, to_stop=sa, transfer_time=time, transport_type=transport_type)
        )


class VehicleTransfers(Transfers):
    """ This class represent a set of VehicleTransfers  """

    """ Override superclass methods with stub, subsuming to VehicleTransfer """

    def add(self, transfer: VehicleTransfer):
        super(VehicleTransfers, self).add(transfer)

    def with_from_stop(self, from_: RentingStation) -> List[VehicleTransfer]:
        """ Returns all transfers with given departing stop  """

        x = super(VehicleTransfers, self).with_from_stop(from_)
        return x

    def with_to_stop(self, to: Stop) -> List[VehicleTransfer]:
        """ Returns all transfers with given arrival stop  """

        return super(VehicleTransfers, self).with_to_stop(to)


class SharedMobilityFeed:
    """
    This class represent a GBFS feed
    All datas comes from gbfs.json (see https://github.com/NABSA/gbfs/blob/v2.3/gbfs.md#gbfsjson)
    """

    def __init__(self, url: str, lang: str = 'it'):
        self.url: str = url  # gbfs.json url
        self.lang: str = lang  # lang of feed
        self.feeds_url: Mapping[str, str] = self._get_feeds_url()  # mapping between feed_name and url
        self.system_id: str = self._get_items_list(feed_name='system_information')['system_id']  # feed system_id
        self.transport_types: List[TransportType] = self._get_transport_types()
        self.renting_stations: RentingStations = self._get_station()

    def __str__(self) -> str:
        return f"{self.system_id}: {len(self.renting_stations)}"

    @property
    def feeds(self):
        """ Name of feeds """
        return list(self.feeds_url.keys())

    @staticmethod
    def open_json(url: str) -> Dict:
        """ Reads json from url """
        return json.loads(urlopen(url=url).read())

    def _get_feeds_url(self) -> Mapping[str, str]:
        """ Returns dictionary keyed by feed name and mapped to associated feed url"""
        info: Dict = SharedMobilityFeed.open_json(url=self.url)
        feeds: List[Dict] = info['data'][self.lang]['feeds']  # list of feed items
        feed_url: Dict[str, str] = {feed['name']: feed['url'] for feed in feeds}

        return feed_url

    def _get_items_list(self, feed_name: str):
        """ Returns items list of given feed """
        if feed_name not in self.feeds:
            raise Exception(f"{feed_name} not in {self.feeds}")

        feed = SharedMobilityFeed.open_json(url=self.feeds_url[feed_name])
        datas = feed['data']

        if feed_name != 'system_information':
            items_name = next(
                iter(datas.keys()))  # name of items is only key in datas (e.g. 'stations', 'vehicles', ...)
            return datas[items_name]
        else:
            return datas  # in system_information datas is an items list

    def _get_transport_types(self) -> List[TransportType]:
        """
        Retrieves vehicle type from associated feeds
        """

        form_propulsion = set([(v_type['form_factor'], v_type['propulsion_type'])
                               for v_type in self._get_items_list(feed_name='vehicle_types')])
        types: List[TransportType] = []
        for form, propulsion in form_propulsion:
            if form == 'bicycle':
                if propulsion == 'human':
                    value = 9002  # Bike
                elif propulsion == 'electric_assist':
                    value = 9003  # Electric Bike
                else:
                    raise ValueError(f"No TransportType for vehicle{form} with {propulsion} propulsion")
            elif form == 'car':
                value = 9001  # Car
            else:
                raise ValueError(f"No TransportType for vehicle{form} with {propulsion} propulsion")
            types.append(TransportType(value))
        return types

    def _get_station(self) -> RentingStations:
        """ Basing on available feeds distinguish from PhysicalRentingStation or GeofenceArea"""
        if 'station_information' not in self.feeds or \
                'station_status' not in self.feeds or \
                len(self._get_items_list(feed_name='station_information')) == 0 or \
                len(self._get_items_list(feed_name='station_status')) == 0:

            if 'geofencing_zones' in self.feeds and 'free_bike_status_url' in self.feeds:
                stations: GeofenceAreas = GeofenceAreas(
                    system_id=self.system_id,
                    system_transport_types=self.transport_types,
                    geofencing_zones_url=self.feeds_url['geofencing_zones'],
                    free_bike_status_url=self.feeds_url['free_bike_status_url']
                )
            else:
                raise Exception(f"No compatible stations with feeds {self.feeds}")
        else:
            stations: PhysicalRentingStations = PhysicalRentingStations(
                system_id=self.system_id,
                system_transport_types=self.transport_types,
                station_info_url=self.feeds_url['station_information'],
                station_status_url=self.feeds_url['station_status']
            )

        stations.init_download()
        stations.update()
        return stations


def public_transport_stop(for_stops: Stops) -> List[Stop]:
    """ Returns its public stops  """
    return filter_public_transport(stops=for_stops)


def shared_mobility_stops(for_stops: Stops) -> List[RentingStation]:
    """ Returns its shared mobility stops  """
    return filter_shared_mobility(stops=for_stops)


def filter_public_transport(stops: Iterable[Stop]) -> List[Stop]:
    """ Filter only Stop objects, not its subclasses  """
    return [s for s in stops if type(s) == Stop]


def filter_shared_mobility(stops: Iterable[Stop]) -> list[RentingStation]:
    """ Filter only subclasses of RentingStation  """
    return [s for s in stops if isinstance(s, RentingStation)]
