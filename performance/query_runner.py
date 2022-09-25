"""
This script provides an interface to test RAPTOR performances
It computes journeys with different:
    - origin and destination stops and time departure
    - raptor settings
"""
from __future__ import annotations

import argparse
import json
import os.path
import random as rnd
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Dict, List, Type, Tuple
from zipfile import ZipFile

import pandas as pd
import requests
from loguru import logger

from pyraptor.model.criteria import (
    Criterion,
    ArrivalTimeCriterion,
    TransfersCriterion,
    DistanceCriterion,
    EmissionsCriterion,
    CriterionConfiguration,
    CriteriaProvider
)
from pyraptor.query import query_raptor
from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.model.timetable import RaptorTimetable, Stop
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import generate_timetable, TIMETABLE_FILENAME, SHARED_MOB_TIMETABLE_FILENAME
from pyraptor.util import mkdir_if_not_exists

OUT_FILENAME = "runner_out.csv"  # output file with performance info


# TODO define some classes for the configuration objects (e.g. timetable config, raptor config)
#   tp facilitate dependency injection


@dataclass
class Query:
    """
    Class that represents a query, with info about origin stop, destination stop
    and departure time.
    """

    origin: str
    destination: str
    dep_time: str

    def __str__(self) -> str:
        return f"[FROM {self.origin} TO {self.destination} AT {self.dep_time}]"


def main(input_: str, output_dir: str, config: str):
    logger.debug("Timetable directory         : {}", input_)
    logger.debug("Output directory            : {}", output_dir)
    logger.debug("Configuration file          : {}", config)

    mkdir_if_not_exists(output_dir)

    logger.debug("Loading runner configuration...")
    runner_config: Dict = _json_to_dict(config)

    max_rounds: int = runner_config["max_rounds"]

    logger.info("Running timetable configurations")
    for tt_config in runner_config["timetable_configs"]:
        run_timetable_configuration(
            timetables_dir=input_,
            output_dir=output_dir,
            timetable_config=tt_config,
            max_rounds=max_rounds
        )


def _json_to_dict(file: str) -> Dict:
    """
    Convert a json to a dictionary
    :param file: path to json file
    :return: data as a dictionary
    """

    return json.load(open(file))


def run_timetable_configuration(timetables_dir: str, output_dir: str, timetable_config: Mapping, max_rounds: int):
    """
    Executes the queries and the RAPTOR variants provided timetable configuration.
    For each timetable configuration, one or two timetables are generated if the use of
    cached data is not enabled or if no previously generated corresponding timetable file is found.
    For each timetable configuration, one .csv output file is generated at the provided
    output directory.

    :param timetables_dir: directory to store the timetables in
    :param output_dir: directory to store the runner results in
    :param timetable_config: timetable configuration obj
    :param max_rounds: maximum number of rounds to execute any RAPTOR variant for.
        If -1, RAPTOR is always executed until convergence.
    """

    timetable_id = timetable_config["timetable_id"]
    timetable, timetable_sm = _generate_timetables(
        timetables_dir=timetables_dir,
        timetable_config=timetable_config
    )

    # Queries and RAPTOR settings
    run_config = timetable_config["run_config"]
    queries_settings: Mapping = run_config["query_settings"]
    raptor_configs: Sequence[Mapping] = run_config["raptor_configs"]

    # Get the queries to execute
    # Generate for both timetables, since shared-mob info is not included
    #   in non-sm timetable
    queries: Sequence[Query] = generate_queries(query_settings=queries_settings, timetable=timetable)

    queries_sm: Sequence[Query] | None = None
    if timetable_sm is not None:
        queries_sm = generate_queries(query_settings=queries_settings, timetable=timetable_sm)

    # Records with the following fields:
    # query,query_time,generalized_cost,config_name,dataset,fwd_deps
    runner_results: List[Mapping] = []

    logger.info("Executing RAPTOR configurations...")
    for raptor_config_obj in raptor_configs:
        sm_enabled = raptor_config_obj["enable_sm"]

        if sm_enabled and queries_sm is None:
            raise ValueError("Shared-mobility option enabled in RAPTOR config, "
                             "but no shared-mobility data provided")

        queries_to_execute = queries_sm if sm_enabled else queries
        timetable = timetable_sm if sm_enabled else timetable

        for q in queries_to_execute:
            query_time, journey_cost = run_raptor_config(
                raptor_config=raptor_config_obj,
                timetable=timetable,
                query=q,
                max_rounds=max_rounds
            )

            result = {
                "query": str(q),
                "query_time": query_time,
                "generalized_cost": journey_cost,
                "fwd_deps_enabled": raptor_config_obj["fwd_deps_heuristic"],
                "config_name": raptor_config_obj["configuration_id"],
                "dataset": timetable_id
            }
            runner_results.append(result)

    # TODO timetable_id is not sanitized
    output_file = os.path.join(output_dir, f"{timetable_id}_{OUT_FILENAME}")
    logger.info(f"Runner execution terminated. Saving output to {output_file}")

    results_df = pd.DataFrame.from_records(data=runner_results)
    results_df.to_csv(output_file, index=False)


def _generate_timetables(timetables_dir: str, timetable_config: Mapping) -> Tuple[RaptorTimetable, RaptorTimetableSM | None]:
    """
    Returns a pair of timetables generated with the provided configuration object.
    There is always a "normal" timetable, while the shared-mob timetable might be None,
    depending on the provided config object.

    :param timetables_dir: directory to store the timetables in
    :param timetable_config: timetable configuration object
    :return: (timetable, share-mob timetable) pair, with the latter potentially being None
    """

    timetable_id = timetable_config["timetable_id"]
    current_tt_dir = os.path.join(timetables_dir, timetable_id)
    raw_gtfs_dir = os.path.join(current_tt_dir, "gtfs")

    mkdir_if_not_exists(current_tt_dir)
    mkdir_if_not_exists(raw_gtfs_dir)

    # Data is downloaded/generated only if the use of cached data is enabled
    # and there actually is some previously downloaded/generated data
    use_cached: bool = timetable_config["use_cached"]
    timetable_path = os.path.join(current_tt_dir, TIMETABLE_FILENAME)
    timetable_sm_path = os.path.join(current_tt_dir, SHARED_MOB_TIMETABLE_FILENAME)

    # If the gtfs is downloaded, the timetables must be generated even if cached data can be used
    refreshed_gtfs = False
    if not use_cached or len(os.listdir(raw_gtfs_dir)) == 0:
        gtfs_download_url = timetable_config["gtfs_download_url"]
        logger.info(f"[{timetable_id}] Downloading GTFS feed from url: {gtfs_download_url}")
        _download_gtfs(
            download_dir=raw_gtfs_dir,
            download_url=gtfs_download_url,
            remove_zip=True,
            overwrite=True
        )

        refreshed_gtfs = True
    else:
        logger.info(f"[{timetable_id}] Using cached GTFS feed")

    print(os.path.abspath(timetable_path))
    if not use_cached or not os.path.exists(timetable_path) or refreshed_gtfs:
        logger.info(f"[{timetable_id}] Generating RAPTOR timetable...")
        # Generate non-sm timetable
        generate_timetable(
            input_folder=raw_gtfs_dir,
            output_folder=current_tt_dir,
            departure_date=timetable_config["date"],
            agencies=[],
        )
        logger.info(f"[{timetable_id}] RAPTOR timetable generated")
    else:
        logger.info(f"[{timetable_id}] Loading cached timetable")

    timetable: RaptorTimetable = read_timetable(input_folder=current_tt_dir, timetable_name=TIMETABLE_FILENAME)

    timetable_sm: RaptorTimetableSM | None = None
    if "gbfs" in timetable_config:
        if not use_cached or not os.path.exists(timetable_sm_path) or refreshed_gtfs:
            logger.info(f"[{timetable_id}] Generating Shared-Mobility RAPTOR timetable...")

            # Generate shared mob config file
            # TODO consider Dep. Inj. instead of having to write to a file
            sm_config_path = os.path.join(current_tt_dir, "sm_feeds_config.json")
            json.dump(timetable_config["gbfs"], open(sm_config_path, 'w'))

            # Generate non-sm timetable
            generate_timetable(
                input_folder=raw_gtfs_dir,
                output_folder=current_tt_dir,
                departure_date=timetable_config["date"],
                agencies=[],
                shared_mobility=True,
                feeds_path=sm_config_path
            )
            logger.info(f"[{timetable_id}] Shared-Mobility RAPTOR timetable generated")
        else:
            logger.info(f"[{timetable_id}] Loading cached shared-mob timetable")

        timetable_sm = read_timetable(input_folder=current_tt_dir, timetable_name=SHARED_MOB_TIMETABLE_FILENAME)

    return timetable, timetable_sm


def _download_gtfs(download_url: str,
                   download_dir: str,
                   remove_zip: bool = True,
                   overwrite: bool = False):
    """
    Downloads gtfs-format file from given url.
    file is then parsed in order be managed with pandas software library.
    Parsing phase consist of:
        - unzip downloaded file
        - save the content in the provided download directory

    :param download_url: url of the file to download.
    :param download_dir: path to save files at
    :param remove_zip: If true, the GTFS zip archive is removed upon extraction
    :param overwrite: if True, overwrites existing files with the same names
    :return:
    """

    # Check to not overwrite file
    # directory exists, it is not empty and overwrite flag is off
    if os.path.exists(download_dir) and len(os.listdir(download_dir)) > 0 and not overwrite:
        # preserve old file
        raise Exception(f"Cannot download: it could overwrite existing files")
    else:
        _clear_dir_content(dir_path=download_dir)

    # Downloading also creates the destination dir if it doesn't exist
    logger.debug("Downloading...")
    downloaded_file_path = _download_file(
        out_dir=download_dir,
        download_url=download_url
    )

    logger.debug("Extracting zip... ")
    _extract_from_zip(
        out_dir=download_dir,
        delete_zip=remove_zip,
        zip_file_path=downloaded_file_path
    )

    logger.debug("GTFS feed successfully prepared.")


def _download_file(download_url: str, out_dir: str) -> str:
    """
    Downloads the file with the provided url and saves it to the provided output directory.
    The filename is extracted from the provided url.
    Returns the path where the file has been saved.

    Reference:
    https://stackoverflow.com/questions/56950987/download-file-from-url-and-save-it-in-a-folder-python

    :param download_url: url of the file to download.
    :param out_dir: output directory path.
        If the directory doesn't exist, it is automatically created.
    :return: path of the file that has been saved.
    """

    mkdir_if_not_exists(out_dir)

    # Extract filename from url and replace whitespaces with "_"
    filename = download_url.split('/')[-1].replace(" ", "_")
    out_path = os.path.join(out_dir, filename)

    # HTTP GET request to download
    req = requests.get(download_url, stream=True)

    if req.ok:
        logger.debug(f"Saving file to {os.path.abspath(out_path)}...")

        with open(out_path, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

    else:  # HTTP status code 4XX/5XX
        logger.debug(f"Download failed: status code {req.status_code}\n{req.text}")

    logger.debug("---- Download terminated. ----\n")

    return out_path


def _extract_from_zip(zip_file_path: str, out_dir: str,
                      delete_zip: bool = True):
    """
    Unzips the provided .zip file and extracts its content in the provided
    output directory. The original .zip file is deleted.

    Reference:
    https://stackoverflow.com/questions/31346790/unzip-all-zipped-files-in-a-folder-to-that-same-folder-using-python-2-7-5

    :param zip_file_path: path of the file to unzip
    :param out_dir: path of the directory to extract the .zip into
    :param delete_zip: True to delete the zip after extraction, False otherwise.
        Defaults to True.
    """

    logger.debug(f'Extracting {zip_file_path}...')
    zip_ref = ZipFile(zip_file_path)
    zip_ref.extractall(out_dir)
    zip_ref.close()

    if delete_zip:
        os.remove(zip_file_path)
        logger.debug(f'Deleting {zip_file_path}...')

    logger.debug("---- Extraction terminated. ----\n")


def _clear_dir_content(dir_path: str) -> bool:
    """
    If dir exists clear his content
        it is useful when downloading to reset starting state
    :param dir_path: path of dir to clear content
    :return: true if content is clear, false if didn't exist
    """
    if os.path.exists(dir_path):
        logger.debug(f'Clear {dir_path} content...')
        for file in os.listdir(dir_path):
            item = os.path.join(dir_path, file)

            if os.path.isdir(item):
                # os.remove can't delete folders
                shutil.rmtree(item)
            else:
                os.remove(item)

        return True

    logger.debug(f"{dir_path} doesn't exist...")
    return False


def generate_queries(query_settings: Mapping, timetable: RaptorTimetable) -> Sequence[Query]:
    """
    Returns a sequence of queries based on the provided settings.

    :param query_settings: value of the "queries_settings" field of the runner configuration
    :param timetable: timetable to retrieve random stops from, if random queries are enabled
    :return: sequence of queries
    """

    queries: List[Query] = []

    # If True, generate random queries
    # if False, consider the queries in the field "queries"
    random_queries = query_settings["random"]
    if random_queries:
        logger.info(f"Generating random queries...")

        # Range of valid distances [Km]
        min_distance = query_settings["min_distance"]
        max_distance = query_settings["max_distance"]

        # Range of possible query hours
        min_hour = query_settings["min_hour"]
        max_hour = query_settings["max_hour"]

        # Total number of random queries to generate
        total_number = query_settings["number"]

        all_stops = timetable.stops
        for i in range(total_number):
            logger.debug(f"Generating random query #{i} of {total_number}")

            # Get two random stops whose distance from each other is in the valid range
            origin_stop: Stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]
            dest_stop: Stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]

            while (origin_stop == dest_stop
                   or not (min_distance <= Stop.stop_distance(origin_stop, dest_stop) <= max_distance)):
                dest_stop = all_stops.set_index[rnd.randint(0, len(all_stops) - 1)]

            # Generate random query time in the provided range
            query_hour = rnd.randint(min_hour, max_hour)
            query_time = f"{query_hour}:00:00"

            query = Query(origin=origin_stop.name, destination=dest_stop.name, dep_time=query_time)
            queries.append(query)
            logger.debug(f"Query generated: {query}")
    else:
        logger.info(f"Using provided queries...")
        for q_obj in query_settings["queries"]:
            query = Query(origin=q_obj["from"], destination=q_obj["to"], dep_time=q_obj["at"])
            queries.append(query)

            logger.debug(f"Query generated: {query}")

    return queries


def run_raptor_config(
        raptor_config: Mapping,
        timetable: RaptorTimetable,
        query: Query,
        max_rounds: int
) -> Tuple[float, float]:
    """
    R
    :param raptor_config: RAPTOR run configuration, i.e. a "raptor_configs" object
        in the runner_config.json file
    :param timetable: RAPTOR timetable
    :param query: object with departure and arrival stop and arrival time
    :param max_rounds: number of rounds
    :return: query execution time and total generalized cost of the resulting journey
    """

    variant = raptor_config["variant"]
    enable_sm = raptor_config["enable_sm"]

    criteria_provider = None
    if variant == 'gc':
        criteria_config_raw: Mapping[str, Mapping[str, float]] = raptor_config["criteria_config"]

        # These are the names defined in the runner_config.json file
        criteria_classes: Mapping[str, Type[Criterion]] = {
            "arrival_time": ArrivalTimeCriterion,
            "transfers": TransfersCriterion,
            "distance": DistanceCriterion,
            "co2": EmissionsCriterion
        }

        criteria_config: Dict[Type[Criterion], CriterionConfiguration] = {}
        for c_name, c_params in criteria_config_raw.items():
            criteria_config[criteria_classes[c_name]] = CriterionConfiguration(
                weight=c_params["weight"],
                upper_bound=c_params["upper_bound"]
            )

        criteria_provider = CriteriaProvider(criteria_config=criteria_config)

    query_time, algo_output = query_raptor(
        variant=variant,
        timetable=timetable,
        origin_station=query.origin,
        destination_station=query.destination,
        departure_time=query.dep_time,
        rounds=max_rounds,
        enable_fwd_deps=raptor_config["fwd_deps_heuristic"],
        criteria_provider=criteria_provider,
        enable_sm=enable_sm,
        preferred_vehicle=raptor_config["sm"]["preferred_vehicle"] if enable_sm else None,
        enable_car=raptor_config["sm"]["enable_car"] if enable_sm else None
    )

    if (algo_output.journeys is None
            or len(algo_output.journeys) == 0):
        journey_cost = -1
    else:
        # TODO as of now, algorithms output just one journey.
        #  How to handle in case multiple ones are produced? Average?
        journey_cost = algo_output.journeys[0].total_cost()

    return query_time, journey_cost


def _parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/input/performance",
        help="Timetable directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output/performance",
        help="Output directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="performance/runner_config.json",
        help="Runner configuration file",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = _parse_arguments()

    main(
        args.input,
        args.output,
        args.config
    )
