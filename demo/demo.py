from __future__ import annotations

import argparse
import json
import os.path
import webbrowser
from os import path
from typing import List

import flask
from flask import Flask, render_template, redirect, url_for, request
from loguru import logger

from pyraptor.dao.timetable import read_timetable
from pyraptor.model.criteria import (
    CriteriaProvider,
    ArrivalTimeCriterion,
    TransfersCriterion,
    DistanceCriterion,
    EmissionsCriterion
)
from pyraptor.model.timetable import RaptorTimetable
from pyraptor.query import query_raptor, RaptorVariants
from pyraptor.visualization.folium_visualizer import visualize_output
from pyraptor.model.output import AlgorithmOutput

app = Flask(__name__)


class Option:
    """ This class is helpful to iterate on both
        class field in Jinja for-loop """
    def __init__(self, id_: str, name: str):
        self.id_: str = id_
        self.name: str = name


DEMO_OUTPUT_DIR = './../data/output/demo'
BASIC_RAPTOR_OUT_DIR = os.path.join(DEMO_OUTPUT_DIR, "basic")
MC_RAPTOR_OUT_DIR = os.path.join(DEMO_OUTPUT_DIR, "mc_raptor")

ALGO_OUTPUT_FILENAME = 'algo-output.pcl'
MC_CONFIG_FILENAME = 'mc_demo_config.json'
MC_CONFIG_FILEPATH = os.path.join(MC_RAPTOR_OUT_DIR, MC_CONFIG_FILENAME)

INPUT_FOLDER: str = "./../data/output"
FEED_CONFIG_PATH: str = "./../data/input/gbfs.json"
STATION_NAMES: List[str] = []
TIMETABLE: RaptorTimetable | None = None
DEBUG: bool = True
ENABLE_SM: bool = False
RAPTOR_ROUNDS = 5

VEHICLES = [Option(id_=id_, name=name) for id_, name in [
    ('regular', 'Regular bike'), ('electric', 'Electric bike'), ('car', 'Car')
]]


@app.route("/")
def root():
    return redirect(url_for('home'))


@app.route("/home")
def home():
    return render_template('home.html')


""" BASIC RAPTOR """


@app.route("/basic_raptor")
def basic_raptor():
    return render_template('raptor_form.html', stop_names=STATION_NAMES, vehicles=VEHICLES,
                           version_name='Basic RAPTOR', action='basic_raptor_run')


@app.route("/basic_raptor_run", methods=["GET", "POST"])
def basic_raptor_run():
    if request.method == "POST":
        # Retrieve data from the form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        departure_time = request.form.get("time")
        enable_sm = request.form.get("enable_sm") == 'on'
        preferred_vehicle = request.form.get("preferred")
        enable_car = request.form.get("car") == 'on'

        query_raptor(
            timetable=TIMETABLE,
            output_folder=BASIC_RAPTOR_OUT_DIR,
            origin_station=origin,
            destination_station=destination,
            departure_time=departure_time,
            rounds=RAPTOR_ROUNDS,
            variant=RaptorVariants.Basic.value,
            enable_sm=enable_sm,
            sm_feeds_path=FEED_CONFIG_PATH,
            preferred_vehicle=preferred_vehicle,
            enable_car=enable_car
        )

        visualize(BASIC_RAPTOR_OUT_DIR)

        return show_journey_descriptions(algo_output_dir=BASIC_RAPTOR_OUT_DIR)


""" WEIGHTED MULTICRITERIA RAPTOR """


@app.route("/wmc_raptor")
def wmc_raptor():
    return render_template('raptor_form.html', stop_names=STATION_NAMES, vehicles=VEHICLES,
                           version_name='Weighted McRAPTOR', action='wmc_raptor_run')


@app.route("/wmc_raptor_weights")
def wmc_raptor_weights():
    criteria_provider = CriteriaProvider(criteria_config_path=MC_CONFIG_FILEPATH)

    try:
        criteria = criteria_provider.get_criteria()
    except FileNotFoundError:
        criteria = [
            ArrivalTimeCriterion(name="arrival_time", weight=1, upper_bound=86400, raw_value=0),
            TransfersCriterion(name="transfers", weight=1, upper_bound=10, raw_value=0),
            DistanceCriterion(name="distance", weight=1, upper_bound=50, raw_value=0),
            ArrivalTimeCriterion(name="co2", weight=1, upper_bound=3000, raw_value=0)
        ]

    criteria_by_name = {c.name: c for c in criteria}

    return render_template('wmc_raptor_weights.html',
                           arrival_time=criteria_by_name["arrival_time"],
                           transfers=criteria_by_name["transfers"],
                           distance=criteria_by_name["distance"],
                           co2=criteria_by_name["co2"])


@app.route("/wmc_raptor_weights_save", methods=["GET", "POST"])
def wmc_raptor_weights_save():
    if request.method == "POST":
        # form
        form = request.form
        weights = {
            criteria: {
                "weight": float(form.get(f"{criteria}-weight")),
                "max": float(form.get(f"{criteria}-max"))
            }
            for criteria in ['distance', 'arrival_time', 'transfers', 'co2']
        }

        logger.debug(MC_RAPTOR_OUT_DIR)
        if not os.path.exists(MC_RAPTOR_OUT_DIR):
            os.makedirs(MC_RAPTOR_OUT_DIR)

        with open(MC_CONFIG_FILEPATH, 'w') as f:
            json.dump(weights, f)

        return redirect(url_for('wmc_raptor'))


@app.route("/wmc_raptor_run", methods=["GET", "POST"])
def wmc_raptor_run():
    if request.method == "POST":
        # form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        departure_time = request.form.get("time")
        enable_sm = request.form.get("enable_sm") == 'on'
        preferred_vehicle = request.form.get("preferred")
        enable_car = request.form.get("car") == 'on'

        query_raptor(
            timetable=TIMETABLE,
            output_folder=MC_RAPTOR_OUT_DIR,
            origin_station=origin,
            destination_station=destination,
            departure_time=departure_time,
            rounds=RAPTOR_ROUNDS,
            variant=RaptorVariants.WeightedMc.value,
            criteria_config=MC_CONFIG_FILEPATH,
            enable_sm=enable_sm,
            sm_feeds_path=FEED_CONFIG_PATH,
            preferred_vehicle=preferred_vehicle,
            enable_car=enable_car
        )

        visualize(MC_RAPTOR_OUT_DIR)
        return show_journey_descriptions(MC_RAPTOR_OUT_DIR)


"""Visualization utils"""


def visualize(algo_output_dir: str, open_browser: bool = True):
    algo_out_path = path.join(algo_output_dir, ALGO_OUTPUT_FILENAME)
    visualize_output(
        algo_output_path=algo_out_path,
        visualization_dir=algo_output_dir,
        open_browser=open_browser
    )


def show_journey_descriptions(algo_output_dir: str) -> flask.templating:
    algo_file: str = path.join(algo_output_dir, ALGO_OUTPUT_FILENAME)
    algo_output = AlgorithmOutput.read_from_file(filepath=algo_file)

    descriptions: List[List[str]] = []
    for jrny in algo_output.journeys:
        desc: str = str(jrny)
        desc += "\n\n\n--------------------------------------------------------------\n\n\n"

        descriptions.append(desc.split("\n"))

    return render_template("journey_desc.html", descs=descriptions)


"""Running the demo"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=INPUT_FOLDER,
        help="Input directory containing timetable.pcl (and names.json)",
    )
    parser.add_argument(
        "-f",
        "--feed",
        type=str,
        default=FEED_CONFIG_PATH,
        help="Path to .json key specifying list of feeds and langs"
    )
    parser.add_argument(
        "-sm",
        "--enable_sm",
        type=bool,
        default=ENABLE_SM,
        action=argparse.BooleanOptionalAction,
        help="Enable use of shared mobility data"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=DEBUG,
        help="Debug mode"
    )

    arguments = parser.parse_args()
    return arguments


def run_demo(input_folder: str, sm_feed_config_path: str, enable_sm: bool, debug: bool):
    logger.debug("Input folder            : {}", input_folder)
    logger.debug("Input feed config path  : {}", sm_feed_config_path)
    logger.debug("Enable shared mob       : {}", enable_sm)
    logger.debug("Debug mode              : {}", debug)

    global INPUT_FOLDER
    INPUT_FOLDER = input_folder

    global FEED_CONFIG_PATH
    FEED_CONFIG_PATH = sm_feed_config_path

    global DEBUG
    DEBUG = debug

    global ENABLE_SM
    ENABLE_SM = enable_sm

    global TIMETABLE
    TIMETABLE = read_timetable(input_folder=input_folder)

    global STATION_NAMES
    STATION_NAMES = _get_station_names(TIMETABLE)

    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=DEBUG, use_reloader=False)


def _get_station_names(timetable: RaptorTimetable):
    names = [st.name.strip() for st in timetable.stations]
    names = sorted(names, key=lambda s: s.lower())

    return names


# TODO bugs/stuff to investigate further:
#   - query from ABBADIA LARIANA to ANZANO DEL PARCO @16:33 generates KeyError: ANZANO DEL PARCO not found
#       -> it seems that with Trenord station names as destination, errors occur.
#       POSSIBLE SOLUTION: this is because Trenord actually does not have available trips,
#           so raptor can't find anything. This edge case, however, is not handled by the demo
#   - add enable_sm checkbox (and remove enable_sm from script args)
#       and disable departure time input if checkbox is ticked: shared mob data is realtime only,
#       so the departure time can only be the current one. Maybe implement check in BaseSharedMobRaptor
#       that ignores dep_secs if enable_sm==True and logs a warning


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    run_demo(
        input_folder=args.input,
        sm_feed_config_path=args.feed,
        enable_sm=args.enable_sm,
        debug=args.debug
    )
