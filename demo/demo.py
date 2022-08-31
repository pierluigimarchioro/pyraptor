from __future__ import annotations

import argparse
import json
import os.path
import re
import webbrowser
from os import path
from typing import List

import flask
from flask import Flask, render_template, redirect, url_for, request
from loguru import logger

from pyraptor.model.shared_mobility import RaptorTimetableSM
from pyraptor.timetable.io import read_timetable
from pyraptor.timetable.timetable import SHARED_MOB_TIMETABLE_FILENAME, TIMETABLE_FILENAME
from pyraptor.model.criteria import (
    CriteriaProvider,
    ArrivalTimeCriterion,
    TransfersCriterion,
    DistanceCriterion
)
from pyraptor.model.timetable import RaptorTimetable
from pyraptor.query import query_raptor, RaptorVariants
from pyraptor.util import sec2minutes
from pyraptor.visualization.folium_visualizer import visualize_output
from pyraptor.model.output import AlgorithmOutput

app = Flask(__name__)


class Option:
    """ This class is helpful to iterate on both
        class field in Jinja for-loop """

    def __init__(self, id_: str, name: str):
        self.id_: str = id_
        self.name: str = name


class LegDescriptor:

    def __init__(self, hour_start: str, stop_start: str,
                 hour_end: str, stop_end: str, transport: str,
                 line: str, trip_change: int):
        self.hour_start: str = hour_start
        self.stop_start: str = stop_start
        self.hour_end: str = hour_end
        self.stop_end: str = stop_end
        self.transport: str = transport
        self.line: str = line
        self.trip_change: int = trip_change


class JourneyDescriptor:

    def __init__(self):
        self.legs_descriptor: List[LegDescriptor] = []
        self.arrival_time = ''
        self.duration = ''
        self.distance = ''
        self.transfers = ''
        self.emissions = ''

    def add(self, leg_descriptor: LegDescriptor):
        self.legs_descriptor.append(leg_descriptor)


def _handle_journey_description(desc: str) -> JourneyDescriptor:
    line_pattern: str = "(.+?) (.+?) TO (.+?) (.+?) WITH Transport: (.+?) Route Name: (.+?$)"
    trip_change_line: str = "-- Trip Change"
    arrival_time_line: str = "Arrival Time: "
    duration_line: str = "Duration: "
    distance_line: str = "Travelled Distance: "
    transfer_line: str = "Total Transfers: "
    emissions_line: str = "Total Emissions: "

    journey_descriptor: JourneyDescriptor = JourneyDescriptor()

    # Split line-per-line
    descs: List[str] = desc.split("\n")

    # Remove double spaces and separators
    descs = [" ".join(d.replace('|', '').split()) for d in descs]
    # Remove empty line and debug log
    descs = [d for d in descs if
             not d.startswith('[Leg]') and not d.startswith('Journey')
             and d != '' and d != ' ']  # TODO remove first check when remove debug

    # Add legs
    n_trip = 1
    for d in descs:
        if d.startswith(trip_change_line):
            n_trip += 1
        elif d.startswith(arrival_time_line):
            journey_descriptor.arrival_time = (d.split(arrival_time_line)[1].strip())
        elif d.startswith(duration_line):
            journey_descriptor.duration = (d.split(duration_line)[1].strip())
        elif d.startswith(distance_line):
            journey_descriptor.distance = (d.split(distance_line)[1].strip())
        elif d.startswith(transfer_line):
            journey_descriptor.transfers = (d.split(transfer_line)[1].strip())
        elif d.startswith(emissions_line):
            journey_descriptor.emissions = (d.split(emissions_line)[1].strip())
        else:
            m = re.search(line_pattern, d)
            journey_descriptor.add(
                LegDescriptor(
                    hour_start=m.group(1),
                    stop_start=m.group(2),
                    hour_end=m.group(3),
                    stop_end=m.group(4),
                    transport=m.group(5),
                    line=m.group(6),
                    trip_change=n_trip
                )
            )

    return journey_descriptor


DEMO_OUTPUT_DIR = './../data/output/demo'
BASIC_RAPTOR_OUT_DIR = os.path.join(DEMO_OUTPUT_DIR, "basic")
MC_RAPTOR_OUT_DIR = os.path.join(DEMO_OUTPUT_DIR, "mc_raptor")

ALGO_OUTPUT_FILENAME = 'algo-output.pcl'
MC_CONFIG_FILENAME = 'mc_demo_config.json'
MC_CONFIG_FILEPATH = os.path.join(MC_RAPTOR_OUT_DIR, MC_CONFIG_FILENAME)

INPUT_FOLDER: str = "./../data/output"
STATION_NAMES: List[str] = []
STATION_NAMES_SM: List[str] = []
TIMETABLE: RaptorTimetable | None = None
TIMETABLE_SM: RaptorTimetableSM | None = None
DEBUG: bool = True
ENABLE_SM: bool = True
RAPTOR_ROUNDS = 10

VEHICLES = [Option(id_=id_, name=name) for id_, name in [
    ('regular', 'Regular bike'), ('electric', 'Electric bike'), ('car', 'Car')
]]


@app.route("/")
def root():
    return redirect(url_for('home'))


@app.route("/home")
def home():
    return render_template('home.html', enable_sm=ENABLE_SM)


@app.route("/switch_enable_sm")
def switch_enable_sm():
    global ENABLE_SM
    ENABLE_SM = not ENABLE_SM
    logger.debug(ENABLE_SM)
    return redirect(url_for('home'))


""" BASIC RAPTOR """


@app.route("/basic_raptor")
def basic_raptor():
    station_names = STATION_NAMES_SM if ENABLE_SM else STATION_NAMES
    return render_template('raptor_form.html', stop_names=station_names, vehicles=VEHICLES,
                           version_name='Basic RAPTOR', action='basic_raptor_run',
                           enable_sm=ENABLE_SM)


@app.route("/basic_raptor_run", methods=["GET", "POST"])
def basic_raptor_run():
    if request.method == "POST":
        # Retrieve data from the form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        departure_time = request.form.get("time")
        preferred_vehicle = request.form.get("preferred")
        enable_car = request.form.get("car") == 'on'

        timetable = TIMETABLE_SM if ENABLE_SM else TIMETABLE

        elapsed_time = query_raptor(
            timetable=timetable,
            output_folder=BASIC_RAPTOR_OUT_DIR,
            origin_station=origin,
            destination_station=destination,
            departure_time=departure_time,
            rounds=RAPTOR_ROUNDS,
            variant=RaptorVariants.Basic.value,
            enable_sm=ENABLE_SM,
            preferred_vehicle=preferred_vehicle,
            enable_car=enable_car
        )

        visualize(BASIC_RAPTOR_OUT_DIR)

        return show_journey_descriptions(algo_output_dir=BASIC_RAPTOR_OUT_DIR, time=elapsed_time)


""" WEIGHTED MULTICRITERIA RAPTOR """


@app.route("/wmc_raptor")
def wmc_raptor():
    station_names = STATION_NAMES_SM if ENABLE_SM else STATION_NAMES
    return render_template('raptor_form.html', stop_names=station_names, vehicles=VEHICLES,
                           version_name='Weighted McRAPTOR', action='wmc_raptor_run',
                           enable_sm=ENABLE_SM)


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
        preferred_vehicle = request.form.get("preferred")
        enable_car = request.form.get("car") == 'on'

        timetable = TIMETABLE_SM if ENABLE_SM else TIMETABLE

        query_raptor(
            timetable=timetable,
            output_folder=MC_RAPTOR_OUT_DIR,
            origin_station=origin,
            destination_station=destination,
            departure_time=departure_time,
            rounds=RAPTOR_ROUNDS,
            variant=RaptorVariants.WeightedMc.value,
            criteria_config=MC_CONFIG_FILEPATH,
            enable_sm=ENABLE_SM,
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


def show_journey_descriptions(algo_output_dir: str, time: float) -> flask.templating:
    algo_file: str = path.join(algo_output_dir, ALGO_OUTPUT_FILENAME)
    algo_output = AlgorithmOutput.read_from_file(filepath=algo_file)

    descriptions: List[JourneyDescriptor] = []

    for jrny in algo_output.journeys:
        descriptions.append(_handle_journey_description(str(jrny)))

    elapsed_time: str = f"{time} sec ({sec2minutes(time)})"

    return render_template("journey_desc.html", descs=descriptions, time=elapsed_time)


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
        "-d",
        "--debug",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=DEBUG,
        help="Debug mode"
    )

    arguments = parser.parse_args()
    return arguments


def run_demo(input_folder: str, debug: bool):
    logger.debug("Input folder            : {}", input_folder)
    logger.debug("Debug mode              : {}", debug)

    # input to global variables
    global INPUT_FOLDER
    INPUT_FOLDER = input_folder

    global DEBUG
    DEBUG = debug

    # loading timetables

    global TIMETABLE
    TIMETABLE = read_timetable(input_folder=input_folder, timetable_name=TIMETABLE_FILENAME)

    global STATION_NAMES
    STATION_NAMES = _get_station_names(TIMETABLE)

    global TIMETABLE_SM
    TIMETABLE_SM = read_timetable(input_folder=input_folder, timetable_name=SHARED_MOB_TIMETABLE_FILENAME)

    global STATION_NAMES_SM
    STATION_NAMES_SM = _get_station_names(TIMETABLE_SM)

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


if __name__ == "__main__":

    args = parse_arguments()
    print(args)
    run_demo(
        input_folder=args.input,
        debug=args.debug
    )
