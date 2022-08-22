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
from pyraptor.model.timetable import RaptorTimetable
from pyraptor.query import query_raptor, RaptorVariants
from pyraptor.visualization.folium_visualizer import visualize_output
from pyraptor.model.output import AlgorithmOutput

app = Flask(__name__)

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
RAPTOR_ROUNDS = 5


@app.route("/")
def root():
    return redirect(url_for('home'))


@app.route("/home")
def home():
    return render_template('home.html')


""" BASIC RAPTOR """


@app.route("/basic_raptor")
def basic_raptor():
    return render_template('basic_raptor.html', stop_names=STATION_NAMES, vehicles=['regular', 'electric', 'car'])


@app.route("/basic_raptor_run", methods=["GET", "POST"])
def shared_mob_raptor_run():
    if request.method == "POST":
        # Retrieve data from the form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        departure_time = request.form.get("time")
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
            enable_sm=True,
            sm_feeds_path=FEED_CONFIG_PATH,
            preferred_vehicle=preferred_vehicle,
            enable_car=enable_car
        )

        visualize(BASIC_RAPTOR_OUT_DIR)

        return show_journey_descriptions(algo_output_dir=BASIC_RAPTOR_OUT_DIR)


""" WEIGHTED MULTICRITERIA RAPTOR """


@app.route("/wmc_raptor")
def mc_raptor():
    return render_template('wmc_raptor.html', stop_names=STATION_NAMES)


@app.route("/wmc_raptor_weights")
def mc_raptor_weights():
    return render_template('wmc_raptor_weights.html')


@app.route("/wmc_raptor_weights_save", methods=["GET", "POST"])
def mc_raptor_weights_save():
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

        with open(MC_CONFIG_FILEPATH, 'w') as f:
            json.dump(weights, f)

        return redirect(url_for('mc_raptor'))


@app.route("/wmc_raptor_run", methods=["GET", "POST"])
def mc_raptor_run():
    if request.method == "POST":
        # form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        departure_time = request.form.get("time")

        query_raptor(
            timetable=TIMETABLE,
            output_folder=MC_RAPTOR_OUT_DIR,
            origin_station=origin,
            destination_station=destination,
            departure_time=departure_time,
            rounds=RAPTOR_ROUNDS,
            variant=RaptorVariants.WeightedMc.value,
            criteria_config=MC_CONFIG_FILEPATH
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

    descriptions: List[str] = []
    for jrny in algo_output.journeys:
        desc: str = jrny.print()  # TODO implement __str__ to return just the string: print() side-effects to console
        desc += "\n\n\n--------------------------------------------------------------\n\n\n"

        descriptions.append(desc)

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
        "-d",
        "--debug",
        type=bool,
        default=DEBUG,
        help="Debug mode"
    )

    arguments = parser.parse_args()
    return arguments


def run_demo(input_folder: str, sm_feed_config_path: str, debug: bool):
    logger.debug("Input folder            : {}", input_folder)
    logger.debug("Input feed config path  : {}", sm_feed_config_path)
    logger.debug("Debug mode              : {}", debug)

    global INPUT_FOLDER
    INPUT_FOLDER = input_folder

    global FEED_CONFIG_PATH
    FEED_CONFIG_PATH = sm_feed_config_path

    global DEBUG
    DEBUG = debug

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
#       -> it seems that with Trenord station names as destination, errors occur


if __name__ == "__main__":
    args = parse_arguments()
    run_demo(
        input_folder=args.input,
        sm_feed_config_path=args.feed,
        debug=args.debug
    )
