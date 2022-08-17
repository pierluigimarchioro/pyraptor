from __future__ import annotations

import argparse
import json
import subprocess
import webbrowser
from os import path
from typing import List

import flask
from flask import Flask, render_template, redirect, url_for, request
from loguru import logger

from generate_station_names import NAMES_FILE
from pyraptor.model.output import AlgorithmOutput
from pyraptor.util import mkdir_if_not_exists

app = Flask(__name__)

QUERY_DIR: str = "./../pyraptor"
QUERY_BASE_RAPTOR: str = 'query_raptor.py'
QUERY_BASE_RAPTOR_DIR: str = 'raptor'
QUERY_RAPTOR_SHARED_MOB: str = 'query_raptor_sm.py'
QUERY_RAPTOR_SHARED_MOB_DIR: str = 'raptor_sm'
QUERY_MC_RAPTOR: str = 'query_mcraptor.py'
QUERY_MC_RAPTOR_DIR: str = 'mc_raptor'

VISUALIZER_PATH = './../pyraptor/visualization/folium_visualizer.py'
ALGO_OUTPUT_NAME = 'algo-output.pcl'
DEMO_OUTPUT = './../data/output/demo'
MC_CONFIG = 'mc_demo_config.json'

IN: str = "./../data/output"
FEED: str = "./../data/input/gbfs.json"
NAMES: List[str] = []

DEBUG: bool = True


def run_script(file_name: str, flags: str):
    cmd = f"python {file_name} {flags}"

    logger.debug(f"Running script with command `{cmd}`")

    if not DEBUG:
        proc = subprocess.run(cmd, text=True, shell=True, capture_output=True)
        if proc.returncode != 0:
            exc = proc.stderr.split('\n')[-2]
            raise Exception(exc)
    else:
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise Exception("Internal Error")


def visualize(dir_: str):
    algo_file = path.join(dir_, ALGO_OUTPUT_NAME)
    flags = f"-a {algo_file} -o {dir_} -b True"
    run_script(file_name=VISUALIZER_PATH, flags=flags)


def journey_desc(dir_: str) -> flask.templating:
    algo_file: str = path.join(dir_, ALGO_OUTPUT_NAME)
    desc: str = AlgorithmOutput.read_from_file(filepath=algo_file).journey.print()
    descs: List[str] = desc.split('\n')
    return render_template("journey_desc.html", descs=descs)


@app.route("/")
def root():
    return redirect(url_for('home'))


@app.route("/home")
def home():
    return render_template('home.html')


""" BASE RAPTOR """


@app.route("/base_raptor")
def base_raptor():
    return render_template('base_raptor.html', stop_names=NAMES)


@app.route("/base_raptor_run", methods=["GET", "POST"])
def base_raptor_run():
    if request.method == "POST":
        # form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        time = request.form.get("time")
        # query command line
        file = path.join(QUERY_DIR, QUERY_BASE_RAPTOR)
        out = path.join(DEMO_OUTPUT, QUERY_BASE_RAPTOR_DIR)
        flags = f"-i {IN} -or \"{origin}\" -d \"{destination}\" -t {time} -o {out}"
        run_script(file_name=file, flags=flags)
        visualize(out)
        return journey_desc(out)


""" SHARED MOBILITY RAPTOR """


@app.route("/shared_mob_raptor")
def shared_mob_raptor():
    return render_template('shared_mob_raptor.html', stop_names=NAMES, vehicles=['regular', 'electric', 'car'])


@app.route("/shared_mob_raptor_run", methods=["GET", "POST"])
def shared_mob_raptor_run():
    if request.method == "POST":
        # form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        time = request.form.get("time")
        preferred = request.form.get("preferred")
        car = request.form.get("car") == 'on'
        # query command line
        file = path.join(QUERY_DIR, QUERY_RAPTOR_SHARED_MOB)
        out = path.join(DEMO_OUTPUT, QUERY_RAPTOR_SHARED_MOB_DIR)
        flags = (f"-i {IN} -f {FEED} -or \"{origin}\" -d \"{destination}\" -t \"{time}\" -p \"{preferred}\" "
                 f"{'-c True' if car else ''} -o {out}")
        run_script(file_name=file, flags=flags)
        visualize(out)
        return journey_desc(out)


""" WEIGHTED MULTICRITERIA RAPTOR """


@app.route("/mc_raptor_weights")
def mc_raptor_weights():
    return render_template('mc_raptor_weights.html')


@app.route("/mc_raptor_weights_save", methods=["GET", "POST"])
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
        mc_dir = path.join(DEMO_OUTPUT, QUERY_MC_RAPTOR_DIR)
        mkdir_if_not_exists(mc_dir)
        with open(path.join(mc_dir, MC_CONFIG), 'w') as f:
            json.dump(weights, f)
        return redirect(url_for('mc_raptor'))


@app.route("/mc_raptor")
def mc_raptor():
    return render_template('mc_raptor.html', stop_names=NAMES)


@app.route("/mc_raptor_run", methods=["GET", "POST"])
def mc_raptor_run():
    if request.method == "POST":
        # form
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        time = request.form.get("time")
        # query command line
        file = path.join(QUERY_DIR, QUERY_MC_RAPTOR)
        output_dir = path.join(DEMO_OUTPUT, QUERY_MC_RAPTOR_DIR)
        mc_path = path.join(DEMO_OUTPUT, QUERY_MC_RAPTOR_DIR, MC_CONFIG)
        flags = f"-i {IN} -or \"{origin}\" -d \"{destination}\" -t {time} -o {output_dir} -wmc True -cfg {mc_path}"
        run_script(file_name=file, flags=flags)
        visualize(output_dir)
        return journey_desc(output_dir)


def open_browser():
    webbrowser.open('http://127.0.0.1:5000/')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=IN,
        help="Input directory containing timetable.pcl (and names.json)",
    )
    parser.add_argument(
        "-f",
        "--feed",
        type=str,
        default=FEED,
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


def main(folder: str, feed: str, debug: bool):
    logger.debug("Input folder     : {}", folder)
    logger.debug("Input feed       : {}", feed)
    logger.debug("Input debug      : {}", debug)

    global IN
    IN = folder

    global FEED
    FEED = feed

    global DEBUG
    DEBUG = debug

    names_file = path.join(folder, NAMES_FILE)
    try:
        global NAMES
        NAMES = json.load(open(names_file, 'r'))['names']
    except:
        raise Exception(f"No {names_file} in {folder}, please run `generate_station_names`")

    open_browser()
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(folder=args.input, feed=args.feed, debug=args.debug)
