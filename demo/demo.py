import argparse
import json
import os
import subprocess
import webbrowser
from os import path
from typing import List

from flask import Flask, render_template, redirect, url_for, request
from loguru import logger

from generate_station_names import NAMES_FILE

app = Flask(__name__)


QUERY_DIR: str = "./../pyraptor"
QUERY_RAPTOR: str = 'query_raptor.py'
QUERY_RAPTOR_DIR: str = 'raptor'
QUERY_RAPTOR_SHARED_MOB: str = 'query_raptor_sm.py'
QUERY_RAPTOR_SHARED_MOB_DIR: str = 'raptor_sm'

VISUALIZER_PATH = './../pyraptor/visualization/folium_visualizer.py'
ALGO_OUTPUT_NAME = 'algo-output.pcl'

IN: str = ''
FEED: str = ''
NAMES: List[str] = []

@app.route("/")
def root():
    return redirect(url_for('home'))


@app.route("/home")
def home():
    return render_template('home.html')


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
        file = path.join(QUERY_DIR, QUERY_RAPTOR)
        out = path.join(IN, QUERY_RAPTOR_DIR)
        flags = f"-i {IN} -or \"{origin}\" -d \"{destination}\" -t {time} -o {out}"
        cmd = f"python {file} {flags}"
        subprocess.run(cmd, shell=True)
        # visualization
        algo_file = path.join(out, ALGO_OUTPUT_NAME)
        flags = f"-a {algo_file} -o {out} -b True"
        cmd = f"python {VISUALIZER_PATH} {flags}"
        subprocess.run(cmd, shell=True)
        return redirect(url_for('home'))


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
        car = request.form.get("car")
        # query command line
        file = path.join(QUERY_DIR, QUERY_RAPTOR_SHARED_MOB)
        out = path.join(IN, QUERY_RAPTOR_SHARED_MOB_DIR)
        flags = f"-i {IN} -f {FEED} -or \"{origin}\" -d \"{destination}\" -t \"{time}\" -p \"{preferred}\" -c {car} -o {out}"
        cmd = f"python {file} {flags}"
        subprocess.run(cmd, shell=True)
        # visualization
        algo_file = path.join(IN, QUERY_RAPTOR_SHARED_MOB_DIR, ALGO_OUTPUT_NAME)
        flags = f"-a {algo_file} -o {out} -b True"
        cmd = f"python {VISUALIZER_PATH} {flags}"
        subprocess.run(cmd, shell=True)
        return redirect(url_for('home'))


def open_browser():
    webbrowser.open('http://127.0.0.1:5000/')


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/output",
        help="Input directory containing timetable.pcl (and names.json)",
    )
    parser.add_argument(
        "-f",
        "--feed",
        type=str,
        default="data/input/gbfs.json",
        help="Path to .json key specifying list of feeds and langs"
    )

    arguments = parser.parse_args()
    return arguments


def main(folder: str, feed: str):
    logger.debug("Input folder     : {}", folder)

    global IN
    IN = folder

    global FEED
    FEED = feed

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
    main(folder=args.input, feed=args.feed)
