from __future__ import annotations

import os
from typing import Dict, List
from zipfile import ZipFile

import pandas as pd
from loguru import logger

from pyraptor.util import mkdir_if_not_exists


def read_gtfs_tables(gtfs_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Reads the GTFS tables from the provided directory.

    :param gtfs_dir: path to a directory containing GTFS tables
    :return: dictionary, keyed by table name, containing the GTFS tables
    """

    logger.info(f"Reading GTFS at {gtfs_dir}")

    if not os.path.exists(gtfs_dir):
        raise Exception(f"No directory {gtfs_dir}")

    gtfs_tables: Dict[str, pd.DataFrame] = {}
    for item in os.listdir(gtfs_dir):
        if os.path.isdir(os.path.join(gtfs_dir, item)):
            # skip directories
            logger.debug(f'Skipping directory: {item}')
            continue
        try:
            table_name, ext = item.split('.')  # split file-name and file_extension

            if ext == 'txt':
                gtfs_tables[table_name] = pd.read_csv(os.path.join(gtfs_dir, item))

                logger.debug(f'Reading {item}')
            else:
                logger.debug(f'Skipping {item}')
        except Exception as ex:
            logger.error(f"An error occurred while processing {item}. Error: {ex}")

    return gtfs_tables


def save_gtfs(gtfs_tables: Dict[str, pd.DataFrame],
              out_dir: str, gtfs_filename: str):
    """
    Stores the provided dataframe, which represent GTFS tables, as csv files in the provided directory.
    Also creates a GTFS archive from said files

    :param gtfs_tables: dictionary of pandas dataframe;
        keys are dataframe in name and values are relative pandas dataframe
    :param out_dir: path of dir to save gtfs files
    :param gtfs_filename: name of the gtfs file that will be created in the provided output directory
    """

    mkdir_if_not_exists(out_dir)

    for df_name, df in gtfs_tables.items():
        logger.debug(f'Saving {df_name}.txt')
        df.to_csv(os.path.join(out_dir, f"{df_name}.txt"), index=False)

    tables_to_gtfs(tables_dir=out_dir,
                   out_dir=out_dir, gtfs_filename=gtfs_filename)


def tables_to_gtfs(tables_dir: str, gtfs_filename: str, out_dir: str=None):
    """
    Given a path to a directory containing gtfs .txt tables, saves them into a .gtfs file.
    Such file is created inside the provided directory.

    :param tables_dir: path of directory where the .txt table files are saved
    :param gtfs_filename: name of gtfs file that will be created as output
    :param out_dir: directory to write the gtfs archive to.
        If None, then the provided table directory is used instead
    """

    # raises an exception if table directory doesn't exist
    if not os.path.exists(tables_dir):
        raise Exception(f"No dir '{tables_dir}'")

    gtfs_dir = tables_dir if out_dir is None else out_dir
    gtfs_path = os.path.join(gtfs_dir, gtfs_filename)

    # Delete previous .gtfs archive in order to overwrite it,
    # else there would be problems when zipping the folder
    try:
        logger.debug(f"Removing {gtfs_path}")
        os.remove(gtfs_path)
        logger.debug("Overwriting existing GTFS file")
    except FileNotFoundError:
        logger.debug("No previous GTFS file found")

    logger.debug(f"Zipping {tables_dir} in {gtfs_path}")
    create_zip(
        out_dir=gtfs_dir,
        zip_filename=gtfs_filename,
        to_zip=tables_dir
    )


def create_zip(out_dir: str, zip_filename: str,
               to_zip: str | List[str] | Dict[str, str]) -> str:
    """
    Creates a zip archive that contains the specified files

    :param out_dir: directory where the zip archive is written
    :param zip_filename: name of the zip archive
    :param to_zip: Depending on the type of the provided value:
        - str: path to a directory whose content needs to be zipped.
            The content of the directory is zipped and placed in the root folder of the zip archive.
            This mimics the behaviour of right-click + "compress" performed on a directory
            using the finder/resource explorer/whatever.
        - List[str]: collection of paths that point to the files to be zipped.
        - Dict[str, str]: a collection of arcnames (filenames inside the zip archive,
            more at https://docs.python.org/3/library/zipfile.html?highlight=arcname),
            keyed by filepath.
            This is useful because it avoids copying the whole directory structure of the original files
            (more at https://stackoverflow.com/questions/27991745/zip-file-and-avoid-directory-structure).
    :return:
    """

    logger.debug("Creating zip archive...")
    zip_path = os.path.join(out_dir, zip_filename)

    # See function docs for the 3 different cases based on the "to_zip" argument type
    if isinstance(to_zip, dict):
        _zip_arcnames_dict(zip_path=zip_path,
                           paths_with_arcnames=to_zip)
    elif isinstance(to_zip, list):
        _zip_path_collection(zip_path=zip_path,
                             paths_to_zip=to_zip)
    else:
        _zip_directory(zip_path=zip_path,
                       dir_path=to_zip)

    logger.debug("Zip archive created.")
    return zip_path


def _zip_arcnames_dict(zip_path: str, paths_with_arcnames: Dict[str, str]):
    with ZipFile(zip_path, "w") as zip_file:
        for filepath in paths_with_arcnames:
            arcname = paths_with_arcnames[filepath]

            logger.debug(f"Adding {filepath} to the archive as {arcname}...")
            zip_file.write(filepath, arcname=arcname)


def _zip_path_collection(zip_path: str, paths_to_zip: List[str]):
    with ZipFile(zip_path, "w") as zip_file:
        for filepath in paths_to_zip:
            logger.debug(f"Adding {filepath} to the archive...")
            zip_file.write(filepath)


def _zip_directory(zip_path: str, dir_path: str):
    # Create arcname dictionary
    # This is so the content of the provided directory is placed directly
    # in the root folder of the zip archive, without carrying any
    # parent directory structure.
    arcnames = {}
    for dir_entry in os.scandir(dir_path):
        arcnames[dir_entry.path] = dir_entry.name

    _zip_arcnames_dict(zip_path=zip_path,
                       paths_with_arcnames=arcnames)
