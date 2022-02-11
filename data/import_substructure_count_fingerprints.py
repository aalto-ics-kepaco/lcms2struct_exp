####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####
import sqlite3
import argparse
import gzip
import os
import pandas as pd
import glob
import logging
import traceback
import numpy as np

from utils import get_backup_db

# ================
# Setup the Logger
LOGGER = logging.getLogger("Import substructure count fingerprints")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.DEBUG)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


if __name__ == "__main__":
    # Setup and parse input arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument(
        "fingerprint_fn", help="CSV-file containing the substructure counting fingerprints for candidates.",

    )
    arg_parser.add_argument("--delimiter", default="\t")
    arg_parser.add_argument("--skip_backup", action="store_true")
    args = arg_parser.parse_args()

    # Backup old database and insert fingerprints into the copy
    if args.skip_backup:
        conn = sqlite3.connect(args.massbank_db_fn)
    else:
        conn = get_backup_db(args.massbank_db_fn, exists="raise", postfix="with_substructure_fps")

    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        with conn:
            # Insert meta-data
            name = "substructure_count__smiles_iso"
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints_meta "
                "  VALUES (?, ?, ?, ?, DATETIME('now', 'localtime'), ?, ?, ?, ?, ?)",
                (
                    name, "substructure", "count", "molecule_representation: smiles_iso", "CDK: 2.5", 307, False,
                    None, None
                )
            )

            # Create a new table for the fingerprints
            conn.execute(
                "CREATE TABLE IF NOT EXISTS fingerprints_data__%s("
                "  molecule    INTEGER NOT NULL PRIMARY KEY,"
                "  bits        VARCHAR NOT NULL,"
                "  vals        VARCHAR NOT NULL,"
                "  FOREIGN KEY (molecule) REFERENCES molecules(cid)"
                ")" % name
            )

        with conn:
            # Insert fingerprints
            with gzip.open(args.fingerprint_fn, "rt") as ifile:
                cnt = 0
                header = ifile.readline().strip()
                LOGGER.debug("Header: %s" % header)

                while True:
                    if (cnt % 10000) == 0:
                        LOGGER.debug("Process fingerprint: %d" % cnt)

                    # Read line containing a compounds and its fingerprint
                    line = ifile.readline().strip()
                    if line is None or line == "":
                        break

                    # Get compound identifier (CID) and the fingerprint string
                    cid, fp = line.split(args.delimiter)
                    cid = int(cid)

                    # Split the fingerprint string into indices and values
                    bits, vals = zip(*[fp_i.split(":") for fp_i in fp.split(",")])

                    # Insert new fingerprint
                    conn.execute(
                        "INSERT OR REPLACE INTO fingerprints_data__%s VALUES (?, ?, ?)" % name,
                        (cid, ",".join(bits), ",".join(vals))
                    )

                    cnt += 1

            n_cand = conn.execute("SELECT count() FROM molecules").fetchone()[0]
            if cnt != n_cand:
                LOGGER.warning(
                    "There is a missmatch between inserted and expected candidates: %d vs %d." % (cnt, n_cand)
                )
            else:
                LOGGER.debug("Inserted %d fingerprints for %d candidates." % (cnt, n_cand))

        # Create index on the molecules
        with conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS fpd__molecule__%s ON fingerprints_data__%s(molecule)" % (name, name)
            )

    except RuntimeError as err:
        traceback.print_exc()
        LOGGER.error(err)
    finally:
        conn.close()
