####
#
# The MIT License (MIT)
#
# Copyright 2022 Eric Bach <eric.bach@aalto.fi>
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
import argparse
import pandas as pd
import sqlite3
import logging
import traceback
import numpy as np

from utils import get_backup_db

# ================
# Setup the Logger
LOGGER = logging.getLogger("Import PubChemLite Categories")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument("pubchemlite_fn", help="Filepath of the PubChemLite CSV file.")
    args = arg_parser.parse_args()

    # Load the PubChemLite DB
    pcl_db = pd.read_csv(args.pubchemlite_fn, sep=",", index_col="FirstBlock")

    # Get a connection for a new DB
    conn = get_backup_db(args.massbank_db_fn, postfix="with_pubchemlite", exists="raise")
    sqlite3.register_adapter(np.int64, int)

    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pubchemlite_categories ("
                "molecule        INTEGER PRIMARY KEY NOT NULL,"
                "PubMed_Count    INTEGER NOT NULL DEFAULT 0,"
                "Patent_Count    INTEGER NOT NULL DEFAULT 0,"
                "AnnoTypeCount   INTEGER NOT NULL DEFAULT 0,"
                "AgroChemInfo    INTEGER NOT NULL DEFAULT 0,"
                "BioPathway      INTEGER NOT NULL DEFAULT 0,"
                "DrugMedicInfo   INTEGER NOT NULL DEFAULT 0,"
                "FoodRelated     INTEGER NOT NULL DEFAULT 0,"
                "PharmacoInfo    INTEGER NOT NULL DEFAULT 0,"
                "SafetyInfo      INTEGER NOT NULL DEFAULT 0,"
                "ToxicityInfo    INTEGER NOT NULL DEFAULT 0,"
                "KnownUse        INTEGER NOT NULL DEFAULT 0,"
                "DisorderDisease INTEGER NOT NULL DEFAULT 0,"
                "Identification  INTEGER NOT NULL DEFAULT 0,"
                "FOREIGN KEY (molecule) references molecules(cid))"
            )

        # Get the list of ground truth compounds
        with conn:
            cmps = conn.execute(
                "SELECT cid, inchikey1 FROM scored_spectra_meta "
                "   INNER JOIN molecules m ON m.cid = scored_spectra_meta.molecule"
            ).fetchall()

        # Insert the PubChemLite information
        with conn:
            for idx, (cid, ikey1) in enumerate(cmps):
                LOGGER.debug("Process: %s (%d/%d)" % (ikey1, idx + 1, len(cmps)))

                try:
                    row = pcl_db.loc[ikey1]  # type: pd.Series
                    pcl_cmp = (
                        cid,
                        row["PubMed_Count"], row["Patent_Count"], row["AnnoTypeCount"], row["AgroChemInfo"],
                        row["BioPathway"], row["DrugMedicInfo"], row["FoodRelated"], row["PharmacoInfo"],
                        row["SafetyInfo"], row["ToxicityInfo"], row["KnownUse"], row["DisorderDisease"],
                        row["Identification"]
                    )
                except KeyError:
                    LOGGER.warning(
                        "Cannot find InChIKey1=%s (with CID=%d) in PubChemLite. (%d/%d)"
                        % (ikey1, cid, idx + 1, len(cmps))
                    )
                    pcl_cmp = (
                        cid,
                        None, None, None, None,
                        None, None, None, None,
                        None, None, None, None,
                        None
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO pubchemlite_categories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pcl_cmp
                )

    except RuntimeError as err:
        traceback.print_exc()
        LOGGER.error(err)
    finally:
        conn.close()


