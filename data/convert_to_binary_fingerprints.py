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
import logging
import argparse
import numpy as np

from typing import List, Tuple

from ssvm.feature_utils import CountingFpsBinarizer
from ssvm.version import __version__ as ssvm_version

from utils import get_backup_db

# ================
# Setup the Logger
LOGGER = logging.getLogger("Convert to Binary Fingerprints")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def get_fp_matrix(res: List[Tuple], d: int):
    cids = []
    X = np.zeros((len(res), d), dtype=int)

    for i, (cid, bits, vals) in enumerate(res):
        cids.append(cid)
        X[i, list(map(int, bits.split(",")))] = list(map(int, vals.split(",")))

    return cids, X


if __name__ == "__main__":
    # Read CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument("fp_type", type=str, choices=["ECFP", "FCFP"])
    arg_parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of parallel jobs used to compute the fingerprints.")
    arg_parser.add_argument("--key_training_set", type=str, choices=["gt_structures", "random_candidates", "all"],
                            default="all")
    arg_parser.add_argument("--batch_size", type=int, default=100000,
                            help="Size of the batches in which the fingerprints are computed and inserted to the DB.")
    args = arg_parser.parse_args()

    # Open connection to database
    conn = get_backup_db(args.massbank_db_fn, exists="raise", postfix="with_binary_fcfp")

    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        for chirality_string in ["2D", "3D"]:
            # ----------------------------
            # Insert fingerprint meta-data
            # ----------------------------
            fp_name_old = "__".join([args.fp_type, "count", args.key_training_set, chirality_string])
            fp_name_new = fp_name_old.replace("count", "binary")

            # 1) Load counting meta-data
            params, library, max_values, d_old = conn.execute(
                "SELECT param, library, max_values, length FROM main.fingerprints_meta WHERE name IS ?", (fp_name_old, )
            ).fetchone()

            # 2) Modify for binary fingerprints
            params = params.replace("fp_mode: count", "fp_mode: binary")
            params += (", bin_centers: based_on_max_values, derived_from: " + fp_name_old)

            library += (", ssvm: " + ssvm_version)

            # 3) Insert new meta-data
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO fingerprints_meta "
                    "  VALUES (?, ?, ?, ?, DATETIME('now', 'localtime'), ?, ?, ?, ?, ?)",
                    (fp_name_new, args.fp_type, "binary", params, library, -1, 0, None, None)
                )

            # 4) Create table for the fingerprint data
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS fingerprints_data__%s("
                             "  molecule    INTEGER NOT NULL PRIMARY KEY,"
                             "  bits        VARCHAR NOT NULL,"
                             "  vals        VARCHAR,"
                             "  FOREIGN KEY (molecule) REFERENCES molecules(cid))" % fp_name_new)

            # ---------------------
            # Binarize Fingerprints
            # ---------------------
            trans = CountingFpsBinarizer(bin_centers=[np.arange(int(mv)) + 1 for mv in max_values.split(",")])

            n_mols_total = conn.execute("SELECT count(*) FROM molecules").fetchone()[0]
            rows = conn.execute("SELECT molecule, bits, vals FROM fingerprints_data__%s" % fp_name_old)

            n_mols_processed = 0
            while res := rows.fetchmany(args.batch_size):
                # Get counting fingerprints as matrix
                cids, X = get_fp_matrix(res, d_old)

                # Transform counting to binary fingerprints
                X_trans = trans.transform(X) if n_mols_processed > 0 else trans.fit_transform(X)

                with conn:
                    conn.executemany(
                        "INSERT INTO fingerprints_data__%s(molecule, bits) VALUES (?, ?)" % fp_name_new,
                        [
                            (
                                cid,
                                ",".join(map(str, np.flatnonzero(fp_i)))
                            )
                            for cid, fp_i in zip(cids, X_trans)
                        ]
                    )
                n_mols_processed += len(res)
                LOGGER.info("Processed: %d out of %d" % (n_mols_processed, n_mols_total))

            with conn:
                conn.execute("UPDATE fingerprints_meta SET length = ? WHERE name IS ?", (len(trans), fp_name_new))

            # Create the index on the molecules
            conn.execute(
                "CREATE INDEX IF NOT EXISTS fpd__molecule__%s ON fingerprints_data__%s(molecule)" %
                (fp_name_new, fp_name_new)
            )

    finally:
        conn.close()

