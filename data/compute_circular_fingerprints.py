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
import logging
import argparse
import more_itertools as mit
import numpy as np

from rdkit import __version__ as rdkit_version

from rosvm.feature_extraction.featurizer_cls import CircularFPFeaturizer
from rosvm import __version__ as rosvm_version

from utils import get_backup_db

# ================
# Setup the Logger
LOGGER = logging.getLogger("Compute Circular Fingerprints")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def get_fp_meta_data(fprinter: CircularFPFeaturizer):
    fp_type = fprinter.fp_type
    fp_mode = fprinter.fp_mode
    name = "__".join([fp_type, fp_mode, args.key_training_set])  # e.g. ECFP__count__gt_structures
    param = ", ".join(["{}: {}".format(k, v) for k, v in fprinter.get_params(deep=False).items()])
    param += ", molecule_representation: %s" % args.molecule_representation
    param += ", n_candidates_for_training: %d" % args.n_candidates_for_training
    library = ", ".join(["{}: {}".format(p, v) for p, v in [("rosvm", rosvm_version), ("RDKit", rdkit_version)]])
    length = fprinter.get_length()
    is_folded = int(fprinter.fp_mode == "binary_folded")

    if hasattr(fprinter, "freq_hash_set_"):
        hash_keys = ",".join(["%d" % k for k in fprinter.freq_hash_set_.keys()])
    else:
        hash_keys = None

    return name, fp_type, fp_mode, param, library, length, is_folded, hash_keys


def get_fingerprints(batch, fprinter):
    cids, smis = map(list, zip(*batch))
    return cids, fprinter.transform(smis)


if __name__ == "__main__":
    # Read CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument("--fp_type", type=str, choices=["ECFP", "FCFP"], default="FCFP")
    arg_parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of parallel jobs used to compute the fingerprints.")
    arg_parser.add_argument("--key_training_set", type=str, choices=["gt_structures", "random_candidates", "all"],
                            default="all")
    arg_parser.add_argument("--batch_size", type=int, default=2**14,
                            help="Size of the batches in which the fingerprints are computed and inserted to the DB.")
    arg_parser.add_argument("--molecule_representation", type=str, default="smiles_iso", choices=["smiles_iso"])
    arg_parser.add_argument("--radius", default=2)
    arg_parser.add_argument("--n_candidates_for_training", type=int, default=50000)
    arg_parser.add_argument("--min_subs_freq", type=int, default=75)
    args = arg_parser.parse_args()

    # Open connection to database
    conn = get_backup_db(args.massbank_db_fn, exists="raise", postfix="with_fcfp")

    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        # tmp, = conn.execute(
        #     "SELECT param FROM fingerprints_meta WHERE name IS '%s__count__all__3D'" % args.fp_type
        # ).fetchone()
        # cids_to_load = tmp.split(",")[-1].split(":")[-1].split(";")
        # rows = conn.execute(
        #     "SELECT cid, %s FROM molecules WHERE cid in %s" %
        #     (
        #         args.molecule_representation,
        #         "(" + ",".join(cids_to_load) + ")"
        #     )
        # ).fetchall()

        # Load the SMILES used to train the frequent circular fingerprint hashes
        rows = []

        if args.key_training_set in ["random_candidates", "all"]:
            rows += conn.execute(
                "SELECT cid, %s FROM molecules ORDER BY random() LIMIT %d" %
                (args.molecule_representation, args.n_candidates_for_training)
            ).fetchall()

        if args.key_training_set in ["gt_structures", "all"]:
            rows += conn.execute(
                "SELECT cid, m.%s FROM scored_spectra_meta "
                "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid" % args.molecule_representation
            ).fetchall()

        LOGGER.info("Number of SMILES used for training (%s): %d." % (args.key_training_set, len(rows)))

        # Train the circular fingerprinter
        cids_train, smis_train = zip(*rows)

        for use_chirality in [False, True]:
            fprinter = CircularFPFeaturizer(
                fp_type=args.fp_type, only_freq_subs=True, min_subs_freq=args.min_subs_freq, n_jobs=args.n_jobs,
                radius=args.radius, use_chirality=args.use_chirality, output_format="dense"
            ).fit(list(smis_train))
            LOGGER.info("Size of frequent hash set: %d" % len(fprinter))

            # Insert fingerprint meta-data
            with conn:
                name, fp_type, fp_mode, param, library, length, is_folded, hash_keys = get_fp_meta_data(fprinter)
                if use_chirality:
                    name += "__3D"
                else:
                    name += "__2D"

                # Track the CIDs used to estimate the frequent substructures
                param += ", cids: %s" % ";".join(map(str, cids_train))

                conn.execute(
                    "INSERT OR REPLACE INTO fingerprints_meta "
                    "  VALUES (?, ?, ?, ?, DATETIME('now', 'localtime'), ?, ?, ?, ?, ?)",
                    (name, fp_type, fp_mode, param, library, length, is_folded, hash_keys, None)
                )

            # Get all molecular candidate structures from the DB
            rows = conn.execute("SELECT cid, %s FROM molecules" % args.molecule_representation).fetchall()

            # Insert table for the new fingerprints
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS fingerprints_data__%s("
                             "  molecule    INTEGER NOT NULL PRIMARY KEY,"
                             "  bits        VARCHAR NOT NULL,"
                             "  vals        VARCHAR,"
                             "  FOREIGN KEY (molecule) REFERENCES molecules(cid))" % name)

            # Compute and insert fingerprints
            max_cnts = np.full(length, fill_value=-np.inf)
            batches = list(mit.chunked(rows, args.batch_size))
            for idx, batch in enumerate(batches):
                LOGGER.info("Batch: %d/%d" % (idx + 1, len(batches)))

                cids, fps = get_fingerprints(batch, fprinter)

                with conn:
                    conn.executemany(
                        "INSERT INTO fingerprints_data__%s VALUES (?, ?, ?)" % name,
                        [
                            (
                                cid_i,
                                ",".join(map(str, np.flatnonzero(fp_i))),
                                ",".join(map(str, fp_i[fp_i != 0]))
                            )
                            for cid_i, fp_i in zip(cids, fps)
                        ]
                    )

                max_cnts = np.maximum(max_cnts, np.max(fps, axis=0))

            # Insert statistics of the maximum count per feature dimension needed to convert the fingerprints to binary
            # vectors later.
            with conn:
                conn.execute(
                    "UPDATE fingerprints_meta SET max_values = ? WHERE name IS ?",
                    (",".join(map(str, max_cnts.astype(int))), name)
                )

            # Create index on the molecules
            with conn:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS fpd__molecule__%s ON fingerprints_data__%s(molecule)" % (name, name)
                )

    finally:
        conn.close()

