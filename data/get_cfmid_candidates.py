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
import os
import argparse
import pandas as pd
import numpy as np
import sqlite3
import logging
import gzip

from tqdm import tqdm

# ================
# Setup the Logger
LOGGER = logging.getLogger("Get CFM-ID Candidates")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================

IONIZATION_MODES = ["neg", "pos"]


def fopener(fn: str):
    # Output writer
    if args.gzip:
        return gzip.open(fn, "wt")
    else:
        return open(fn, "w")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("base_output_dir")
    arg_parser.add_argument("--massbank_db_fn", help="Filepath of the Massbank database.", default="./massbank.sqlite")
    arg_parser.add_argument("--gzip", action="store_true")
    arg_parser.add_argument("--store_candidates_separately", action="store_true")
    args = arg_parser.parse_args()

    # Read in training molecules (inchikeys) and their left-out cv-folds
    df_train = {}
    for imode in IONIZATION_MODES:
        df_train[imode] = pd.read_csv(os.path.join(args.base_output_dir, imode, "mol_list_cv.tsv"), sep="\t")
        df_train[imode]["INCHIKEY1"] = [ikey.split("-")[0] for ikey in df_train[imode]["INCHIKEY"]]

    # There is a candidate set for each CFM-ID model
    candidates = {imode: [set() for _ in range(10)] for imode in IONIZATION_MODES}

    # Track which model was used for which spectrum
    df_spec2model = {imode: [] for imode in IONIZATION_MODES}

    # Connect to db
    conn = sqlite3.connect(args.massbank_db_fn)
    try:
        # Get all spectrum ids and the corresponding InChIKey(1)s
        rows = conn.execute(
            "SELECT accession, cid, inchikey1, precursor_type FROM scored_spectra_meta"
            "   INNER JOIN molecules m on m.cid = scored_spectra_meta.molecule"
        ).fetchall()

        for idx, (acc, cid, ikey1, ptype) in tqdm(enumerate(rows), desc="Process spectra", total=len(rows)):
            # Determine ionization time
            if ptype.endswith("+"):
                imode = "pos"
            elif ptype.endswith("-"):
                imode = "neg"
            else:
                raise ValueError("Cannot determine ionization mode from precursor type: '%s'." % ptype)

            # Check for the spectrum, whether it is used for the CFM-ID training and if yes in which fold
            try:
                idx = df_train[imode]["INCHIKEY1"].tolist().index(ikey1)
                cv_fold = df_train[imode].iloc[idx]["CV"]
            except ValueError:
                cv_fold = np.random.RandomState(idx).randint(0, 10)  # Use a random fold as fallback

            # Get the candidates for the current spectrum
            for cid_can, smi_cnd in conn.execute(
                "SELECT cid, smiles_iso FROM candidates_spectra "
                "   INNER JOIN molecules m ON m.cid = candidates_spectra.candidate"
                "   WHERE spectrum IS ?", (acc, )
            ):
                # Add the molecule and its isomeric SMILES representation to prediction list for the current model
                candidates[imode][cv_fold] |= {(cid_can, smi_cnd)}

            # Track spectra information and their corresponding models
            df_spec2model[imode].append((acc, cid, cv_fold, imode, ikey1))
    finally:
        conn.close()

    # Write out which model is used for which spectrum
    for imode in IONIZATION_MODES:
        pd.DataFrame(df_spec2model[imode], columns=["accession", "cid", "cv_fold", "ionization", "inchikey1"]) \
            .to_csv(os.path.join(args.base_output_dir, imode, "spec2model.tsv"), sep="\t", index=False)

    # Write out the model specific candidate sets
    if args.store_candidates_separately:
        for imode in IONIZATION_MODES:
            for cv_fold in tqdm(range(10), desc="Write out candidate files (%s)" % imode):
                if len(candidates[imode][cv_fold]) > 0:
                    for cid, smi in candidates[imode][cv_fold]:
                        ofn = os.path.join(args.base_output_dir, imode, "%d__cv=%d.cand" % (cid, cv_fold))
                        with open(ofn, "w") as ofile:
                            ofile.write("%s %s\n" % (cid, smi))
    else:
        for imode in IONIZATION_MODES:
            for cv_fold in tqdm(range(10), desc="Write out candidate files (%s)" % imode):
                if len(candidates[imode][cv_fold]) > 0:
                    ofn = os.path.join(args.base_output_dir, imode, "candidates__cv=%d.csv" % cv_fold)

                    if args.gzip:
                        ofn += ".gz"

                    with fopener(ofn) as ofile:
                        for cid, smi in candidates[imode][cv_fold]:
                            ofile.write("%s %s\n" % (cid, smi))
