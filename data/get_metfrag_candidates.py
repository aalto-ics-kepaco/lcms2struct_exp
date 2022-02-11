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
import sys
import gzip
import glob
import sqlite3
import logging
import argparse
import pandas as pd

from massbank2db.db import MassbankDB

# ================
# Setup the Logger
LOGGER = logging.getLogger("Get MetFrag Candidates")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


if __name__ == "__main__":
    # Read CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument("spectra_dir")
    arg_parser.add_argument("--gzip", action="store_true")
    args = arg_parser.parse_args()

    # Connect to the Massbank DB (with candidates!)
    conn = sqlite3.connect(args.massbank_db_fn)

    # Output writer
    fopener = lambda fn: gzip.open(fn, "wb") if args.gzip else lambda fn: open(fn, "w+")

    try:
        fn_list = glob.glob(os.path.join(args.spectra_dir, "*", "*.peaks"))
        for idx, fn in enumerate(fn_list):
            dataset = os.path.dirname(fn).split(os.sep)[-1]
            spec_id = os.path.basename(fn).split(os.extsep)[0]

            LOGGER.info("[%s] Process spectrum: %s (%05d / %05d)" % (dataset, spec_id, idx + 1, len(fn_list)))

            # Load candidates for current spectrum
            df = pd.read_sql(
                "SELECT monoisotopic_mass, inchi as InChI, cid, inchikey as InChIKey, molecular_formula, smiles_iso"
                "   FROM (SELECT candidate FROM candidates_spectra WHERE spectrum IS '%s') "
                "   INNER JOIN molecules ON molecules.cid = candidate"
                % spec_id, conn
            )

            # Prepare candidates for MetFrag
            cands = MassbankDB.candidates_to_metfrag_format(df, smiles_column="smiles_iso", return_as_str=False)

            # Write candidate lists out (ready to be used by MetFrag)
            ofn = fn.replace(os.extsep + "peaks", os.extsep + "cands")

            if args.gzip:
                ofn += (os.extsep + "gz")

            with fopener(ofn) as obuf:
                cands.to_csv(obuf, sep="|", index=False)

    finally:
        conn.close()

    sys.exit(0)
