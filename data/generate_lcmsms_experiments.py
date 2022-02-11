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
import argparse
import pandas as pd

from utils import get_backup_db

from sklearn.model_selection import ShuffleSplit, KFold


if __name__ == "__main__":
    # Read CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    args = arg_parser.parse_args()

    conn = get_backup_db(args.massbank_db_fn, exists="raise", postfix="with_test_splits")

    try:
        for experiment in ["default", "with_stereo"]:
            # Read accessions
            if experiment == "default":
                df = pd.read_sql(
                    "SELECT accession, dataset FROM scored_spectra_meta"
                    "   INNER JOIN datasets d on scored_spectra_meta.dataset = d.name"
                    "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid"
                    "   WHERE retention_time >= 3 * column_dead_time_min",
                    conn
                )
            else:  # experiment == "with_stereo":
                df = pd.read_sql(
                    "SELECT accession, dataset FROM scored_spectra_meta"
                    "   INNER JOIN datasets d on scored_spectra_meta.dataset = d.name"
                    "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid"
                    "   WHERE retention_time >= 3 * column_dead_time_min"
                    "     AND inchikey2 != 'UHFFFAOYSA'",
                    conn
                )

            # Insert a new table that will hold the test set split ids
            with conn:
                conn.execute("PRAGMA foreign_keys = ON")

                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lcms_data_splits ("
                    "   dataset     VARCHAR NOT NULL,"
                    "   accession   VARCHAR NOT NULL,"
                    "   split_id    INTEGER NOT NULL,"
                    "   experiment  VARCHAR NOT NULL,"
                    "   FOREIGN KEY (dataset) references datasets(name),"
                    "   FOREIGN KEY (accession) references scored_spectra_meta(accession),"
                    "   CONSTRAINT unique_lcms_data_split_row UNIQUE (dataset, accession, split_id, experiment))"
                )

                # Generate LC-MS2 data (random test sets) for each dataset
                for idx, ds in enumerate(df["dataset"].unique()):
                    # Subset accessions belonging to the current dataset
                    accs = df[df["dataset"] == ds]["accession"].tolist()
                    n_accs = len(accs)

                    # SQLite insertion statement
                    stmt = "INSERT OR IGNORE INTO lcms_data_splits VALUES ('%s', ?, ?, '%s')" % (ds, args.experiment)

                    if n_accs < 30:
                        # We do not add splits that have less than 30 examples
                        continue
                    if n_accs <= 75:
                        conn.executemany(stmt, zip(accs, [0] * n_accs))
                        continue  # only a single split is needed
                    elif n_accs <= 250:
                        cv = ShuffleSplit(n_splits=15, test_size=50, random_state=idx)
                    else:
                        cv = KFold(n_splits=(n_accs // 50), shuffle=True, random_state=idx)

                    # Split the data using the CV splitter and insert
                    for jdx, (_, test) in enumerate(cv.split(accs)):
                        conn.executemany(stmt, zip([accs[i] for i in test], [jdx] * len(test)))

    finally:
        conn.close()

