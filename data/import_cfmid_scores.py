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
import os
import time

import pandas as pd
import logging
import traceback
import numpy as np

from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from matchms import __version__ as __matchms_version__

from massbank2db.utils import get_precursor_mz
from massbank2db.spectrum import MBSpectrum

from utils import get_backup_db


# ================
# Setup the Logger
LOGGER = logging.getLogger("Import CFM-ID Scores")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.DEBUG)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def toSpectrum(ispec: MBSpectrum) -> Spectrum:
    return Spectrum(np.array(ispec.get_mz()), np.array(ispec.get_int()), ispec.get_meta_information())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument("working_dir", help="Input directory containing the measured MS2 spectra and meta-data.")

    arg_parser.add_argument(
        "--pred_spec_dir",
        help="Input directory containing the predicted MS2 spectra.",
        default=None,
        type=str
    )
    arg_parser.add_argument(
        "--build_unittest_db",
        action="store_true",
        help="Use this option to create a smaller database (subset of SIRIUS scores) that we can use for unittests."
    )
    args = arg_parser.parse_args()

    if args.pred_spec_dir is None:
        args.pred_spec_dir = args.working_dir

    sqlite3.register_adapter(np.int64, int)

    conn = get_backup_db(args.massbank_db_fn, exists="raise", postfix="with_cfmid")

    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        with conn:
            ms2scorer_name = os.path.split(args.working_dir.rstrip(os.path.sep))[-1]  # e.g. tool_output/cfmid2/ --> cfmid2
            description = \
                "The CFM-ID software was used to predict the MS2 spectra for all candidates \
                associated with the MS2 spectra in Massbank. We use the single energy (se) pre-trained models from \
                the online repository for the spectra prediction. The predictions are done in a structure disjoint \
                fashion. That means, the pre-trained model for each Massbank spectrum (and its candidates) is chosen \
                so that the ground truth structure was not part of the training. The similarity between the predicted \
                and measured MS2 spectra is computed using the modified cosine similarity (matchms Python package, \
                version %s). As in the original CFM-ID implementation, the similarity is computed for each energy \
                separately and the scores are summed up. If the measured MassBank entry was available at multiple \
                collision energies (i.e. we have multiple measured spectra for the same compound), MS2 spectra have \
                been merged using the 'mzClust_hclust' function from XCMS prior to the similarity scoring." % __matchms_version__

            if ms2scorer_name.endswith("2"):
                version = 2
            elif ms2scorer_name.endswith("4"):
                version = 4
            else:
                raise ValueError("Invalid CFM-ID version: '%s'" % ms2scorer_name)

            # We store normalized scores
            ms2scorer_name += "__norm"

            conn.execute(
                "INSERT OR REPLACE INTO scoring_methods VALUES (?, ?, ?)",
                (ms2scorer_name, "CFM-ID: %d" % version, description)
            )

        # Load the list spectra for which candidate spectra have been predicted
        df = pd.DataFrame()
        for ion_mode in ["neg", "pos"]:
            df = pd.concat(
                (
                    df,
                    pd.read_csv(os.path.join(args.working_dir, ion_mode, "spec2model.tsv"), sep="\t")
                ),
                axis=0,
                ignore_index=True
            )

        for idx, (spec_id, gt_cid, cv, ion_mode, _) in df.iterrows():
            dataset, accession, precursor_mz, original_accessions, instrument_type = conn.execute(
                "SELECT dataset, accession, precursor_mz, original_accessions, instrument_type FROM scored_spectra_meta "
                "   INNER JOIN datasets d on d.name = scored_spectra_meta.dataset"
                "   WHERE accession IS ?", (spec_id, )
            ).fetchone()
            LOGGER.info(
                "Process spectrum %05d / %05d: id = %s, dataset = %s" % (idx + 1, len(df), spec_id, dataset)
            )

            # Merging parameters depend on the utilized MS device
            if "ITFT" in instrument_type or "QFT" in instrument_type:
                mzppm = 5
                mzabs = 0.001
            elif "QTOF" in instrument_type:
                mzppm = 20
                mzabs = 0.001
            else:
                mzppm = 5
                mzabs = 0.001

            # =======================================================================================
            # Load the measured spectrum of the unknown compound and score it against the candidates
            spec_unknown = []
            for oacc in original_accessions.split(","):
                _mzs, _ints = zip(*conn.execute(
                    "SELECT mz, intensity FROM main.spectra_peaks WHERE spectrum IS ? ORDER BY mz", (oacc, )
                ).fetchall())

                spec_unknown.append(MBSpectrum())
                spec_unknown[-1].set_mz(_mzs)
                spec_unknown[-1].set_int(_ints)
                spec_unknown[-1].set("accession", oacc)
                spec_unknown[-1].set("precursor_mz", precursor_mz)

            spec_unknown = MBSpectrum.merge_spectra(spec_unknown, merge_peak_lists=True, eppm=mzppm, eabs=mzabs)
            LOGGER.debug("Loaded original accessions and merged the spectra.")

            # Convert to spectrum object compatible with matchms
            spec_unknown = toSpectrum(spec_unknown)

            # Track the candidate scores
            cids = []
            scores = []

            # Get all molecule IDs of the candidates belonging to the current MB spectrum
            cur = conn.execute(
                "SELECT cid, monoisotopic_mass FROM candidates_spectra "
                "   INNER JOIN molecules m on m.cid = candidates_spectra.candidate"
                "   WHERE spectrum IS ?",
                (spec_id, )
            )

            _t_load = 0.0
            _t_sim = 0.0

            for cid, mm in cur:
                try:
                    # Load the predicted candidate spectra and merge the energies
                    s = time.time()
                    spec_candidate = MBSpectrum.from_cfmid_output(
                        os.path.join(args.pred_spec_dir, ion_mode, "predicted_spectra__cv=%d" % cv, "%s.log" % cid),
                        cfmid_4_format=(version == 4), merge_energies=True
                    )
                    spec_candidate.set(
                        "precursor_mz", get_precursor_mz(mm, "[M+H]+" if ion_mode == "pos" else "[M-H]-")
                    )

                    # Convert to spectrum object compatible with matchms
                    spec_candidate = toSpectrum(spec_candidate)
                    _t_load += (time.time() - s)

                    # Compute the spectra similarity
                    s = time.time()
                    scores.append(ModifiedCosine().pair(spec_unknown, spec_candidate).item()[0])
                    _t_sim += (time.time() - s)

                    cids.append(cid)
                except FileNotFoundError:
                    LOGGER.warning(
                        "Cannot find predicted spectrum for mol=%s, spec=%s, cv=%d, ion_mode=%s"
                        % (cid, spec_id, cv, ion_mode)
                    )
                except AssertionError:
                    LOGGER.warning(
                        "Something went wrong while parsing the predicted spectrum: %s, %s, %s" % (cid, cv, ion_mode)
                    )

            if len(scores) == 0:
                LOGGER.warning("[%s] Empty candidate set." % spec_id)
                continue
            else:
                LOGGER.info("[%s] Number of scored candidates (CFM-ID): %d" % (spec_id, len(scores)))

            if gt_cid not in cids:
                LOGGER.error(
                    "[%s] Correct molecular structure (cid = %d) is not in the candidate set." % (spec_id, gt_cid)
                )
                continue

            # Normalize the scores per candidate set
            scores = np.array(scores)

            max_s = np.max(scores)
            if max_s > 0:
                scores /= max_s
            else:
                scores = np.ones_like(scores)

            LOGGER.debug("Loading the spectra file took: %.4fs" % (_t_load / len(scores)))
            LOGGER.debug("Computing the spectra similarity took: %.4fs" % (_t_sim / len(scores)))
            # =======================================================================================

            # ===========================
            # Insert new data into the DB
            with conn:
                # CFM-ID candidate scores
                conn.executemany(
                    "INSERT OR REPLACE INTO spectra_candidate_scores VALUES (?, ?, ?, ?, ?)",
                    [
                        (spec_id, cid, ms2scorer_name, dataset, score) for cid, score in zip(cids, scores)
                    ]
                )
            # ===========================
    except RuntimeError as err:
        traceback.print_exc()
        LOGGER.error(err)
    finally:
        conn.close()
