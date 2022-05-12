####
#
# The MIT License (MIT)
#
# Copyright 2021, 2022 Eric Bach <eric.bach@aalto.fi>
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
import glob
import os.path
import gzip
import pickle
import sys
import logging
import re
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import List

from run_with_gridsearch import dict2fn, get_topk_score_df, aggregate_candidates

# ----------------------------------------------------------------------------------------------------------------------
# Description of the margin aggregation procedure
# ----------------------------------------------------------------------------------------------------------------------
#
# In our experiments we train eight (8) SSVM models for each evaluation set. The max-marginal scores (see Eq. (4)) for
# each molecular candidate and each MS-feature in a particular evaluation LC-MS2 experiment are predicted with each SSVM
# model independently.
#
#   results_raw/publication/ssvm_lib=vs__exp_ver=4/massbank/
#   └── ds=AU_002__lloss_mode=mol_feat_fps__mol_feat=FCFP__binary__all__2D__mol_id=cid__ms2scorer=sirius__norm__ssvm_flavor=default/
#       └── ssvm_model_idx=0/
#           └── ...
#           └── marginals__spl=0.pkl.gz
#           └── ...
#       └── ssvm_model_idx=1/
#           └── ...
#           └── marginals__spl=0.pkl.gz
#           └── ...
#       └── ...
#       └── ssvm_model_idx=7/
#           └── ...
#           └── marginals__spl=0.pkl.gz
#           └── ...
#   └── ...
#
# The dictionaries (type: dict) "marginals__spl=0.pkl.gz" contain the max-marginal scores for all MS-features of
# evaluation LC-MS2 experiment 0 (spl=0). As an example, let's consider the max-marginals for a particular MS-feature
# s. Those scores are stored in a numpy-array with the length corresponding to the number of molecular candidates.
#
#   [0.1, 0.2, 0.9, 1, 0.8]
#
# For each SSVM model we have such an array:
#
#   SSVM 0: [0.1, 0.2, 0.9, 1.0, 0.8]
#   SSVM 1: [0.2, 0.1, 1.0, 0.9, 0.8]
#   ...
#   SSVM 7: [0.1, 0.2, 0.7, 1.0, 0.8]
#
# and we compute averaged max-marginal scores (see Methods "Feasible inference using random spanning trees (RST)"):
#
#   AVG: [0.133, 0.166, 0.866, 0.966, 0.8]
#
# Based on the average max-marginal scores, we compute the top-k accuracy. However, before the accuracy computation,
# we need to performa another aggregation step. That is, as we identify the molecular candidates by their PubChem id
# (cid), we can have the situation, that different stereo-isomers of the same 2D molecular structure are present in
# the molecular candidate set. For the experiments not considering the stereo-chemistry (first two paragraphs in the
# "Result" section), we eliminate the stereo-isomers by collapsing the (averaged) max-marginal scores based on their
# first InChIKey part (AAA-BBB-C --> AAA). Within each candidate "group", we can, however, have different scores for
# the different stereo-isomers. We therefore take the maximum score per group and assign it to the group [1].
#
#   [
#    0.133,     A1-B1-C
#    0.166,     A1-B2-C
#    0.866,     A2-B1-C
#    0.966,     A3-B1-C
#    0.8        A2-B2-C
#   ]
#
#   candidate grouping by inchikey1 ==>
#
#   [
#    0.166,     A1
#    0.866,     A2
#    0.966,     A3
#   ]
#
# Now, we can compute the top-k accuracy from the collapsed (candidate aggregated) averaged max-marginal scores. To
# compare the LC-MS2Struct performance with the "Only MS2" performance, we perform the same candidate aggregation also
# for the MS2 scores.
#
# This script, implements the above described max-margin averaging and candidate aggregation procedure. It, furthermore,
# stores the averaged margins (if requested) and computes the top-k accuracy:
#
#   results_processed/publication/ssvm_lib=vs__exp_ver=4/massbank/
#   └── ds=AU_002__lloss_mode=mol_feat_fps__mol_feat=FCFP__binary__all__2D__mol_id=cid__ms2scorer=sirius__norm__ssvm_flavor=default/
#       └── combined__cand_agg_id=inchikey1__marg_agg_fun=average/
#           └── ...
#           └── marginals__spl=0.pkl.gz
#           └── ...
#           └── top_k.tsv
#   └── ...
#
# REFERENCES
#
#   [1] Schymanski, Emma L. et al., Critical Assessment of Small Molecule Identification 2016: automated methods, 2017
#
# ----------------------------------------------------------------------------------------------------------------------


# ================
# Setup the Logger
LOGGER = logging.getLogger("combine_results")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================

SPL_PATTERN = re.compile(r".*=([0-9]+)\.tsv")
DS_PATTERN = re.compile(r"ds=([A-Z]{2,3}_[0-9]{3})")


def get_cli_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("base_dir", help="Directory containing the results for the different tree indices.")
    arg_parser.add_argument("ms2scorer", choices=["metfrag__norm", "sirius__norm", "cfmid4__norm"], type=str)
    arg_parser.add_argument("--candidate_aggregation_identifier", default=None, help="Identifier to use for collapsing the candidate scores.")
    arg_parser.add_argument("--dataset", default=None, help="If not None, only the margins of the specified MassBank setsub are aggregated.")
    arg_parser.add_argument("--write_out_averaged_margins", action="store_true")
    arg_parser.add_argument("--debug", action="store_true")

    # Constant parameters (legacy)
    arg_parser.add_argument("--margin_aggregation_fun", default="average", choices=["average"])
    arg_parser.add_argument(
        "--mol_feat_retention_order",
        type=str, default="FCFP__binary__all__3D", choices=["FCFP__binary__all__2D", "FCFP__binary__all__3D"]
    )
    arg_parser.add_argument("--molecule_identifier", type=str, default="cid", choices=["cid"])

    return arg_parser.parse_args()


def combine_margins_and_get_top_k(ssvm_model_result_dirs: List[str], dataset: str, sample_idx: int, output_dir: str):
    """
    Function to combine the marginals scores (mu) of the candidates predicted by different SSVM models.

    :param ssvm_model_result_dirs: list of strings, directories containing the results for the different SSVM models.

    :param dataset: string, identifier of the dataset

    :param sample_idx: scalar, index of the random evaluation sample (MS-feature sequence) for which the margins should
        be combined.

    :param output_dir: string, output directory for the aggregated marginals and top-k accuracies.

    :return:
    """
    marginals_out__msplrt = None
    marginals_out__onlyms = None

    # Top-k accuracy performance
    df_top_k = pd.DataFrame()
    df_top_k_max_models = pd.DataFrame()

    # We load the marginals associated with the different SSVM models in a random order
    for i, idir in enumerate(np.random.RandomState(sample_idx).permutation(ssvm_model_result_dirs)):
        ifn = os.path.join(idir, dict2fn({"spl": sample_idx}, pref="marginals", ext="pkl.gz"))
        with gzip.open(ifn, "rb") as ifile:
            # Load the marginals (dictionary also contains the Only MS scores)
            _marginals__msplrt = pickle.load(ifile)  # type: dict

            # Extract the Only MS scores
            _marginals__onlyms = extract_ms_score(_marginals__msplrt)  # type: dict
            assert _marginals__onlyms.keys() == _marginals__msplrt.keys()

            # Aggregate the Only-MS scores (MS2 scorers) by their candidate aggregation identifier. For example, if the
            # identifier is "inchikey1", than for all candidates with the same inchikey1 the highest MS2 score is chosen.
            # The output candidate set only contains a unique set of candidate identifiers.
            _marginals__onlyms = aggregate_candidates(
                _marginals__onlyms, args.candidate_aggregation_identifier
            )  # type: dict
        if i == 0:
            # We use the first SSVM-model as reference
            marginals_out__msplrt = _marginals__msplrt
            marginals_out__onlyms = _marginals__onlyms

            # For the marginals we need to construct a matrix with marginals scores in the row and columns corresponding
            # to the SSVM-models
            for s in marginals_out__msplrt:
                marginals_out__msplrt[s]["score"] = marginals_out__msplrt[s]["score"][:, np.newaxis]
        else:
            # Combine the marginals
            assert marginals_out__msplrt.keys() == _marginals__msplrt.keys()
            assert marginals_out__onlyms.keys() == _marginals__onlyms.keys()

            for s in _marginals__msplrt:
                assert marginals_out__msplrt[s].keys() == _marginals__msplrt[s].keys()
                assert marginals_out__onlyms[s].keys() == _marginals__onlyms[s].keys()

                # Perform some sanity checks
                for k in [
                    "spectrum_id", "correct_structure", "index_of_correct_structure", "label", "n_cand", "score"
                ]:
                    if k == "spectrum_id":
                        # Spectrum ID (=accession) needs to match
                        assert marginals_out__msplrt[s][k].get("spectrum_id") == _marginals__msplrt[s][k].get("spectrum_id")
                        assert marginals_out__onlyms[s][k].get("spectrum_id") == _marginals__onlyms[s][k].get("spectrum_id")
                        assert _marginals__msplrt[s][k].get("spectrum_id") == _marginals__onlyms[s][k].get("spectrum_id")
                    elif k == "score":
                        # Score should only be equal for Only-MS
                        assert np.all(marginals_out__onlyms[s][k] == _marginals__onlyms[s][k])
                    else:
                        assert np.all(marginals_out__msplrt[s][k] == _marginals__msplrt[s][k])
                        assert np.all(marginals_out__onlyms[s][k] == _marginals__onlyms[s][k])

                # Add up the normalized marginals
                assert np.allclose(1.0, np.max(_marginals__msplrt[s]["score"]))
                assert np.allclose(1.0, np.max(_marginals__onlyms[s]["score"]))
                marginals_out__msplrt[s]["score"] = np.hstack((marginals_out__msplrt[s]["score"], _marginals__msplrt[s]["score"][:, np.newaxis]))
                assert marginals_out__msplrt[s]["score"].shape == (marginals_out__msplrt[s]["n_cand"], i + 1)

        # Calculate the ranking performance
        for km in ["csi"]:  # could also use "casmi"
            # Aggregated marginals
            _marginals__msplrt = aggregate_candidates(aggregate_scores(marginals_out__msplrt), args.candidate_aggregation_identifier)
            for s in _marginals__msplrt:
                assert np.all(_marginals__msplrt[s]["label"] == marginals_out__onlyms[s]["label"])

            # LC-MS2Struct performance
            _tmp = get_topk_score_df(None, _marginals__msplrt, topk_method=km, scoring_method="MS + RT") \
                .assign(n_models=(i + 1), eval_indx=sample_idx, dataset=dataset)
            _tmp["top_k_acc"] = (_tmp["correct_leq_k"] / _tmp["seq_length"]) * 100

            # Only-MS performance
            _tmp_baseline = get_topk_score_df(None, marginals_out__onlyms, topk_method=km, scoring_method="Only MS") \
                .assign(n_models=(i + 1), eval_indx=sample_idx, dataset=dataset)
            _tmp_baseline["top_k_acc"] = (_tmp_baseline["correct_leq_k"] / _tmp_baseline["seq_length"]) * 100

            df_top_k = pd.concat((df_top_k, _tmp, _tmp_baseline), ignore_index=True)

            if i == (len(ssvm_model_result_dirs) - 1):
                df_top_k_max_models = pd.concat((df_top_k_max_models, _tmp, _tmp_baseline), ignore_index=True)

    # Write out the aggregated marginals if requested
    if args.write_out_averaged_margins:
        with gzip.open(
                os.path.join(output_dir, dict2fn({"spl": sample_idx}, pref="marginals", ext="pkl.gz")), "wb"
        ) as ofile:
            marginals_out__msplrt = aggregate_candidates(
                aggregate_scores(marginals_out__msplrt), args.candidate_aggregation_identifier
            )  # Aggregate the max-marginal scores if the different SSVM models

            # Add the Only MS scores again
            for s in marginals_out__msplrt:
                assert np.all(marginals_out__msplrt[s]["label"] == marginals_out__onlyms[s]["label"])
                marginals_out__msplrt[s]["ms_score"] = marginals_out__onlyms[s]["score"]

            # Write out the dictionary
            pickle.dump(marginals_out__msplrt, ofile)

    return df_top_k, df_top_k_max_models


def extract_ms_score(marginals):
    marginals_out = deepcopy(marginals)
    for s in marginals:
        marginals_out[s]["score"] = marginals_out[s]["ms_score"]
    return marginals_out


def aggregate_scores(marginals, score_key="score"):
    marginals_out = deepcopy(marginals)

    for s in marginals:
        marginals_out[s]["score"] = np.mean(marginals[s][score_key], axis=1)

    return marginals_out


if __name__ == "__main__":
    args = get_cli_arguments()

    # Find all 'ds=*__*/ssvm_model_idx=*/top_k__spl=*.tsv'
    avail_res_df = []

    # Parameters describing the experiment
    setting = {
        "ds": "*",
        "mol_feat": args.mol_feat_retention_order,
        "mol_id": args.molecule_identifier,
        "ms2scorer": args.ms2scorer,
        "ssvm_flavor": "default",
        "lloss_mode": "mol_feat_fps"
    }

    # Glob the result files
    top_k_fns = glob.iglob(
        os.path.join(
            args.base_dir,
            dict2fn(setting, pref="debug" if args.debug else None),
            dict2fn({"ssvm_model_idx": "*"}),
            dict2fn({"spl": "*"}, pref="top_k", ext="tsv")
        )
    )

    # Parse information from the result files: Dataset, SSVM-model, ...
    for fn in top_k_fns:
        _ssvm_model, _topk = fn.split(os.sep)[-2:]
        _ds = DS_PATTERN.findall(fn)[0]
        avail_res_df.append([_ds, _ssvm_model, _topk, os.path.dirname(fn)])

    # DataFrame storing the (triton) result directories for each (sample, SSVM-model) combination
    avail_res_df = pd.DataFrame(avail_res_df, columns=["dataset", "ssvm_model", "sample", "fn"])

    # Aggregate the margins for each dataset separately
    for ds in avail_res_df["dataset"].unique():
        if (args.dataset is not None) and (ds != str(args.dataset)):
            continue

        LOGGER.info("Dataset: %s" % ds)

        _avail_res_df = avail_res_df[avail_res_df["dataset"] == ds].drop("dataset", axis=1)

        # Define output directory relative to the input directory
        _o_setting = {
            "ds": ds,
            "mol_feat": args.mol_feat_retention_order,
            "mol_id": args.molecule_identifier,
            "ssvm_flavor": "default",
            "lloss_mode": "mol_feat_fps",
            "ms2scorer": args.ms2scorer
        }

        output_dir = os.path.join(
            os.sep,
            os.path.join(*_avail_res_df["fn"].iloc[0].split(os.sep)[:-2]).replace("results_raw", "results_processed"),
            dict2fn(_o_setting, pref="debug" if args.debug else None),
            dict2fn(
                {
                    "marg_agg_fun": args.margin_aggregation_fun,
                    "cand_agg_id": str(args.candidate_aggregation_identifier)  # Make None --> 'None'
                },
                pref="combined"
            )
        )
        os.makedirs(output_dir, exist_ok=True)

        # For each sample we need to aggregate the tree marginals separately
        it_groupby_spl_idx = _avail_res_df.groupby(["sample"])["fn"].apply(list).iteritems()

        # We aggregate the marginals and calculate the top-k accuracy based on the new marginals
        df_combined = pd.DataFrame()
        df_combined_max_models = pd.DataFrame()

        for filename, model_dirs in it_groupby_spl_idx:
            # train / test sample index extracted from, e.g. top_k__spl=3.tsv
            sample_idx = int(SPL_PATTERN.findall(filename)[0])

            LOGGER.info("Sample index: {} (Number of models = {})".format(sample_idx, len(model_dirs)))

            # Average the marginals of all ssvm_models ('model_dirs') for a specific sample index
            _df, _df_max_models = combine_margins_and_get_top_k(model_dirs, ds, sample_idx, output_dir)

            df_combined = pd.concat((df_combined, _df), ignore_index=True)
            df_combined_max_models = pd.concat((df_combined_max_models, _df_max_models), ignore_index=True)

        df_combined.to_csv(os.path.join(output_dir, os.extsep.join(["top_k", "tsv"])), sep="\t", index=False)
        df_combined_max_models.to_csv(
            os.path.join(output_dir, os.extsep.join(["top_k__max_models", "tsv"])), sep="\t", index=False
        )

    sys.exit(0)
