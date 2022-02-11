####
#
# The MIT License (MIT)
#
# Copyright 2020 - 2022 Eric Bach <eric.bach@aalto.fi>
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
import os
import pandas as pd
import numpy as np
import itertools as it
import argparse
import pickle
import gzip
import logging
import sys
import time

from threadpoolctl import threadpool_limits
from typing import Tuple, Dict, Optional, Union, Type, List
from matchms.Spectrum import Spectrum

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandSQLiteDB_Massbank, CandSQLiteDB_Massbank
from ssvm.data_structures import LabeledSequence, SequenceSample, SpanningTrees
from ssvm.ssvm import StructuredSVMSequencesFixedMS2
from ssvm.feature_utils import get_rbf_gamma_based_in_median_heuristic, RemoveCorrelatedFeatures
from ssvm import __version__ as ssvm_version

# ================
# Setup the Logger
LOGGER = logging.getLogger("massbank_exp")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def get_cli_arguments() -> argparse.Namespace:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("eval_set_id", type=int, choices=range(0, 3200))
    arg_parser.add_argument(
        "ssvm_model_idx",
        type=int,
        help="The SSVM model index serves as random seed to the sequence (LC-MS data) generation as well as the random "
             "spanning tree sampling. In this way different SSVM models can be generated and subsequently combined, e.g., "
             "by margin aggregation."
    )
    arg_parser.add_argument("--n_jobs", type=int, default=1)
    arg_parser.add_argument("--n_threads_test_prediction", type=int, default=None)
    arg_parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    arg_parser.add_argument(
        "--db_fn",
        type=str,
        default="/home/bach/Documents/doctoral/projects/lcms2struct/data/massbank.sqlite",
        help="Path to the MassBank database."
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="./debugging/",
        help="Base directory to store the logging files, train and test splits, top-k accuracies, ..."
    )
    arg_parser.add_argument(
        "--n_samples_train",
        type=int,
        default=768,
        help="Number of training example sequences (= simulated LC-MS data)."
    )
    arg_parser.add_argument(
        "--n_epochs",
        type=int,
        default=4,
        help="Number of epochs (passes over the full dataset) used to optimize the final SSVM model after the "
             "hyper-parameter optimization."
    )
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument(
        "--n_init_per_example",
        type=int,
        default=1,
        help="Number of initially active dual variables (= candidate label sequences) per training example sequence."
    )
    arg_parser.add_argument(
        "--max_n_train_candidates",
        type=int,
        default=75,
        help="Maximum number of candidates per spectrum used for the training. Those candidates are randomly sampled."
    )
    arg_parser.add_argument("--stepsize", type=str, default="linesearch")
    arg_parser.add_argument(
        "--ms2scorer",
        type=str,
        default="sirius__norm",
        choices=["sirius__norm", "metfrag__norm", "cfmid4__norm"],
    )
    arg_parser.add_argument(
        "--mol_kernel",
        type=str,
        default="tanimoto",
        choices=["tanimoto", "generalized_tanimoto", "rbf__median"]
    )
    arg_parser.add_argument(
        "--molecule_identifier",
        type=str,
        default="cid",
        choices=["cid"],
        help="Molecule representation used as identifier to distinguish between molecules. If, for example, 'inchikey1' "
             "is used, than molecules with the same 2D structure but different stereo-chemistry are considered equal. "
             "For the model training all molecules of the evaluation set are removed from the training set using this "
             "identifier."
    )
    arg_parser.add_argument("--lloss_fps_mode", type=str, default="mol_feat_fps", choices=["mol_feat_fps"])

    arg_parser.add_argument("--C_grid", nargs="+", type=float, default=[0.125, 2048, 1, 512, 8, 128, 32])
    arg_parser.add_argument("--grid_scoring", type=str, default="ndcg_ohc", choices=["top1_mm", "ndcg_ohc"])
    arg_parser.add_argument("--L_min_train", type=int, default=4)
    arg_parser.add_argument("--L_max_train", type=int, default=32)
    arg_parser.add_argument(
        "--training_dataset",
        type=str,
        default="massbank",
        choices=["massbank", "massbank__with_stereo"]
    )
    arg_parser.add_argument(
        "--mol_feat_retention_order",
        type=str,
        default="FCFP__binary__all__2D",
        choices=["FCFP__binary__all__2D", "FCFP__binary__all__3D", "bouwmeester__smiles_iso"],
        help="The molecule feature representation used in the SSVM model."
    )
    arg_parser.add_argument("--load_optimal_C_from_tree_zero", type=int, default=0, choices=[0, 1])
    arg_parser.add_argument("--ssvm_flavor", type=str, default="default", choices=["default"])
    arg_parser.add_argument(
        "--test_frac_inner_split",
        type=float,
        default=1/3,
        help="Fraction of the spectra respectively sequences, available for training, used as validation set for the "
             "hyper-parameter selection."
    )

    # Parameters are should not be changed
    arg_parser.add_argument("--std_threshold", type=float, default=0.01)
    arg_parser.add_argument("--corr_threshold", type=float, default=0.98)

    # TODO: CAN BE REMOVED
    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=1, choices=[1])
    arg_parser.add_argument("--n_trees_for_training", type=int, default=1, choices=[1])

    return arg_parser.parse_args()


def aggregate_candidates(marginals: Dict[int, Dict], aggregation_identifier: Optional[str] = None) -> Dict[int, Dict]:
    """

    :param marginals:
    :param aggregation_identifier:
    :return:
    """
    if aggregation_identifier is None:
        return marginals

    if aggregation_identifier not in marginals[0]:
        raise ValueError(
            "The molecule identifier requested to be used for the aggregation of the candidates is not in the margin"
            "dictionary: '%s'." % aggregation_identifier
        )

    keys_to_copy = [aggregation_identifier, "score", "label"]

    # if "ms_score" in marginals[0]:
    #     keys_to_copy += ["ms_score"]

    marginals_out = {}
    for s in marginals:
        # Copy over information that do not change
        marginals_out[s] = {}
        if "spectrum_id" in marginals[s]:
            marginals_out[s]["spectrum_id"] = marginals[s]["spectrum_id"]

        # Get the aggregation-identifier of the correct molecular structure
        if np.isnan(marginals[s]["index_of_correct_structure"]):
            marginals_out[s]["correct_structure"] = None
        else:
            marginals_out[s]["correct_structure"] = \
                marginals[s][aggregation_identifier][marginals[s]["index_of_correct_structure"]]

        # Load the dictionary into a DataFrame
        tmp = pd.DataFrame({k: v for k, v in marginals[s].items() if k in keys_to_copy})

        # Sort the values by the aggregation-identifier (e.g. InChIKey1) and the candidate (margin) score.
        tmp.sort_values(by=[aggregation_identifier, "score"], ascending=False, inplace=True)

        # Drop duplicated aggregation-identifier and keep only the one with the highest score
        tmp.drop_duplicates(subset=aggregation_identifier, keep="first", ignore_index=True, inplace=True)

        # Rename axis: aggregation-identifier --> label
        tmp.rename({aggregation_identifier: "label", "label": "original_label"}, axis=1, inplace=True)

        for k, v in tmp.to_dict(orient="list").items():
            marginals_out[s][k] = v

        # Convert some lists into numpy arrays
        marginals_out[s]["score"] = np.array(marginals_out[s]["score"])

        # if "ms_score" in marginals_out[s]:
        #     marginals_out[s]["ms_score"] = np.array(marginals_out[s]["ms_score"])

        # Add more information to the output dictionary
        if marginals_out[s]["correct_structure"] is None:
            marginals_out[s]["index_of_correct_structure"] = np.nan
        else:
            marginals_out[s]["index_of_correct_structure"] = marginals_out[s]["label"].index(
                marginals_out[s]["correct_structure"]
            )
        marginals_out[s]["n_cand"] = len(marginals_out[s]["label"])

    return marginals_out


def get_topk_score_df(
        seq_eval: Union[None, LabeledSequence], marginals_eval: Dict[int, Dict], topk_method: str, scoring_method: str,
) -> pd.DataFrame:
    """
    Compute the top-k accuracy given the SSVM marginals.
    """
    _scores = StructuredSVMSequencesFixedMS2._topk_score(
        marginals_eval, seq_eval, topk_method=topk_method, max_k=50, return_percentage=False
    )
    return pd.DataFrame(
        data=list(
            zip(
                np.arange(len(_scores)) + 1,  # k
                [topk_method] * len(_scores),  # method to determine correct candidates <= k
                [scoring_method] * len(_scores),  # which scores where used for the ranking
                _scores,  # number of correct candidates <= k
                [len(marginals_eval)] * len(_scores)  # length of the test sequence
            )
        ),
        columns=["k", "top_k_method", "scoring_method", "correct_leq_k", "seq_length"]
    )


def get_hparam_estimation_setting(debug: bool) -> Tuple:
    """
    :param debug: boolean, indicating whether the settings should be returned for debug mode.

    :return: tuple (
        n_epochs_inner: scalar, Number of epochs used to train the SSVM model during hyper-parameter estimation
        C_grid: array-like, list of scalar values for the regularisation parameter C
    )
    """
    if debug:
        n_epochs_inner = 1
        C_grid = args.C_grid
    else:
        n_epochs_inner = 3
        C_grid = args.C_grid

    return n_epochs_inner, C_grid


def dict2fn(d: Dict, sep: str = "__", pref: Optional[str] = None, ext: Optional[str] = None) -> str:
    """

    :param d:
    :param sep:
    :return:
    """
    out = []

    for k, v in sorted(d.items()):
        if v is None:
            out.append("{}".format(k))
        else:
            out.append("{}={}".format(k, v))

    out = sep.join(out)

    if pref is not None:
        out = sep.join([pref, out])

    if ext is not None:
        out = os.extsep.join([out, ext])

    return out


def get_odir_res(eval_ds, args, ssvm_model_idx: Optional[int] = None) -> str:
    """
    Generate the output directory given the evaluation parameters.

    :param eval_ds:
    :param args: argparse.Namespace, command line interface arguments
    :param ssvm_model_idx: scalar, index of the Structured SVM model
    :return: string, path to the output directory
    """
    if ssvm_model_idx is None:
        ssvm_model_idx = args.ssvm_model_idx

    return os.path.join(
        args.output_dir, args.training_dataset, dict2fn(
            {
                "ds": eval_ds,
                "ms2scorer": args.ms2scorer,
                "lloss_mode": args.lloss_fps_mode,
                "mol_feat": args.mol_feat_retention_order,
                "ssvm_flavor": args.ssvm_flavor,
                "mol_id": args.molecule_identifier
            },
            pref=("debug" if args.debug else None)
        ),
        "ssvm_model_idx=%d" % ssvm_model_idx
    )


def get_ssvm_cls(args) -> Type[StructuredSVMSequencesFixedMS2]:
    """

    :param args:
    :return:
    """
    if args.ssvm_flavor == "default":
        ssvm_cls = StructuredSVMSequencesFixedMS2
    else:
        raise ValueError("Invalid SSVM flavor: '%s'. Choices are 'default'." % args.ssvm_flavor)

    return ssvm_cls


def parse_eval_set_id_information(args) -> Tuple[str, int, int]:
    """
    :param args: argparse.Namespace, command line interface arguments

    :return: tuple (
        string, target dataset name to evaluate
        scalar, index of the random sequence sample
        scalar, evaluation set id (simply the index passed as argument), can be for example used as random seed
    )
    """
    # Convert evaluation set id into fixed length string representation (with leading zeros)
    eval_set_id = "%04d" % args.eval_set_id  # 302 --> '0302'

    # Extract dataset name from the DB corresponding the evaluation set
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)
    try:
        ds, = db.execute("SELECT name FROM datasets ORDER BY name").fetchall()[int(eval_set_id[:2])]
    finally:
        db.close()

    # Get the sub-sample set index
    spl_idx = int(eval_set_id[2:])

    return ds, spl_idx, args.eval_set_id


def load_candidate_features(
        args, candidates: Union[CandSQLiteDB_Massbank, RandomSubsetCandSQLiteDB_Massbank], spectra: List[Spectrum],
        sample_size: float = 0.1, random_state: Optional[int] = None
) -> np.ndarray:
    """

    :param args: argparse.Namespace, command line interface arguments

    :param candidates: candidate db wrapper to load the molecule features

    :param spectra: list of Spectra, available spectra. Are used to define the candidate sets from which we load
        the candidate features.

    :param sample_size: float, fraction of spectra for which the candidates are loaded (random sample)

    :param random_state: scalar or None, random state

    :return:
    """
    with candidates:
        X = np.vstack([
            candidates.get_molecule_features(spec, args.mol_feat_retention_order)
            for spec in np.random.RandomState(random_state).choice(
                spectra, size=int(np.round(sample_size * len(spectra))), replace=False
            )
        ])

    return X


if __name__ == "__main__":
    LOGGER.info("We run the SSVM version: {}".format(ssvm_version))

    # ===================
    # Parse arguments
    # ===================
    args = get_cli_arguments()
    LOGGER.info("=== Arguments ===")
    for k, v in args.__dict__.items():
        LOGGER.info("{} = {}".format(k, v))

    # Handle parameters regarding the label-loss
    if args.lloss_fps_mode == "mol_feat_fps":
        mol_feat_label_loss = args.mol_feat_retention_order

        if args.mol_feat_retention_order in ["FCFP__binary__all__2D", "FCFP__binary__all__3D"]:
            # Binarized FCFP fingerprints are computed with the tanimoto loss
            label_loss = "tanimoto_loss"
        elif args.mol_feat_retention_order == "bouwmeester__smiles_iso":
            label_loss = "kernel_loss"
        else:
            raise ValueError("Invalid molecule feature: %s" % args.mol_feat_retention_order)
    else:
        raise ValueError("Invalid 'lloss_fps_mode': %s" % args.lloss_fps_mode)

    # Set up the feature transformer based on the molecular features used
    if args.mol_feat_retention_order == "bouwmeester__smiles_iso":
        feature_transformer_pipeline = {
            "bouwmeester__smiles_iso": Pipeline([
                ("feature_removal_low_variance_features", VarianceThreshold(threshold=(args.std_threshold ** 2))),
                ("feature_removal_correlated_features", RemoveCorrelatedFeatures(corr_threshold=args.corr_threshold)),
                ("feature_standardizer", StandardScaler())
            ])
        }
    else:
        feature_transformer_pipeline = None

    if args.debug:
        LOGGER.warning("WE RUN IN DEBUG MODE")

    n_epochs_inner, C_grid = get_hparam_estimation_setting(args.debug)

    # ==================================
    # Get the evaluation dataset from ID
    # ==================================
    eval_ds, eval_spl_idx, eval_set_id = parse_eval_set_id_information(args)

    LOGGER.info("=== Dataset ===")
    LOGGER.info("EVAL_SET_IDX = %d, DS = %s, SAMPLE_IDX = %d" % (eval_set_id, eval_ds, eval_spl_idx))

    if args.load_optimal_C_from_tree_zero and (args.ssvm_model_idx > 0):
        LOGGER.warning("C Value will be loaded from the hype-parameter search of tree with index 0.")
        _C_opt = None

        with open(
                os.path.join(
                    get_odir_res(eval_ds, args, ssvm_model_idx=0),
                    dict2fn({"spl": eval_spl_idx}, pref="parameters", ext="list")
                ),
                "r"
        ) as ofile:
            for line in ofile.read().splitlines():
                if line.startswith("C_opt"):
                    _C_opt = float(line[len("C_opt = "):])
                    break

        if _C_opt is None:
            raise RuntimeError("Could not load results from tree index 0.")

        C_grid = [_C_opt]  # grid only contains a single element --> hyper parameter search is skipped.

    # ===================
    # Get list of Spectra
    # ===================
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)

    # Read in spectra and labels
    if args.training_dataset == "massbank":
        res = db.execute(
            "SELECT accession, %s as molecule, retention_time, dataset, inchikey1 FROM scored_spectra_meta"
            "   INNER JOIN datasets d on scored_spectra_meta.dataset = d.name"
            "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid"
            "   WHERE retention_time >= 3 * column_dead_time_min"  # Filter non-interacting molecules      
            "     AND column_type == 'RP'"                         # We consider only reversed phased (RP) columns
            % args.molecule_identifier,
        )

        lcms_split_name = "default"

    elif args.training_dataset == "massbank__with_stereo":
        # Get the accessions for which at least one of the following requirements is satisfied:
        # 1) The molecule has a stereo-annotation
        # 2) The molecule has no stereo-annotation and the candidate set contains only a single conformer
        tmp_accs = db.execute(
            "select accession from ("
            "   select accession from scored_spectra_meta"
            "      inner join molecules m on m.cid = scored_spectra_meta.molecule"
            "      inner join datasets d on scored_spectra_meta.dataset = d.name"
            "   where retention_time >= 3 * column_dead_time_min"
            "     and inchikey2 != 'UHFFFAOYSA'"
            "     and column_type == 'RP'"
            " "
            "   union"
            " "
            "   select accession from ("
            "      select accession, count(distinct m2.inchikey) as cnt from ("
            "         select accession, dataset, inchikey1, inchikey2 from scored_spectra_meta"
            "            inner join molecules m on m.cid = scored_spectra_meta.molecule"
            "            inner join datasets d on scored_spectra_meta.dataset = d.name"
            "            where retention_time >= 3 * column_dead_time_min"
            "              and column_type == 'RP'"
            "              and inchikey2 == 'UHFFFAOYSA'"
            "      ) t1"
            "   inner join molecules m2 on m2.inchikey1 = t1.inchikey1"
            "   group by accession, m2.inchikey1"
            "   ) t2"
            "   where t2.cnt == 1"
            ")"
        )

        res = db.execute(
            "SELECT accession, %s as molecule, retention_time, dataset, inchikey1 FROM scored_spectra_meta"
            "   INNER JOIN datasets d on scored_spectra_meta.dataset = d.name"
            "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid"
            "   WHERE accession IN %s"
            % (args.molecule_identifier, "(" + ",".join("'%s'" % a for a, in tmp_accs) + ")")
        )

        lcms_split_name = "with_stereo"
    else:
        raise ValueError("Invalid training dataset: '%s'" % args.training_dataset)

    spectra = [
        Spectrum(
            np.array([]), np.array([]),
            {
                "spectrum_id": spec_id,  # e.g. AU02202954
                "molecule_id": mol_id,   # e.g. InChIKey1, InChI, cid, ..., it is used for the CV splitting
                "retention_time": rt,
                "dataset": ds,           # e.g. AU_001
                "cv_molecule_id": ikey1  # InChIKey1
            }
        )
        for spec_id, mol_id, rt, ds, ikey1 in res
    ]

    # ========================
    # Get test set spectra IDs
    # ========================

    spec_ids_eval = [
        spec_id for spec_id, in db.execute(
            "SELECT accession FROM lcms_data_splits"
            "   WHERE dataset IS ?"
            "     AND split_id IS ?"
            "     AND experiment IS ?",
            (eval_ds, eval_spl_idx, lcms_split_name)
        )
    ]
    if len(spec_ids_eval) == 0:
        raise ValueError(
            "No spectra for the dataset and split-id combination found: (%s, %d)" % (eval_ds, eval_spl_idx)
        )

    db.close()

    # Get the spectrum objects and their labels which are in the evaluation (test) set
    spectra_eval, labels_eval, cv_labels_eval = zip(*[
        (spectrum, spectrum.get("molecule_id"), spectrum.get("cv_molecule_id"))
        for spectrum in spectra
        if spectrum.get("spectrum_id") in spec_ids_eval
    ])

    # Get the spectrum objects and their labels for training: All spectra that are not in the evaluation set!
    spectra_train, labels_train, cv_labels_train = zip(*[
        (spectrum, spectrum.get("molecule_id"), spectrum.get("cv_molecule_id"))
        for spectrum in spectra
        if spectrum.get("cv_molecule_id") not in cv_labels_eval
    ])

    LOGGER.info(
        "Number of spectra: total = %d, train = %d, evaluation = %d"
        % (len(spectra), len(spectra_train), len(spectra_eval))
    )
    # ================================
    # Random state for reproducibility
    # ================================

    # Used for:
    # - Train-test-splitting of the training spectra
    # - Random candidate sub-sets for training and testing
    # - Sequence (LC-MS) generation for training and testing
    # - SSVM model: Choice of initially active sequence and training spanning trees
    # - Spanning trees imposed on the test sequences

    rs = (eval_set_id + 3) * (args.ssvm_model_idx + 1)

    # The evaluation set [0, 3200) and the SSVM model index effect the random state.

    # --------------------------------

    if len(C_grid) > 1:
        # =================================
        # Find the optimal hyper-parameters
        # =================================
        param_grid = list(it.product(C_grid))
        df_opt_values = pd.DataFrame(columns=["C", args.grid_scoring])

        # ------------------------------------
        # Inner cross-validation data-splitter

        #  - Splits the set of training labels, i.e. molecular structures
        #  - Training and test sets are structure disjoint

        train, test = next(
            GroupShuffleSplit(
                n_splits=1, test_size=args.test_frac_inner_split, random_state=rs
            ).split(spectra_train, groups=cv_labels_train)
        )
        train_frac_inner_split = (1 - args.test_frac_inner_split)

        spectra_train_inner = [spectra_train[i] for i in train]
        labels_train_inner = [spectra_train[i].get("molecule_id") for i in train]
        # ------------------------------------

        # -------------------------------------
        # Candidate DB Wrapper for the training
        candidate_set_training = RandomSubsetCandSQLiteDB_Massbank(
            db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, include_correct_candidate=True,
            random_state=rs, number_of_candidates=args.max_n_train_candidates, init_with_open_db_conn=False,
        )
        # -------------------------------------

        # -------------------------------------
        # Fit the feature transformer if needed
        if feature_transformer_pipeline is not None:
            # Load candidate features for 10% of the training spectra from the DB
            X_train_sub = load_candidate_features(args, candidate_set_training, spectra_train_inner, random_state=11)
            LOGGER.info(
                "Feature transformer is learned based on %d examples (with feature dimension = %d)."
                % X_train_sub.shape
            )

            # Fit the feature transformer pipeline
            _start = time.time()
            feature_transformer_pipeline[args.mol_feat_retention_order].fit(X_train_sub)
            LOGGER.info("Fitting the transformer pipeline took: %.3fs" % (time.time() - _start))
            LOGGER.info(
                "Original feature dimension = %d --> after feature selection = %d"
                % (
                    X_train_sub.shape[1],
                    feature_transformer_pipeline[args.mol_feat_retention_order].transform(X_train_sub[:1]).shape[1]
                )
            )

            # Update the feature transformer of the candidate DB wrapper
            candidate_set_training.set_feature_transformer(feature_transformer_pipeline)
        # -------------------------------------

        # -------------------------------------
        # Handle kernel parameters
        mol_kernel = args.mol_kernel  # type: str

        if mol_kernel.startswith("rbf"):
            if mol_kernel.endswith("median"):
                # Load candidate features for 10% of the training spectra from the DB
                X_train_sub = load_candidate_features(
                    args, candidate_set_training, spectra_train_inner, random_state=11
                )
                LOGGER.info(
                    "Median heuristic is evaluated on %d examples to estimate the RBF scaling parameter."
                    % X_train_sub.shape[0]
                )

                # Compute the gamma using the median heuristic: Features should be already are already z-transformed
                _start = time.time()
                gamma = get_rbf_gamma_based_in_median_heuristic(X_train_sub, standardize=False)
                LOGGER.info("Gamma of the RBF kernel (median heuristic): %f" % gamma)
                LOGGER.info("Computation took: %.3fs" % (time.time() - _start))
            else:
                gamma = None  # 1 / n_features

            mol_kernel = "rbf"
        else:
            gamma = None  # parameter not needed for the other kernels
        # -------------------------------------

        LOGGER.info("=== Search hyper parameter grid ===")
        LOGGER.info("C-grid: {}".format(C_grid))
        for idx, (C, ) in enumerate(param_grid):
            LOGGER.info("({} / {}) C = {}".format(idx + 1, len(param_grid), C))
            LOGGER.info("Number of spectra (inner cv): train = %d, test = %d" % (len(train), len(test)))

            # ----------------------
            # Get training sequences
            training_sequences = SequenceSample(
                spectra_train_inner, labels_train_inner, candidate_set_training, random_state=rs,
                N=int(np.round(args.n_samples_train * train_frac_inner_split)), L_min=args.L_min_train,
                L_max=args.L_max_train, ms_scorer=args.ms2scorer,
                use_sequence_specific_candidates=False
            )
            # ----------------------

            # --------------
            # Train the SSVM
            ssvm = get_ssvm_cls(args)(
                mol_feat_label_loss=mol_feat_label_loss, mol_feat_retention_order=args.mol_feat_retention_order,
                mol_kernel=mol_kernel, C=C, step_size_approach=args.stepsize, batch_size=args.batch_size,
                n_epochs=n_epochs_inner, label_loss=label_loss, random_state=rs, n_jobs=args.n_jobs,
                n_trees_per_sequence=args.n_trees_for_training, gamma=gamma
            ).fit(training_sequences, n_init_per_example=args.n_init_per_example)
            # --------------

            # ------------------------------------------------------------------
            # Access test set performance of the current hyper-parameter setting
            spectra_test_inner = [spectra_train[i] for i in test]
            labels_test_inner = [spectra_train[i].get("molecule_id") for i in test]

            candidate_set_class_test = RandomSubsetCandSQLiteDB_Massbank(
                db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs,
                number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
                init_with_open_db_conn=False, feature_transformer=candidate_set_training.get_feature_transformer()
            )

            test_sequences = SequenceSample(
                spectra_test_inner, labels_test_inner, candidate_set_class_test,
                N=int(np.round(args.n_samples_train * args.test_frac_inner_split)), L_min=args.L_min_train,
                L_max=args.L_max_train, random_state=rs, ms_scorer=args.ms2scorer,
                use_sequence_specific_candidates=False
            )

            _start = time.time()
            LOGGER.info("Score hyper-parameter tuple (%s) ..." % args.grid_scoring)
            df_opt_values = pd.concat((df_opt_values, pd.DataFrame(
                [[
                    C,
                    ssvm.score(test_sequences, stype=args.grid_scoring, spanning_tree_random_state=rs, max_k_ndcg=10)
                ]],
                columns=["C", args.grid_scoring]
            )))
            LOGGER.info("\n{}".format(df_opt_values.sort_values(by="C")))
            LOGGER.info("Scoring took: %.3fs" % (time.time() - _start))
            # ------------------------------------------------------------------

        df_opt_values = df_opt_values.set_index("C")
        C_opt = df_opt_values.idxmax().item()

        LOGGER.info("C_opt=%f" % C_opt)
        LOGGER.info("C-grid: {}".format(C_grid))
        LOGGER.info("Grid-score: %s" % args.grid_scoring)
        LOGGER.info("\n{}".format(df_opt_values.sort_index()))
    else:
        # We can skip the hyper-parameter search if only a single value for C is given
        df_opt_values = pd.DataFrame([[C_grid[0], np.nan]], columns=["C", args.grid_scoring]).set_index("C")
        C_opt = C_grid[0]
        LOGGER.info("C_opt=%f" % C_opt)

    # =========================================
    # Train the SSVM with best hyper-parameters
    # =========================================
    LOGGER.info("=== Train SSVM with all training data ===")

    candidate_set_training = RandomSubsetCandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs,
        number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
        init_with_open_db_conn=False
    )

    # Fit the feature transformer if needed
    if feature_transformer_pipeline is not None:
        # Load candidate features for 10% of the training spectra from the DB
        X_train_sub = load_candidate_features(args, candidate_set_training, spectra_train, random_state=11)
        LOGGER.info(
            "Feature transformer is learned based on %d examples (with feature dimension = %d)."
            % X_train_sub.shape
        )

        # Fit the feature transformer pipeline
        _start = time.time()
        feature_transformer_pipeline[args.mol_feat_retention_order].fit(X_train_sub)
        LOGGER.info("Fitting the transformer pipeline took: %.3fs" % (time.time() - _start))
        LOGGER.info(
            "Original feature dimension = %d --> after feature selection = %d"
            % (
                X_train_sub.shape[1],
                feature_transformer_pipeline[args.mol_feat_retention_order].transform(X_train_sub[:1]).shape[1]
            )
        )

        # Update the feature transformer of the candidate DB wrapper
        candidate_set_training.set_feature_transformer(feature_transformer_pipeline)

    training_sequences = SequenceSample(
        spectra_train, labels_train, candidate_set_training,
        N=args.n_samples_train, L_min=args.L_min_train, L_max=args.L_max_train, random_state=rs,
        ms_scorer=args.ms2scorer, use_sequence_specific_candidates=False)

    # Handle kernel parameters
    mol_kernel = args.mol_kernel  # type: str

    if mol_kernel.startswith("rbf"):
        if mol_kernel.endswith("median"):
            # Load candidate features for 10% of the training spectra from the DB
            X_train_sub = load_candidate_features(
                args, candidate_set_training, spectra_train, random_state=11
            )
            LOGGER.info(
                "Median heuristic is evaluated on %d examples to estimate the RBF scaling parameter."
                % X_train_sub.shape[0]
            )

            # Compute the gamma using the median heuristic: Features should be already are already z-transformed
            _start = time.time()
            gamma = get_rbf_gamma_based_in_median_heuristic(X_train_sub, standardize=False)
            LOGGER.info("Gamma of the RBF kernel (median heuristic): %f" % gamma)
            LOGGER.info("Computation took: %.3fs" % (time.time() - _start))
        else:
            gamma = None  # 1 / n_features

        mol_kernel = "rbf"
    else:
        gamma = None  # parameter not needed for the other kernels

    ssvm = get_ssvm_cls(args)(
        mol_feat_label_loss=mol_feat_label_loss, mol_feat_retention_order=args.mol_feat_retention_order,
        mol_kernel=mol_kernel, C=C_opt, step_size_approach=args.stepsize, batch_size=args.batch_size,
        n_epochs=args.n_epochs, label_loss=label_loss, random_state=rs,
        n_jobs=args.n_jobs, n_trees_per_sequence=args.n_trees_for_training, gamma=gamma
    ).fit(training_sequences, n_init_per_example=args.n_init_per_example)

    # ====================
    # Evaluate performance
    # ====================
    LOGGER.info("=== Evaluate SSVM performance on the evaluation set ===")
    candidates = CandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier,  init_with_open_db_conn=False,
        feature_transformer=candidate_set_training.get_feature_transformer()
    )

    with candidates:
        LOGGER.info("\tSpectrum - n_candidates:")
        for spec in spectra_eval:
            LOGGER.info("\t%s - %5d" % (spec.get("spectrum_id"), len(candidates.get_labelspace(spec))))

    # Wrap the spectra in to a label-sequence
    seq_eval = LabeledSequence(spectra_eval, labels=labels_eval, candidates=candidates, ms_scorer=args.ms2scorer)

    # Calculate the marginals for all candidates sets along the sequence
    with threadpool_limits(limits=args.n_threads_test_prediction):
        marginals_eval = ssvm.predict(
            seq_eval, Gs=SpanningTrees(seq_eval, n_trees=args.n_trees_for_scoring, random_state=rs)
        )

    df_top_k = pd.DataFrame()

    # ---------------------------------------------------------------
    # Calculate the top-k accuracies using the based on the marginals
    for km in ["casmi", "csi"]:
        df_top_k = pd.concat(
            (
                df_top_k,
                get_topk_score_df(seq_eval, marginals_eval, topk_method=km, scoring_method="MS + RT")
            )
        )
    # ---------------------------------------------------------------

    # --------------------------------------------------------
    # Calculate top-k accuracies using the original MS2 scores
    with candidates:
        candidates_with_ms2scores = {
            s: {
                "label": candidates.get_labelspace(spec),
                "index_of_correct_structure": candidates.get_labelspace(spec).index(lab),
                "score": candidates.get_ms_scores(spec, args.ms2scorer, return_as_ndarray=True),
                "n_cand": candidates.get_n_cand(spec)
            }
            for s, (spec, lab) in enumerate(zip(spectra_eval, labels_eval))
        }

        for km in ["casmi", "csi"]:
            df_top_k = pd.concat(
                (
                    df_top_k,
                    get_topk_score_df(
                        seq_eval, candidates_with_ms2scores, topk_method=km, scoring_method="Only MS"
                    )
                )
            )
    # --------------------------------------------------------

    df_top_k["top_k_acc"] = (df_top_k["correct_leq_k"] / df_top_k["seq_length"]) * 100

    # ----------------------------------------------
    # Collect some statistics about the training set
    n_train_spec = len(spectra_train)
    n_train_mol = len(set((spec.get("molecule_id") for spec in spectra_train)))
    n_train_spec_ds = len([spec for spec in spectra_train if spec.get("dataset") == eval_ds])
    n_train_mol_ds = len(set((spec.get("molecule_id") for spec in spectra_train if spec.get("dataset") == eval_ds)))
    n_spec_total_ds = len([spec for spec in spectra if spec.get("dataset") == eval_ds])
    n_mol_total_ds = len(set((spec.get("molecule_id") for spec in spectra if spec.get("dataset") == eval_ds)))

    df_train_stats = pd.DataFrame(
        {
            "dataset": eval_ds,
            "eval_indx": eval_spl_idx,
            "n_train_spec": n_train_spec,
            "n_train_mol": n_train_mol,
            "n_train_spec_ds": n_train_spec_ds,
            "n_train_mol_ds": n_train_mol_ds,
            "n_train_seq": args.n_samples_train,
            "n_spec_total_ds": n_spec_total_ds,
            "n_mol_total_ds": n_mol_total_ds
        },
        index=[0]
    )
    # ----------------------------------------------

    # =================
    # Write out results
    # =================
    LOGGER.info("=== Write out results ===")
    odir_res = get_odir_res(eval_ds, args)

    os.makedirs(odir_res, exist_ok=True)
    LOGGER.info("Output directory: %s" % odir_res)

    df_opt_values \
        .assign(eval_indx=eval_spl_idx, dataset=eval_ds) \
        .reset_index() \
        .to_csv(
            os.path.join(
                odir_res, dict2fn({"spl": eval_spl_idx}, pref="grid_search_results", ext="tsv"),
            ),
            sep="\t",
            index=False
        )

    df_top_k \
        .assign(eval_indx=eval_spl_idx, dataset=eval_ds) \
        .to_csv(
            os.path.join(
                odir_res, dict2fn({"spl": eval_spl_idx}, pref="top_k", ext="tsv"),
            ),
            sep="\t",
            index=False
        )

    df_train_stats \
        .to_csv(
            os.path.join(
                odir_res, dict2fn({"spl": eval_spl_idx}, pref="train_stats", ext="tsv"),
            ),
            sep="\t",
            index=False
        )

    with open(os.path.join(odir_res,  dict2fn({"spl": eval_spl_idx}, pref="eval_spec_ids", ext="list")), "w+") as ofile:
        ofile.write("\n".join(spec_ids_eval))

    with open(os.path.join(odir_res,  dict2fn({"spl": eval_spl_idx}, pref="parameters", ext="list")), "w+") as ofile:
        for k, v in args.__dict__.items():
            ofile.write("{} = {}\n".format(k, v))

        ofile.write("C_grid = {}\n".format(C_grid))
        ofile.write("C_opt = {}\n".format(C_opt))
        ofile.write("ssvm_version = {}\n".format(ssvm_version))

    with gzip.open(os.path.join(odir_res, dict2fn({"spl": eval_spl_idx}, pref="marginals", ext="pkl.gz")), "wb") as ofile:
        with candidates:
            for s, (spec, lab) in enumerate(zip(spectra_eval, labels_eval)):
                lab_space = candidates.get_labelspace(spec, return_inchikeys=True)

                marginals_eval[s]["spectrum_id"] = spec
                marginals_eval[s]["correct_structure"] = lab
                marginals_eval[s]["index_of_correct_structure"] = lab_space["molecule_identifier"].index(lab)

                # Add MS2 scores to the dictionary to ease downstream analysis
                assert len(marginals_eval[s]["score"]) == len(candidates_with_ms2scores[s]["score"])
                marginals_eval[s]["ms_score"] = candidates_with_ms2scores[s]["score"]

                # Add InChIKey information for candidate grouping
                marginals_eval[s]["inchikey"] = lab_space["inchikey"]
                marginals_eval[s]["inchikey1"] = lab_space["inchikey1"]

        pickle.dump(marginals_eval, ofile)

    sys.exit(0)
