import os
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from typing import List, Tuple, Dict, Union, Optional
from matchms.Spectrum import Spectrum

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from ssvm.data_structures import RandomSubsetCandSQLiteDB_Massbank, SequenceSample, LabeledSequence, CandSQLiteDB_Massbank
from ssvm.ssvm import StructuredSVMSequencesFixedMS2
from ssvm.ssvm_seq_spec_cand_db import StructuredSVMSequencesFixedMS2SeqSpecCandDB
from ssvm.version import __version__ as ssvm_lib_version
from ssvm.feature_utils import RemoveCorrelatedFeatures, get_rbf_gamma_based_in_median_heuristic

# ================
# Setup the Logger
LOGGER = logging.getLogger("parameter_study__massbank")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


D_SHORT_PARAMNAMES = {
    "L_max_train": "L_max",
    "L_min_train": "L_min",
    "label_loss": "ll",
    "mol_feat_label_loss": "mf_ll",
    "mol_feat_retention_order": "mf_ro",
    "molecule_identifier": "moli",
    "n_init_per_example": "n_i",
    "ssvm_update_direction": "update",
    "step_size_approach": "ssize",
    "seq_spec_cand_set": "sscs",
    "batch_size": "bs",
    "max_n_candidates_train": "n_cnd_tr",
    "n_epochs": "n_e",
    "n_trees_for_testing": "n_t"
}

D_LABEL_LOSS = {
    "FCFP__binary__all__3D": "tanimoto_loss",
    "FCFP__binary__all__2D": "tanimoto_loss",
    "bouwmeester__smiles_iso": "kernel_loss",
    "MOL_ID": "zeroone_loss"
}

D_MOL_KERNEL = {
    "fCFP__binary__all__3D": "tanimoto",
    "FCFP__binary__all__2D": "tanimoto",
    "bouwmeester__smiles_iso": "rbf__median",
}


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


def get_topk_score_df(
        seq_eval: Union[None, LabeledSequence], marginals_eval: Dict, topk_method: str, scoring_method: str,
        dataset: str
) -> pd.DataFrame:
    """

    :param seq_eval:
    :param marginals_eval:
    :param topk_method:
    :param scoring_method:
    :return:
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
                [len(marginals_eval)] * len(_scores),  # length of the test sequence
                [dataset] * len(_scores)
            )
        ),
        columns=["k", "top_k_method", "scoring_method", "correct_leq_k", "seq_length", "dataset"]
    )


def load_spectra_and_labels(dbfn: str, molecule_identifier: str) -> Tuple[List[Spectrum], List[str]]:
    """
    Loads all spectra ids, retention times and ground truth labels.
    """
    db = sqlite3.connect("file:" + dbfn + "?mode=ro", uri=True)

    try:
        # Read in spectra and labels
        res = db.execute(
            "SELECT accession, %s as molecule, retention_time, dataset FROM scored_spectra_meta"
            "   INNER JOIN datasets d on scored_spectra_meta.dataset = d.name"
            "   INNER JOIN molecules m on scored_spectra_meta.molecule = m.cid"
            "   WHERE retention_time >= 3 * column_dead_time_min"  # Filter non-interacting molecules      
            "     AND column_type IS 'RP'"                         # We consider only reversed phased (RP) columns
            % molecule_identifier,
        )

        spectra = [
            Spectrum(
                np.array([]), np.array([]),
                {
                    "spectrum_id": spec_id,  # e.g. AU02202954
                    "molecule_id": mol_id,   # e.g. InChIKey1, InChI, cid, ...
                    "retention_time": rt,
                    "dataset": ds,           # e.g. AU_001
                }
            )
            for spec_id, mol_id, rt, ds in res
        ]
    finally:
        db.close()

    # Labels / molecule ids
    labels = [spec.get("molecule_id") for spec in spectra]

    return spectra, labels


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


def get_cli_arguments() -> argparse.Namespace:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--db_fn", type=str, help="Path to the MassBank database.",
        default="/home/bach/Documents/doctoral/projects/lcms2struct/data/massbank.sqlite"
    )
    arg_parser.add_argument(
        "--output_dir", type=str, help="Base directory to store the Tensorboard logging files etc.",
        default="../../../results_raw/development/massbank/"
    )

    arg_parser.add_argument("parameter_to_study", type=str)
    arg_parser.add_argument("parameter_grid", type=str, nargs="+")

    arg_parser.add_argument(
        "--n_samples_train", type=int, default=360, help="Number of training sample sequences."
    )
    arg_parser.add_argument(
        "--n_samples_test", type=int, default=64, help="Number of test / validation sample sequences."
    )

    # =======================
    # SSVM default parameters
    # =======================
    arg_parser.add_argument("--n_epochs", type=int, default=3)
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--n_init_per_example", type=int, default=1)
    arg_parser.add_argument("--step_size_approach", type=str, default="linesearch")
    arg_parser.add_argument("--C", type=float, default=1)

    arg_parser.add_argument("--mol_feat_retention_order", type=str, default="FCFP__binary__all")
    arg_parser.add_argument("--label_loss", default="mol_fps_kernel_loss", choices=["zeroone_loss", "mol_fps_kernel_loss"])

    arg_parser.add_argument("--ms2scorer", type=str, default="sirius__sd__correct_mf__norm")
    arg_parser.add_argument("--molecule_identifier", type=str, default="inchikey1", choices=["inchikey1", "inchikey"])
    arg_parser.add_argument("--max_n_candidates_train", type=int, default=50)
    arg_parser.add_argument("--max_n_candidates_test", type=int, default=300)
    arg_parser.add_argument("--L_min_train", type=int, default=4)
    arg_parser.add_argument("--L_max_train", type=int, default=32)
    arg_parser.add_argument("--L_test", type=int, default=50)

    arg_parser.add_argument("--n_trees_for_testing", type=int, default=1)

    arg_parser.add_argument("--n_jobs", type=int, default=6)

    arg_parser.add_argument("--ssvm_update_direction", type=str, default="map")
    arg_parser.add_argument("--seq_spec_cand_set", type=int, default=0)

    arg_parser.add_argument("--no_tensorboard_output", action="store_true")

    arg_parser.add_argument("--debug", action="store_true")

    arg_parser.add_argument("--std_threshold", type=float, default=0.01)
    arg_parser.add_argument("--corr_threshold", type=float, default=0.98)

    return arg_parser.parse_args()


def train_and_score(parameter_name: str, parameter_value: str):
    # Set the molecule kernel based on the feature
    args.__setattr__("mol_kernel", D_MOL_KERNEL[args.mol_feat_retention_order])

    if args.label_loss == "zeroone_loss":
        args.__setattr__("mol_feat_label_loss", "MOL_ID")
    else:
        # We always use the same features for the label loss and retention order
        args.__setattr__("mol_feat_label_loss", args.mol_feat_retention_order)

    args.__setattr__("label_loss", D_LABEL_LOSS[args.mol_feat_label_loss])

    # Handle multiple MS2 scorer
    if args.ms2scorer == "cfmid2_pl_metfrag":
        ms2scorer = ["cfmid2__norm", "metfrag__norm"]
    else:
        ms2scorer = args.ms2scorer

    # ===================================
    # Set parameters to the studied value
    # ===================================
    if parameter_name == "batch_size":
        args.batch_size = int(parameter_value)
    elif parameter_name == "max_n_candidates_train":
        args.max_n_candidates_train = int(parameter_value)
    elif parameter_name == "step_size_approach":
        args.step_size_approach = parameter_value
    elif parameter_name == "C":
        args.C = float(parameter_value)
    elif parameter_name == "n_epochs":
        args.n_epochs = int(parameter_value)
    elif parameter_name == "L_train":
        args.L_min_train = int(parameter_value)
        args.L_max_train = int(parameter_value)
    elif parameter_name == "n_samples_train":
        args.n_samples_train = int(parameter_value)
    elif parameter_name == "ssvm_update_direction":
        args.ssvm_update_direction = parameter_value
    elif parameter_name == "seq_spec_cand_set":
        args.seq_spec_cand_set = bool(parameter_value)
    elif parameter_name == "label_loss":
        if parameter_value == "zeroone_loss":
            args.mol_feat_label_loss = "MOL_ID"
        else:
            # We always use the same features for the label loss and retention order
            args.mol_feat_label_loss = args.mol_feat_retention_order

        args.label_loss = D_LABEL_LOSS[args.mol_feat_label_loss]
    elif parameter_name == "mol_feat_retention_order":
        if parameter_value == "zeroone_loss":
            args.mol_feat_label_loss = "MOL_ID"
        else:
            # We always use the same features for the label loss and retention order
            args.mol_feat_label_loss = args.mol_feat_retention_order

        args.label_loss = D_LABEL_LOSS[args.mol_feat_label_loss]
        args.mol_kernel = D_MOL_KERNEL[args.mol_feat_retention_order]
    elif parameter_name == "n_trees_for_testing":
        args.n_trees_for_testing = int(parameter_value)
    else:
        raise ValueError("Invalid parameter name: '%s'." % parameter_name)

    if args.debug:
        args.n_samples_train = 64
        args.n_samples_test = 16
        args.n_epochs = 2
        args.max_n_candidates_train = 15
        args.max_n_candidates_test = 15

    LOGGER.info("Parameters:")
    for k, v in args.__dict__.items():
        LOGGER.info("{} = {}".format(k, v))

    # ===================================================================
    # Set up the feature transformer based on the molecular features used
    # ===================================================================
    if args.mol_feat_retention_order == "bouwmeester__smiles_can":
        feature_transformer_pipeline = {
            "bouwmeester__smiles_can": Pipeline([
                ("feature_removal_low_variance_features", VarianceThreshold(threshold=(args.std_threshold ** 2))),
                ("feature_removal_correlated_features", RemoveCorrelatedFeatures(corr_threshold=args.corr_threshold)),
                ("feature_standardizer", StandardScaler())
            ])
        }
    else:
        feature_transformer_pipeline = None

    # ===================
    # Sequence Sample
    # ===================
    candidates = RandomSubsetCandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs_cand,
        number_of_candidates=args.max_n_candidates_train, include_correct_candidate=True, init_with_open_db_conn=False
    )

    candidates_test = RandomSubsetCandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs_cand,
        number_of_candidates=args.max_n_candidates_test, include_correct_candidate=True, init_with_open_db_conn=False
    )

    seq_sample = SequenceSample(
        spectra, labels, candidates=candidates, N=args.n_samples_train, L_min=args.L_min_train, L_max=args.L_max_train,
        random_state=rs_seqspl, ms_scorer=ms2scorer
    )

    seq_sample_train, seq_sample_test = seq_sample.get_train_test_split(
        spectra_cv=GroupShuffleSplit(random_state=rs_gss, test_size=0.33),  # 33% of the spectra a reserved for testing
        N_train=args.n_samples_train, N_test=args.n_samples_test, L_min_test=args.L_test, L_max_test=args.L_test,
        candidates_test=candidates_test, use_sequence_specific_candidates_for_training=args.seq_spec_cand_set)
    # type: SequenceSample, SequenceSample

    Ls_train = [len(seq) for seq in seq_sample_train]
    Ls_test = [len(seq) for seq in seq_sample_test]

    LOGGER.info(
        "Training sequences length: min = %d, max = %d, median = %d" % (
            min(Ls_train), max(Ls_train), np.median(Ls_train).item()
        )
    )
    LOGGER.info(
        "Test sequences length: min = %d, max = %d, median = %d" % (
            min(Ls_test), max(Ls_test), np.median(Ls_test).item()
        )
    )

    # -------------------------------------
    # Fit the feature transformer if needed
    if feature_transformer_pipeline is not None:
        # Load candidate features for 10% of the training spectra from the DB
        X_train_sub = load_candidate_features(args, candidates, seq_sample_train.get_spectra(), random_state=11)
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
        candidates.set_feature_transformer(feature_transformer_pipeline)
        candidates_test.set_feature_transformer(feature_transformer_pipeline)
    # -------------------------------------

    # ===================
    # Setup a SSVM
    # ===================
    if args.seq_spec_cand_set:
        ssvm_class = StructuredSVMSequencesFixedMS2SeqSpecCandDB
    else:
        ssvm_class = StructuredSVMSequencesFixedMS2

    # ------------------------
    # Handle kernel parameters
    mol_kernel = args.mol_kernel  # type: str

    if mol_kernel.startswith("rbf"):
        if mol_kernel.endswith("median"):
            # Load candidate features for 10% of the training spectra from the DB
            X_train_sub = load_candidate_features(
                args, candidates, seq_sample_train.get_spectra(), random_state=11
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
    # ------------------------

    ssvm = ssvm_class(
        mol_feat_label_loss=args.mol_feat_label_loss, mol_feat_retention_order=args.mol_feat_retention_order,
        mol_kernel=mol_kernel, C=args.C, step_size_approach=args.step_size_approach, batch_size=args.batch_size,
        n_epochs=args.n_epochs, label_loss=args.label_loss, random_state=rs_ssvm, n_jobs=args.n_jobs,
        update_direction=args.ssvm_update_direction, gamma=gamma
    )

    # ==============
    # Train the SSVM
    # ==============
    odir = os.path.join(
        args.output_dir,
        "ssvm_lib=v%s__exp=params" % ssvm_lib_version.split(".")[0],
        dict2fn({
            D_SHORT_PARAMNAMES.get(k, k): v
            for k, v in [
                (atr, args.__getattribute__(atr))
                for atr in [
                    "batch_size", "n_init_per_example", "step_size_approach", "mol_kernel", "C", "label_loss",
                    "mol_feat_retention_order", "ms2scorer", "molecule_identifier", "L_max_train", "L_min_train",
                    "ssvm_update_direction", "seq_spec_cand_set", "max_n_candidates_train", "n_epochs",
                    "n_trees_for_testing"
                ]
            ]
        },
            pref=("debug" if args.debug else None)
        )
    )
    os.makedirs(odir, exist_ok=True)

    if args.no_tensorboard_output:
        summary_writer = None
        validation_data = None
    else:
        summary_writer = tf.summary.create_file_writer(odir)
        validation_data = seq_sample_test

    ssvm.fit(
        seq_sample_train, n_init_per_example=args.n_init_per_example, summary_writer=summary_writer,
        validation_data=validation_data
    )

    # ===================
    # Score test sequence
    # ===================
    out = []

    # Max-marginals (MS + RT)
    scores = ssvm.score(
        seq_sample_test, stype="topk_mm", average=False, n_trees_per_sequence=args.n_trees_for_testing,
        topk_method="csi", return_percentage=False, spanning_tree_random_state=rs_score,
    )

    for i, seq in enumerate(seq_sample_test):
        for k in range(scores.shape[1]):
            out.append([i, len(seq), seq.get_dataset(), k + 1, scores[i, k], "MS + RT"])

    # Baseline performance (Only MS)
    scores = ssvm.score(
        seq_sample_test, stype="topk_mm", topk_method="csi", average=False, return_percentage=False,
        spanning_tree_random_state=rs_score, only_ms_performance=True
    )

    for i, seq in enumerate(seq_sample_test):
        for k in range(scores.shape[1]):
            out.append([i, len(seq), seq.get_dataset(), k + 1, scores[i, k], "Only MS"])

    # Write out top-k performance
    pd.DataFrame(out, columns=["sample_id", "sequence_length", "dataset", "k", "n_top_k", "method"]) \
        .to_csv(os.path.join(odir, "topk_mm.tsv"), index=False, sep="\t")

    # Write out parameters
    with open(os.path.join(odir, "parameters_.tsv"), "w+") as ofile:
        for k, v in args.__dict__.items():
            ofile.write("{} = {}\n".format(k, v))


if __name__ == "__main__":
    args = get_cli_arguments()

    # Random states
    rs_ssvm = 1993
    rs_cand = 391
    rs_seqspl = 25
    rs_gss = 103
    rs_score = 2942

    # ===================
    # Get list of Spectra
    # ===================
    spectra, labels = load_spectra_and_labels(args.db_fn, args.molecule_identifier)

    # ===============
    # Train and score
    # ===============
    for idx, value in enumerate(args.parameter_grid):
        LOGGER.info("Parameter: %s=%s %d/%d" % (args.parameter_to_study, value, idx + 1, len(args.parameter_grid)))
        train_and_score(args.parameter_to_study, value)



