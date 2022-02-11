import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
import time
import seaborn as sns
import itertools as it


from typing import Union, Tuple, List
from scipy.stats import pearsonr, norm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from matchms import Spectrum

from ssvm.data_structures import CandSQLiteDB_Massbank, LabeledSequence, RandomSubsetCandSQLiteDB_Massbank
from ssvm.data_structures import ImputationError
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

# FIXME: Local imports --> move to separate place
from run_with_gridsearch import parse_eval_set_id_information, get_topk_score_df, dict2fn, aggregate_candidates


# ================
# Setup the Logger
LOGGER = logging.getLogger("XLogP3 prediction approach")
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
    arg_parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    arg_parser.add_argument(
        "--db_fn",
        type=str,
        help="Path to the MassBank database.",
        default="/home/bach/Documents/doctoral/projects/lcms2struct_experiments/data/massbank.sqlite"
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        help="Base directory to store the logging files, train and test splits, top-k accuracies, ...",
        default="./debugging/"
    )
    arg_parser.add_argument(
        "--molecule_identifier",
        type=str,
        default="cid",
        choices=["cid"],
        help="Identifier used to distinguish molecules in the candidate sets during the prediction."
    )
    arg_parser.add_argument(
        "--predictor",
        type=str,
        help="Prediction model used to establish the RT --> XLogP3 mapping.",
        default="linear_reg",
        choices=["linear_reg"]
    )
    arg_parser.add_argument(
        "--ms2scorer",
        type=str,
        default="metfrag__norm",
        choices=["sirius__norm", "metfrag__norm", "cfmid4__norm"],
    )

    arg_parser.add_argument(
        "--beta_grid",
        nargs="+",
        type=float,
        default=np.round(np.arange(0, 1 + 0.05, 0.05), 3).tolist()
    )
    arg_parser.add_argument("--n_splits_ranking", default=25, type=int)
    arg_parser.add_argument("--max_n_candidates_training", default=500, type=int)
    arg_parser.add_argument("--molecule_features", default="xlogp3", type=str, choices=["xlogp3"])
    arg_parser.add_argument("--no_plot", action="store_true")

    # Unused parameters (for compatibility with the slurm-scripts)
    arg_parser.add_argument("--n_jobs_scoring_eval", type=int, default=None)
    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=None)
    arg_parser.add_argument("--C_grid", nargs="+", type=float, default=None)
    arg_parser.add_argument("--n_jobs", type=int, default=None)
    arg_parser.add_argument("--n_tuples_sequences_ranking", type=int, default=None)

    # Constant parameters
    arg_parser.add_argument(
        "--score_integration_approach", default="score_combination", type=str, choices=["score_combination"]
    )

    return arg_parser.parse_args()


def find_optimal_beta(
        X: np.array, y: np.array, spec: List[Spectrum], predictor: LinearRegression, cv: GroupShuffleSplit,
        beta_grid: List[float], candidates: Union[CandSQLiteDB_Massbank, RandomSubsetCandSQLiteDB_Massbank]
) -> Tuple[float, pd.DataFrame]:
    """

    :param X:
    :param y:
    :param spec:
    :param predictor:
    :param cv:
    :param beta_grid:
    :param candidates:
    :return:
    """
    # Track the ranking scores for each beta-value
    ranking_score = np.zeros((len(beta_grid), cv.get_n_splits()))

    # Split the training data into nested training and evaluation sets for the estimation of beta
    for idx, (train, test) in enumerate(cv.split(X, y, groups=[s.get("cv_molecule_id") for s in spec])):
        LOGGER.info("Split %d/%d (n_test = %d)" % (idx + 1, cv.get_n_splits(), len(test)))

        # Load the label-spaces
        s = time.time()
        lab_spaces = []
        with candidates:
            for t in test:
                lab_spaces.append(candidates.get_labelspace(spec[t]))
        LOGGER.debug("Loading the label-spaces took: %.3fs" % (time.time() - s))

        s = time.time()
        # Get an unfitted copy of the predictor with the previously optimized hyper-parameters
        _predictor = clone(predictor)
        # Fit the estimator on the training subset
        _predictor.fit(X[train], y[train])
        LOGGER.debug("Fitting the predictor took: %.3fs" % (time.time() - s))

        # Compute the RT scores
        s = time.time()
        xlogp3_scores = []
        with candidates:
            for jdx, t in enumerate(test):
                # Get the measured retention time of the MS-feature (unknown)
                _rt_unkn = spec[t].get("retention_time")

                # Predict the XLogP3 for the unknown
                _xlogp3_unkn = _predictor.predict(np.array(_rt_unkn)[np.newaxis, np.newaxis]).item()

                # Get the XLogP3 values for all candidates
                _xlogp3_cnds = candidates.get_xlogp3_by_molecule_id(lab_spaces[jdx], missing_value="impute_mean")

                # Predict the retention time score for all candidates (scaled RT error)
                # We use the sigma (standard deviation) = 1.5 as used by Ruttkies et al. 2016
                _xlogp3_scores = norm.pdf(_xlogp3_unkn - _xlogp3_cnds, loc=0, scale=1.5)

                # We normalize the scores to be in (0, 1]
                xlogp3_scores.append(_xlogp3_scores / np.max(_xlogp3_scores))
        LOGGER.debug("Computing XLogP3 scores took: %.3fs" % (time.time() - s))

        # Load the MS scores
        s = time.time()
        ms_scores = []
        with candidates:
            for t in test:
                ms_scores.append(candidates.get_ms_scores(spec[t], args.ms2scorer, return_as_ndarray=True))
        LOGGER.debug("Loading the MS2 scores took: %.3fs" % (time.time() - s))

        # Set up the label sequences (needed for the top-1-scoring)
        sequence = LabeledSequence([spec[t] for t in test], candidates, args.ms2scorer)

        for jdx, beta in enumerate(beta_grid):
            marginals = {}
            for kdx, t in enumerate(test):
                marginals[kdx] = {
                    "n_cand": len(lab_spaces[kdx]),
                    "score": beta * ms_scores[kdx] + (1 - beta) * xlogp3_scores[kdx],
                    "label": lab_spaces[kdx],
                }

            # Compute the ranking performance (top-1 accuracy as used by Ruttkies et al. 2016)
            ranking_score[jdx, idx] = StructuredSVMSequencesFixedMS2("_", "_", "tanimoto") \
                ._topk_score(marginals, sequence, max_k=1, pad_output=True, return_percentage=True, topk_method="csi")

        LOGGER.info("Scoring the beta-grid took: %.3fs" % (time.time() - s))

    # Average the ranking scores over the samples and find the optimal beta
    opt_beta = beta_grid[np.argmax(np.mean(ranking_score, axis=1))]

    ranking_score = pd.DataFrame({
        "beta": np.repeat(beta_grid, cv.get_n_splits()),               # [1, 2, 3] --> [1, ..., 1, 2, ..., 2, 3, ..., 3]
        "spl": np.tile(np.arange(cv.get_n_splits()), len(beta_grid)),  # [1, 2, 3] --> [1, 2, 3, 1, 2, 3, ...]
        "score": ranking_score.flatten()
    })

    return opt_beta, ranking_score


def get_odir_res(eval_ds, args) -> str:
    """
    Generate the output directory given the evaluation parameters.

    :param eval_ds: string, name of the MB sub-dataset that is studied

    :param args: argparse.Namespace, command line interface arguments

    :return: string, path to the output directory
    """

    return os.path.join(
        args.output_dir, dict2fn(
            {
                "ds": eval_ds,
                "ms2scorer": args.ms2scorer,
                "mol_feat": args.molecule_features,
                "rt_predictor": args.predictor,
                "score_int_app": args.score_integration_approach,
                "mol_id": args.molecule_identifier
            },
            pref=("debug" if args.debug else None)
        )
    )


if __name__ == "__main__":
    args = get_cli_arguments()

    # ------------------------------------------------------------------------------------------------------------------
    # Set up the RT --> XLogP3 prediction model
    if args.predictor == "linear_reg":
        # Linear regression as used by Ruttkies et al. 2016
        xlogp3_predictor = LinearRegression(fit_intercept=True, normalize=False)
    else:
        raise ValueError("Invalid predictor: '%s'" % args.predictor)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Load the RT data
    eval_ds, eval_spl_idx, eval_set_id = parse_eval_set_id_information(args)
    LOGGER.info("=== Dataset ===")
    LOGGER.info("EVAL_SET_IDX = %d, DS = %s, SAMPLE_IDX = %d" % (eval_set_id, eval_ds, eval_spl_idx))

    # Setup output directory
    odir = get_odir_res(eval_ds, args)
    os.makedirs(odir, exist_ok=True)

    # Prepare DB data query statement
    stmt = "SELECT accession, %s AS molecule, retention_time, dataset, inchikey1 FROM scored_spectra_meta" \
           "   INNER JOIN datasets d ON scored_spectra_meta.dataset = d.name" \
           "   INNER JOIN molecules m ON scored_spectra_meta.molecule = m.cid" \
           "   WHERE retention_time >= 3 * column_dead_time_min" \
           "     AND column_type IS 'RP'" \
           "     AND dataset IS '%s'" % (args.molecule_identifier, eval_ds)

    # DB Connection
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)

    # Load data
    try:
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
            for spec_id, mol_id, rt, ds, ikey1 in db.execute(stmt)
        ]

        # ------------------------
        # Get test set spectra IDs
        spec_ids_eval = [
            spec_id for spec_id, in db.execute(
                "SELECT accession FROM lcms_data_splits"
                "   WHERE dataset IS ?"
                "     AND split_id IS ?"
                "     AND experiment IS 'default'",
                (eval_ds, eval_spl_idx)
            )
        ]
        if len(spec_ids_eval) == 0:
            raise ValueError(
                "No spectra for the dataset and split-id combination found: (%s, %d)" % (eval_ds, eval_spl_idx)
            )
        # ------------------------
    finally:
        db.close()

    if len(spectra) == len(spec_ids_eval):
        raise ValueError("Dataset is too small.")

    # Get the spectrum objects and their labels which are in the evaluation (test) set
    spectra_eval, labels_eval, cv_labels_eval = zip(*[
        (spectrum, spectrum.get("molecule_id"), spectrum.get("cv_molecule_id"))
        for spectrum in spectra
        if spectrum.get("spectrum_id") in spec_ids_eval
    ])

    # Get the spectrum objects and their labels for training: All spectra that are not in the evaluation set!
    spectra_train, labels_train = zip(*[
        (spectrum, spectrum.get("molecule_id"))
        for spectrum in spectra
        if spectrum.get("cv_molecule_id") not in cv_labels_eval
    ])

    LOGGER.info(
        "Number of spectra: total = %d, train = %d, evaluation = %d"
        % (len(spectra), len(spectra_train), len(spectra_eval))
    )
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Set up the candidate DB and load molecule features
    candidates = CandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, init_with_open_db_conn=False
    )

    # Load the XLogP3 values for the training and evaluation examples (~ MS features)
    with candidates:
        y_train = candidates.get_xlogp3_by_molecule_id(labels_train, missing_value="ignore")
        y_eval = candidates.get_xlogp3_by_molecule_id(labels_eval, missing_value="ignore")

    # There might be nan-values in the XLogP3 training data as PubChem doesn't provide them for each structure.
    # For the training, we simply remove these structure.
    _is_nan_train = np.isnan(y_train)
    if np.any(_is_nan_train):
        LOGGER.warning("There are nan-values in the XLogP3 training data (#=%d)." % np.sum(_is_nan_train))

        y_train = np.array([_y for _y, _is_nan in zip(y_train, _is_nan_train) if not _is_nan])
        spectra_train = [s for s, _is_nan in zip(spectra_train, _is_nan_train) if not _is_nan]

    # Load the retention times (RT) for training and evaluation
    X_train = np.array([spec.get("retention_time") for spec in spectra_train])[:, np.newaxis]
    X_eval = np.array([spec.get("retention_time") for spec in spectra_eval])[:, np.newaxis]

    # Also the XLogP3 values for the evaluation data could be NaN
    _is_nan_eval = np.isnan(y_eval)
    if np.any(_is_nan_eval):
        LOGGER.warning("There are nan-values in the XLogP3 evaluation data (#=%d)." % np.sum(_is_nan_eval))

        y_eval = np.array([_y for _y, _is_nan in zip(y_eval, _is_nan_eval) if not _is_nan])
        X_eval = X_eval[~ _is_nan_eval, :]

        assert len(y_eval) == len(X_eval)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Train the RT prediction model: RT --> XLOGP3
    xlogp3_predictor.fit(X_train, y_train)

    # Predict XLOGP3 for the evaluation set
    y_eval_pred = xlogp3_predictor.predict(X_eval)

    # Plot predicted (from RT) vs. PubChem XLogP3 values
    plt.figure()
    plt.scatter(y_eval, y_eval_pred)
    plt.title(
        "%s: MAE=%.2f, cor=%.2f, n_train=%.0f\nmolecule-feature=%s" % (
            eval_ds, mean_absolute_error(y_eval, y_eval_pred),
            pearsonr(y_eval, y_eval_pred)[0], len(spectra_train), args.molecule_features
        )
    )
    plt.xlabel("XLogP3")
    plt.ylabel("Predicted XLogP3 (based on retention time)")
    x1, x2 = plt.xlim()
    y1, y2 = plt.ylim()
    plt.plot([np.minimum(x1, y1), np.maximum(x2, y2)], [np.minimum(x1, y1), np.maximum(x2, y2)], 'k--')
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="predicted_rt_scatter", ext=ext)
            )
        )
    if not args.no_plot:
        plt.show()

    # Plot measured RTs against the PubChem XLogP3 values
    _y = np.concatenate((y_train, y_eval))  # XLogP3
    _x = np.concatenate((X_train.flatten(), X_eval.flatten()))  # RT
    plt.figure()
    plt.scatter(_x, _y)
    plt.title(
        "%s: MAE=%.2f, cor=%.2f" % (eval_ds, mean_absolute_error(_x, _y), pearsonr(_x, _y)[0])
    )
    plt.xlabel("Retention time (min)")
    plt.ylabel("PubChem XLogP3")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="rt_vs_xlogp3_scatter", ext=ext)
            )
        )
    if not args.no_plot:
        plt.show()

    # Compute different error measures on the evaluation set
    pd.DataFrame(
        {
            "dataset": eval_ds,
            "split": eval_spl_idx,
            "id": eval_set_id,
            "rt_predictor": args.predictor,
            "molecule_features": args.molecule_features,
            "n_train": len(X_train),
            "n_eval": len(X_eval),
            "mae": mean_absolute_error(y_eval, y_eval_pred),
            "mse": mean_squared_error(y_eval, y_eval_pred),
            "mape": mean_absolute_percentage_error(y_eval, y_eval_pred),
            "pearsonr": pearsonr(y_eval, y_eval_pred)[0],
            "r2": r2_score(y_eval, y_eval_pred)
        },
        index=[0]  # needs to be passes as the dataframe has only one row
    ) \
        .to_csv(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="xlogp3_prediction_performance_eval", ext="tsv")
            ),
            index=False,
            sep="\t"
        )
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Random splits used to estimate the ranking performance
    ranking_cv = GroupShuffleSplit(n_splits=args.n_splits_ranking, test_size=0.2, random_state=381281)

    # Set up the candidate DB to train the optimal beta (we use a random subset here)
    candidates_train = RandomSubsetCandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, init_with_open_db_conn=False,
        number_of_candidates=args.max_n_candidates_training, random_state=eval_set_id,
        include_correct_candidate=True
    )

    # Find the optimal MS2 score weight (beta)
    opt_beta, beta_scores = find_optimal_beta(
        X_train, y_train, spectra_train, xlogp3_predictor, ranking_cv, args.beta_grid, candidates_train
    )
    LOGGER.info("Optimal MS2 weight: %.2f" % opt_beta)

    plt.figure()
    _g = sns.pointplot(data=beta_scores, x="beta", y="score")
    _g.set_xlabel("MS2 score weight (beta)")
    _g.set_ylabel("Ranking performance (Top-1 Accuracy %)")
    _g.set_xticklabels(_g.get_xticklabels(), rotation=45)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="beta_vs_ranking_score", ext=ext)
            )
        )

    if not args.no_plot:
        plt.show()

    beta_scores \
        .assign(dataset=eval_ds, split=eval_spl_idx, id=eval_set_id) \
        .to_csv(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="beta_grid", ext="tsv")
            ),
            sep="\t",
            index=False
        )
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    d_scores__msplusrt = {}
    d_scores__onlyms = {}

    LOGGER.info("Re-score candidates:")
    for s, spec in enumerate(spectra_eval):
        # Get the ground-truth label
        correct_label = spec.get("molecule_id")

        # Get all potential molecular structures
        with candidates:
            labelspace = candidates.get_labelspace(spec, return_inchikeys=True)

        ikeys = labelspace["inchikey"]
        ikeys1 = labelspace["inchikey1"]
        labelspace = labelspace["molecule_identifier"]

        LOGGER.info("spectrum = %s" % spec.get("spectrum_id"))

        # Get the MS2 scores scaled to (0, 1]
        with candidates:
            ms2scores = candidates.get_ms_scores(
                spec, scale_scores_to_range=True, ms_scorer=args.ms2scorer, return_as_ndarray=True
            )

        # Collect all information for the top-k accuracy calculation (Only MS)
        d_scores__onlyms[s] = {
            "n_cand": len(labelspace),
            "score": ms2scores,
            "label": labelspace,
            "inchikey": ikeys,
            "inchikey1": ikeys1,
            "index_of_correct_structure": labelspace.index(correct_label)
        }

        # Predict XLogP3 value for the unknown structure associated with the MS feature
        _xlogp3_unkn = xlogp3_predictor.predict(np.array(spec.get("retention_time"))[np.newaxis, np.newaxis]).item()

        # Load the candidates' XLogP3 values. If a candidate does not have an XLogP3 value, than we impute the mean
        # XLogP3 value that is computed from the respective candidate set.
        try:
            with candidates:
                _xlogp3_cnds = candidates.get_xlogp3_by_molecule_id(labelspace, missing_value="impute_mean")
        except ImputationError as e:
            LOGGER.error(e.args[0])
            LOGGER.error(labelspace)
            LOGGER.error("Number of candidates: %d" % len(labelspace))
            LOGGER.error("We set the candidates XLogP3 values to the predicted one of the unknown --> no added information.")
            _xlogp3_cnds = np.full_like(labelspace, fill_value=_xlogp3_unkn)

        assert len(_xlogp3_cnds) == len(labelspace)
        assert len(ms2scores) == len(labelspace)

        # Compute the XLogP3 score for all candidates
        xlogp3_scores = norm.pdf(_xlogp3_cnds - _xlogp3_unkn, loc=0, scale=1.5)

        # We normalize the scores to be in (0, 1]
        xlogp3_scores /= np.max(xlogp3_scores)

        # Collect all information for the top-k accuracy calculation (MS + RT)
        d_scores__msplusrt[s] = {
            "n_cand": len(labelspace),
            "score": opt_beta * ms2scores + (1 - opt_beta) * xlogp3_scores,
            "label": labelspace,
            "inchikey": ikeys,
            "inchikey1": ikeys1,
            "index_of_correct_structure": d_scores__onlyms[s]["index_of_correct_structure"]
        }

    # ---------------------------------------------------------------
    # Calculate the top-k accuracies
    for agg_key in ["inchikey", "inchikey1"]:
        df_top_k = pd.DataFrame()

        for km, (l, d) in it.product(["casmi", "csi"], [("MS + RT", d_scores__msplusrt), ("Only MS", d_scores__onlyms)]):
            df_top_k = pd.concat(
                (
                    df_top_k,
                    get_topk_score_df(
                        None, aggregate_candidates(d, aggregation_identifier=agg_key), topk_method=km, scoring_method=l
                    )
                )
            )

        df_top_k \
            .assign(top_k_acc=(df_top_k["correct_leq_k"] / df_top_k["seq_length"]) * 100) \
            .to_csv(
                os.path.join(
                    odir,
                    dict2fn({"spl": eval_spl_idx, "cand_agg_id": agg_key}, pref="top_k", ext="tsv")
                ),
                sep="\t",
                index=False
            )
    # ---------------------------------------------------------------
