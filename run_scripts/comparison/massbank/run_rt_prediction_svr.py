import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
import itertools as it
import time

from typing import Union, Callable, Tuple

from scipy.stats import uniform, pearsonr

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.fixes import loguniform
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from matchms import Spectrum

from ssvm.data_structures import CandSQLiteDB_Massbank
from ssvm.feature_utils import RemoveCorrelatedFeatures

# FIXME: Local imports --> move to separate place
from run_with_gridsearch import parse_eval_set_id_information, get_topk_score_df, dict2fn, aggregate_candidates


# ================
# Setup the Logger
LOGGER = logging.getLogger("rt_prediction_svr")
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
    arg_parser.add_argument("--n_jobs", type=int, default=1)
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
        choices=["inchikey1", "inchikey", "cid"],
        help="Identifier used to distinguish molecules"
    )
    arg_parser.add_argument(
        "--ms2scorer",
        type=str,
        default="metfrag__norm",
        choices=["sirius__norm", "metfrag__norm", "cfmid4__norm"],
    )
    arg_parser.add_argument(
        "--score_integration_approach",
        default="filtering__global",
        type=str,
        choices=["filtering__global", "filtering__local"]
    )
    arg_parser.add_argument("--no_plot", action="store_true")

    # Unused parameters (for compatibility with the slurm-scripts)
    arg_parser.add_argument("--beta_grid", nargs="+", type=float, default=None)
    arg_parser.add_argument("--n_jobs_scoring_eval", type=int, default=None)
    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=None)
    arg_parser.add_argument("--C_grid", nargs="+", type=float, default=None)
    arg_parser.add_argument("--n_tuples_sequences_ranking", type=int, default=None)
    arg_parser.add_argument("--n_splits_ranking", type=int, default=None)
    arg_parser.add_argument("--max_n_candidates_training", type=int, default=None)

    # Constant parameters
    arg_parser.add_argument(
        "--molecule_features", default="bouwmeester__smiles_iso", type=str, choices=["bouwmeester__smiles_iso"]
    )
    arg_parser.add_argument("--n_gridsearch_parameters", type=int, default=300)
    arg_parser.add_argument("--n_splits_error_estimate", default=200, type=int)
    arg_parser.add_argument("--predictor", type=str, default="svr", choices=["svr"])
    arg_parser.add_argument("--std_threshold", type=float, default=0.01)
    arg_parser.add_argument("--corr_threshold", type=float, default=0.98)

    return arg_parser.parse_args()


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Function to compute the relative RT prediction error.

    :param y_true: array-like, shape = (n_samples, ), true RTs

    :param y_pred: array-like, shape = (n_samples, ), predicted RTs

    :return: scalar, relative prediction error
    """
    epsilon = np.finfo(np.float64).eps

    return np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)


def estimate_prediction_error(
        X: np.ndarray, y: np.ndarray, labels: np.ndarray, predictor: BaseEstimator, cv: ShuffleSplit,
        error: str = "relative_error"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: array-like, shape = (n_examples, n_features), molecule feature matrix

    :param y: array-like, shape = (n_examples, ), true molecule RTs

    :param predictor: BaseEstimator, a scikit-learn estimator with all parameters set. It is used to build the RT
        prediction model.

    :param cv: ShuffleSplit, data splitter to generate the training and test set to estimate the error distributions.

    :param error: string, indicating which error measure should be computed. Choices are "relative_error" and
        "squared_error".

    :return: Tuple (
        array-like, shape = (n_samples, ), true molecule RT for each sample. An example can be sampled multiple times
        array-like, shape = (n_samples, ), predicted molecule RTs
        array-like, shape = (n_samples, ), requested error measure
    )
    """
    # Collect all ground truth and predicted RTs (examples can appear repeatedly in the data)
    y_gt = []
    y_pred = []

    for idx, (train, test) in enumerate(cv.split(X, y, labels)):
        LOGGER.debug("Split %d/%d" % (idx + 1, cv.get_n_splits()))

        # Get an unfitted copy of the predictor with the previously optimized hyper-parameters
        s = time.time()
        _predictor = clone(predictor)
        LOGGER.debug("Cloning the predictor took: %.3fs" % (time.time() - s))

        # Fit the estimator on the training subset
        s = time.time()
        _predictor.fit(X[train], y[train])
        LOGGER.debug("Fitting the predictor took: %.3fs" % (time.time() - s))

        # Track the true and predicted RT for the test set
        y_gt.append(y[test])
        s = time.time()
        y_pred.append(_predictor.predict(X[test]))
        LOGGER.debug("Cloning the predictor took: %.3fs" % (time.time() - s))

    y_gt = np.concatenate(y_gt)
    y_pred = np.concatenate(y_pred)

    if error == "relative_error":
        # Compute the relative error for each sample point
        err = relative_error(y_gt, y_pred)
    elif error == "squared_error":
        err = (y_gt - y_pred) ** 2
    else:
        raise ValueError("Invalid error requested: '%s'" % error)

    return y_gt, y_pred, err


def get_rt_filtering_threshold(y, err, localized: bool = False, q: float = 0.95, m: int = 3, n_jobs: int = 1) \
        -> Union[Callable[[float], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    :param y: array-like, shape = (n_samples, )

    :param err: array-like, shape = (n_samples, )

    :param localized: boolean, indicating whether a local or global RT filtering threshold should be returned.

    :param q: scalar, quantile to determine the error threshold

    :param m: scalar, number of neighbours for the error sample fusion

    :return:
    """
    if localized:
        # Get the unique RTs used in the test sets
        y_unq = np.unique(y)
        n = len(y_unq)

        # Find the nearest RTs for each unique RT
        nn = NearestNeighbors(
            n_neighbors=m, n_jobs=n_jobs, algorithm="brute", p=1
        ) \
            .fit(y_unq[:, np.newaxis]) \
            .kneighbors(y_unq[:, np.newaxis], return_distance=False)

        # Fuse the relative errors of the nearest samples and compute the error quantiles
        loc_err = np.full(n, fill_value=np.nan)
        for i in range(n):
            loc_err[i] = np.quantile(
                np.concatenate(
                    [err[y == y_unq[nn_idx]] for nn_idx in nn[i]]
                ),
                q=q
            )

        return lambda t: np.interp(t, y_unq, loc_err)
    else:
        return lambda t: np.full_like(t, fill_value=np.quantile(err, q=q))


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
    # Set up the feature processing and prediction pipeline
    feature_pipeline = Pipeline([
        ("feature_removal_low_variance_features", VarianceThreshold(threshold=(args.std_threshold ** 2))),
        ("feature_removal_correlated_features", RemoveCorrelatedFeatures(corr_threshold=args.corr_threshold)),
        ("feature_scaling", StandardScaler())
    ])

    if args.predictor == "svr":
        # Support Vector Regression (SVR)
        model = SVR(kernel="rbf", tol=1e-8, max_iter=1e8)

        # Set up the model parameters
        model_parameters = {
            "svr__C": uniform(0.01, 300.0),
            "svr__epsilon": uniform(0.01, 10.0),
            "svr__gamma": loguniform(0.001, 1)
        }
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
    spectra_train, labels_train, cv_labels_train = zip(*[
        (spectrum, spectrum.get("molecule_id"), spectrum.get("cv_molecule_id"))
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

    with candidates:
        X_train = candidates.get_molecule_features_by_molecule_id(labels_train, args.molecule_features)
        X_eval = candidates.get_molecule_features_by_molecule_id(labels_eval, args.molecule_features)

    y_train = np.array([spec.get("retention_time") for spec in spectra_train])
    y_eval = np.array([spec.get("retention_time") for spec in spectra_eval])
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Train the RT prediction model
    pipeline = Pipeline([("feature_processing", feature_pipeline), (args.predictor, model)])
    predictor = RandomizedSearchCV(
        pipeline, model_parameters, cv=GroupShuffleSplit(n_splits=25, test_size=0.25, random_state=eval_set_id),
        scoring="neg_mean_absolute_error", n_jobs=args.n_jobs, random_state=eval_set_id,
        n_iter=args.n_gridsearch_parameters
    ).fit(X_train, y_train, groups=cv_labels_train)

    # Track the best parameters
    LOGGER.info("Best parameters:")
    for k, v in predictor.best_params_.items():
        LOGGER.info("\t{}: {}".format(k, v))

    # Track parameter performance
    df_params = {k: [] for k in predictor.cv_results_["params"][0]}
    df_params["score"] = []
    df_params["is_best"] = []
    df_params["dataset"] = eval_ds
    df_params["split"] = eval_spl_idx
    df_params["id"] = eval_set_id

    for i, d in enumerate(predictor.cv_results_["params"]):
        for k, v in d.items():
            df_params[k].append(v)

        df_params["score"].append(predictor.cv_results_["mean_test_score"][i])
        df_params["is_best"].append(predictor.best_index_ == i)

    pd.DataFrame(df_params).to_csv(
        os.path.join(odir, dict2fn({"spl": eval_spl_idx}, pref="grid_search_results", ext="tsv")),
        sep="\t",
        index=False
    )

    # Extract the best estimator
    predictor = predictor.best_estimator_

    # Predict RTs for the evaluation set
    y_eval_pred = predictor.predict(X_eval)

    plt.figure()
    plt.scatter(y_eval, y_eval_pred)
    plt.title(
        "%s: MAE=%.2f, rel-err=%.2f%%, cor=%.2f, n_train=%.0f\nmolecule-feature=%s" % (
            eval_ds, mean_absolute_error(y_eval, y_eval_pred),
            100 * mean_absolute_percentage_error(y_eval, y_eval_pred),
            pearsonr(y_eval, y_eval_pred)[0], len(spectra_train), args.molecule_features
        )
    )
    plt.xlabel("Retention time")
    plt.ylabel("Predicted retention time")
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
                dict2fn({"spl": eval_spl_idx}, pref="rt_prediction_performance_eval", ext="tsv")
            ),
            index=False,
            sep="\t"
        )
    # ------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Random splits used to estimate the error
    error_cv = GroupShuffleSplit(
        n_splits=args.n_splits_error_estimate, test_size=0.25, random_state=(eval_set_id + 38563)
    )

    # Estimate the RT prediction error
    y_gt_spl, _, err = estimate_prediction_error(
        X_train, y_train, np.array(cv_labels_train), predictor, error_cv, error="relative_error"
    )

    # Estimate the filtering threshold
    gamma_fun = get_rt_filtering_threshold(
        y_gt_spl, err, localized=(args.score_integration_approach == "filtering__local"), n_jobs=args.n_jobs
    )

    # Plot the threshold function
    plt.figure()
    _x = np.linspace(np.min(y_train), np.max(y_train), num=100)
    plt.plot(_x, gamma_fun(_x), label="Filtering threshold")
    plt.scatter(y_gt_spl, err, marker=".", color="black", label="Estimated errors")
    plt.title(
        "Estimated errors and thresholds\nDS=%s, n_train=%d, n_test=%d"
        % (eval_ds, len(X_train) * 0.8, len(X_train) * 0.2)
    )
    plt.xlabel("Retention time")
    plt.ylabel("Relative error")
    plt.legend()
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="rt_error_threshold", ext=ext)
            )
        )

    if not args.no_plot:
        plt.show()
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Fit the prediction model using all training data
    predictor = clone(predictor).fit(X_train, y_train)  # type: Pipeline
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    LOGGER.info("Filter candidates:")

    d_scores__msplusrt = {}
    d_scores__onlyms = {}
    df_cand_set_info = []

    for s, spec in enumerate(spectra_eval):
        LOGGER.info("spectrum = %s" % spec.get("spectrum_id"))

        # Get the ground-truth label
        correct_label = spec.get("molecule_id")

        # Get all potential molecular structures (the labelspace)
        with candidates:
            labelspace = candidates.get_labelspace(spec, return_inchikeys=True)

        ikeys = labelspace["inchikey"]
        ikeys1 = labelspace["inchikey1"]
        labelspace = labelspace["molecule_identifier"]

        # Get the MS2 scores
        with candidates:
            ms2scores = candidates.get_ms_scores(spec, ms_scorer=args.ms2scorer, return_as_ndarray=True)

        # Collect all information for the top-k accuracy calculation (Only MS)
        d_scores__onlyms[s] = {
            "n_cand": len(labelspace),
            "score": ms2scores,
            "label": labelspace,
            "inchikey": ikeys,
            "inchikey1": ikeys1,
            "index_of_correct_structure": labelspace.index(correct_label)
        }
        assert d_scores__onlyms[s]["inchikey1"][d_scores__onlyms[s]["index_of_correct_structure"]] \
               == spec.get("cv_molecule_id")

        # Predict the retention times for all candidates
        with candidates:
            X_cnd = candidates.get_molecule_features(spec, features=args.molecule_features)

        rt_cand_pred = predictor.predict(X_cnd)

        assert len(X_cnd) == len(labelspace)
        assert len(ms2scores) == len(labelspace)

        # Get the retention time (RT)
        rt = spec.get("retention_time")

        # Get the RT threshold
        gamma = gamma_fun(rt).item()
        LOGGER.info("\tmeasured RT = %f, gamma = %f" % (rt, gamma))

        # Compute the RT error
        rel_err = relative_error(np.full_like(rt_cand_pred, fill_value=rt), rt_cand_pred)
        assert len(rel_err) == len(labelspace)

        # Check which candidates to keep
        keep = (rel_err <= gamma)
        LOGGER.info(
            "\tNumber of candidates: %d (before filtering) => %d (after filtering)"
            % (len(keep), np.sum(keep).item())
        )

        # Track some candidate set statistics
        n_cand_before = len(keep)
        n_cand_after = np.sum(keep).item()
        cand_reduc_perc = 100 - (100 / n_cand_before * n_cand_after)

        _cand_set_info = [
            spec.get("spectrum_id"),                                    # Spectrum id
            eval_ds,                                                    # Dataset
            spec.get("retention_time"),                                 # Retention time
            len(keep),                                                  # Total number of candidates
            n_cand_after,                                               # Number of candidates after the filtering
            cand_reduc_perc,                                            # Percentage of candidate set size reduction
            gamma,                                                      # RT error filtering threshold
            np.round(rel_err, 4).tolist(),                              # relative errors
            d_scores__onlyms[s]["index_of_correct_structure"],          # Index of the correct molecular structure
            rel_err[d_scores__onlyms[s]["index_of_correct_structure"]]  # RT error of the correct structure
        ]

        # Filter the candidates
        labelspace = [l for i, l in enumerate(labelspace) if keep[i]]
        ikeys = [k for i, k in enumerate(ikeys) if keep[i]]
        ikeys1 = [k for i, k in enumerate(ikeys1) if keep[i]]
        ms2scores = [s for i, s in enumerate(ms2scores) if keep[i]]

        assert np.sum(keep) == len(labelspace)
        assert np.sum(keep) == len(ikeys)
        assert np.sum(keep) == len(ikeys1)
        assert np.sum(keep) == len(ms2scores)

        # Is the correct candidate still in the candidate set?
        try:
            index_of_correct_structure = labelspace.index(correct_label)
            _cand_set_info.append(True)
        except ValueError:
            index_of_correct_structure = np.nan
            _cand_set_info.append(False)

            LOGGER.info("\t! Correct candidates has been filtered out. !")

        # Collect all information for the top-k accuracy calculation (MS + RT)
        d_scores__msplusrt[s] = {
            "n_cand": len(labelspace),
            "score": ms2scores,
            "label": labelspace,
            "inchikey": ikeys,
            "inchikey1": ikeys1,
            "index_of_correct_structure": index_of_correct_structure
        }

        df_cand_set_info.append(_cand_set_info)

    df_cand_set_info = pd.DataFrame(
        df_cand_set_info,
        columns=[
            "spectrum",
            "dataset",
            "rt",
            "n_cand",
            "n_cand_after_filtering",
            "cand_reduc_perc",
            "filtering_threshold",
            "relative_errors",
            "index_of_correct_structure",
            "rel_err_of_correct_structure",
            "correct_structure_remains_after_filtering"
        ]
    )

    # False-negative rate: Correct structure was filtered out.
    fn_rate = len(df_cand_set_info) - np.sum(df_cand_set_info["correct_structure_remains_after_filtering"])
    fn_rate *= (100 / len(df_cand_set_info))
    df_cand_set_info["false_negative_rate_in_perc"] = fn_rate

    df_cand_set_info.to_csv(
        os.path.join(odir, dict2fn({"spl": eval_spl_idx}, pref="cand_set_info", ext="tsv")), sep="\t", index=False
    )

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
