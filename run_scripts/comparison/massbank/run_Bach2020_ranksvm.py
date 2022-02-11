import sqlite3

import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import logging
import time

from joblib import delayed, Parallel
from typing import Union, Tuple, List
from scipy.stats import pearsonr, kendalltau

from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone

from matchms import Spectrum

from ssvm.data_structures import CandSQLiteDB_Massbank, LabeledSequence, RandomSubsetCandSQLiteDB_Massbank, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2
from ssvm.kernel_utils import generalized_tanimoto_kernel_FAST as minmax_kernel

from msmsrt_scorer.lib.exact_solvers import RandomTreeFactorGraph
from msmsrt_scorer.lib.data_utils import sigmoid

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC, Labels
from rosvm.ranksvm.platt_cls import PlattProbabilities

# FIXME: Local imports --> move to separate place
from run_with_gridsearch import parse_eval_set_id_information, get_topk_score_df, dict2fn, aggregate_candidates


# ================
# Setup the Logger
LOGGER = logging.getLogger("Bach et al. (2020) prediction approach")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.DEBUG)

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
    arg_parser.add_argument("--n_jobs_scoring_eval", type=int, default=1)
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
        help="Identifier used to distinguish molecules"
    )
    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=128)
    arg_parser.add_argument("--predictor", type=str, default="ranksvm", choices=["ranksvm"])
    arg_parser.add_argument(
        "--ms2scorer",
        type=str,
        default="metfrag__norm",
        choices=["sirius__norm", "metfrag__norm", "cfmid4__norm"],
    )
    arg_parser.add_argument(
        "--score_integration_approach",
        default="msms_pl_rt_score_integration",
        type=str,
        choices=["msms_pl_rt_score_integration"]
    )
    arg_parser.add_argument("--n_tuples_sequences_ranking", default=50, type=int)
    arg_parser.add_argument("--n_splits_C_grid", default=10, type=int)
    arg_parser.add_argument("--max_n_candidates_training", default=500, type=int)
    arg_parser.add_argument(
        "--beta_grid",
        nargs="+",
        type=float,
        default=np.round(np.arange(0, 1 + 0.05, 0.05), 3).tolist()
    )
    arg_parser.add_argument(
        "--C_grid",
        nargs="+",
        type=float,
        default=[1 / 32, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32]
    )
    arg_parser.add_argument(
        "--molecule_features",
        default="substructure_count__smiles_iso",
        type=str,
        choices=["substructure_count__smiles_iso"]
    )
    arg_parser.add_argument("--no_plot", action="store_true")

    return arg_parser.parse_args()


def find_optimal_beta(
        X: np.array, y: np.array, spec: List[Spectrum], eval_ds: str, predictor: BaseEstimator, beta_grid: List[float],
        candidates: Union[CandSQLiteDB_Massbank, RandomSubsetCandSQLiteDB_Massbank], n_tuples_sequences_ranking: int,
        n_trees: int = 128
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
    # Get the number of available (MS2, RT)-tuples belonging to the evaluation dataset
    n_test_eval_ds = sum([s.get("dataset") == eval_ds for s in spec])

    # Depending on the number of available target system data, we either optimize the MS2 weight based on the
    # (MS2, RT)-tuple sequences sampled only from target system data OR from all available training data.
    if n_test_eval_ds >= 50:
        LOGGER.debug("\t(MS2, RT)-tuples samples from the EVALUATION dataset.")
        seqsample = SequenceSample(
            spectra=[s for s in spec if s.get("dataset") == eval_ds],
            labels=[s.get("molecule_id") for s in spec if s.get("dataset") == eval_ds],
            candidates=candidates,
            N=n_tuples_sequences_ranking,
            L_min=30,
            sort_sequence_by_rt=True,
            random_state=93,
            ms_scorer=args.ms2scorer
        )
    else:
        LOGGER.debug("\t(MS2, RT)-tuples samples from ALL datasets.")
        seqsample = SequenceSample(
            spectra=spec,
            labels=[s.get("molecule_id") for s in spec],
            candidates=candidates,
            N=n_tuples_sequences_ranking,
            L_min=30,
            sort_sequence_by_rt=True,
            random_state=97,
            ms_scorer=args.ms2scorer
        )

    # Track the ranking scores for each beta-value
    ranking_score = np.zeros((len(beta_grid), n_tuples_sequences_ranking))

    # For each (MS2, RT)-tuple sequence we train a RankSVM. We remove the sequence labels (InChIKey1) from the training
    # dataset.
    for idx, seq in enumerate(seqsample):
        LOGGER.debug("\tProcess sequence sample: %d/%d" % (idx + 1, len(seqsample)))

        # Get all cross-validation labels (=InChIKey1) for the tuple sequence
        ikey1_seq = [s.get("cv_molecule_id") for s, _ in seq]

        # Get the training set indices
        train = [i for i in range(len(X)) if spec[i].get("cv_molecule_id") not in ikey1_seq]

        # Get an unfitted copy of the predictor with the previously optimized hyper-parameters
        s = time.time()
        _predictor = clone(predictor)
        LOGGER.debug("\tCloning the predictor took: %.3fs" % (time.time() - s))

        # Fit the estimator on the training subset
        s = time.time()
        _predictor.fit(minmax_kernel(X[train], X[train]), y[train])
        LOGGER.debug("\tFitting the predictor took: %.3fs" % (time.time() - s))

        # Load the label space and retention time information
        seq_cands = {}
        for jdx in range(len(seq)):
            with candidates:
                seq_cands[jdx] = {
                    "label": seq.get_labelspace(jdx),
                    "retention_time": seq.get_retention_time(jdx)
                }

        # Load the MS2 scores
        s = time.time()

        with candidates:
            for jdx in range(len(seq)):
                seq_cands[jdx]["score"] = seq.get_ms_scores(jdx, scale_scores_to_range=False, return_as_ndarray=True)
                seq_cands[jdx]["n_cand"] = len(seq_cands[jdx]["score"])

        # Normalize the MS scores as done by Bach et al. 2020 and add log-score
        _all_ms2_scores = np.hstack([cnd["score"] for cnd in seq_cands.values()])
        c1, c2 = candidates.get_normalization_parameters_c1_and_c2(_all_ms2_scores)
        for jdx in range(len(seq)):
            seq_cands[jdx]["log_score"] = candidates.normalize_scores(seq_cands[jdx]["score"], c1, c2)  # in (0, 1]
            seq_cands[jdx]["log_score"] = np.log(seq_cands[jdx]["log_score"])  # logarithmize

        LOGGER.debug("\t\tLoading and normalizing the MS2 scores took: %.3fs" % (time.time() - s))

        # Predict preference scores
        s = time.time()
        with candidates:
            for jdx in range(len(seq)):
                # Get the molecule features for all candidates
                X_cnd = seq.get_molecule_features_for_candidates(features=args.molecule_features, s=jdx)

                # Predict the preference score for all candidates
                seq_cands[jdx]["pref_score"] = _predictor.predict_pointwise(minmax_kernel(X_cnd, X[train]))
        LOGGER.debug("\t\tComputing the preference scores took: %.3fs" % (time.time() - s))

        # Compute the ranking performance for each beta-value
        s = time.time()
        for jdx, beta in enumerate(beta_grid):
            # Run the max-margin computation
            mmarg = Parallel(n_jobs=args.n_jobs)(
                delayed(_max_margin_wrapper)(
                    candidates=seq_cands, make_order_prob=_make_order_prob, D=(1 - beta), random_state=kdx
                )
                for kdx in range(n_trees)
            )

            # Average the marginals
            mmarg_averaged = {
                kdx: {
                    "n_cand": len(seq_cands[kdx]["label"]),
                    "label": seq_cands[kdx]["label"],
                    "score": mmarg[0][kdx] / n_trees,
                    "index_of_correct_structure": seq_cands[kdx]["label"].index(seq.get_labels(kdx))
                }
                for kdx in range(len(seq))
            }
            for kdx in range(1, n_trees):
                for ldx in range(len(seq)):
                    mmarg_averaged[ldx]["score"] += (mmarg[kdx][ldx] / n_trees)

            # Compute the ranking performance (top20AUC as used by Bach et al. 2020)
            _top_20 = StructuredSVMSequencesFixedMS2("_", "_", "tanimoto") \
                ._topk_score(mmarg_averaged, None, max_k=20, pad_output=True, return_percentage=False)
            ranking_score[jdx, idx] = np.sum(_top_20) / (20 * len(mmarg_averaged))

        LOGGER.debug("\t\tScoring the beta-grid took: %.3fs" % (time.time() - s))

    # Average the ranking scores over the samples and find the optimal beta
    opt_beta = beta_grid[np.argmax(np.mean(ranking_score, axis=1))]

    ranking_score = pd.DataFrame({
        "beta": np.repeat(beta_grid, ranking_score.shape[1]),               # [1, 2, 3] --> [1, ..., 1, 2, ..., 2, 3, ..., 3]
        "spl": np.tile(np.arange(ranking_score.shape[1]), len(beta_grid)),  # [1, 2, 3] --> [1, 2, 3, 1, 2, 3, ...]
        "score": ranking_score.flatten()
    })

    return opt_beta, ranking_score


def _max_margin_wrapper(candidates, make_order_prob, D, random_state):
    """
    Wrapper to compute the max-marginals in parallel
    """
    return RandomTreeFactorGraph(
        candidates, make_order_probs=make_order_prob, random_state=random_state, D=D,
        remove_edges_with_zero_rt_diff=True
    ).max_product().get_max_marginals(normalize=True)


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


def estimate_sigmoid_parameters(X, y, cv_labels, cv, estimator: KernelRankSVC):
    """

    :param X:
    :param y:
    :param cv:
    :param estimator:
    :return:
    """
    # Input parameters for our platt estimation
    y_marg = []  # predicted margins
    y_sign = []  # ground truth labels = sign of the RT difference

    # Split the data --> train a model --> predict margins
    for idx, (train, test) in enumerate(cv.split(X, y, groups=cv_labels)):
        # Get a fresh estimator
        est = clone(estimator)

        # Fit the estimator
        est.fit(minmax_kernel(X[train], X[train]), y[train])

        # Predict pairwise margin (preference value difference) for all pairs
        y_test_pref = est.predict_pointwise(minmax_kernel(X[test], X[train]))

        # We need to process each dataset separately
        dss_test = np.array(y[test].get_dss())
        rts_test = np.array(y[test].get_rts())
        for ds in set(dss_test):
            # Compute the RankSVM margin values for each test pair belonging to the current dataset
            _tmp = y_test_pref[dss_test == ds, np.newaxis] - y_test_pref[np.newaxis, dss_test == ds]
            y_marg.append(_tmp[np.tril_indices_from(_tmp, k=-1)])

            # Get sign of RT difference for all pairs belonging to the current dataset
            _tmp = np.sign(rts_test[dss_test == ds, np.newaxis] - rts_test[np.newaxis, dss_test == ds])
            y_sign.append(_tmp[np.tril_indices_from(_tmp, k=-1)])

    # Concatenate all margins and rt-diff-signs
    y_marg = np.concatenate(y_marg)
    y_sign = np.concatenate(y_sign)

    # The platt estimator only takes examples with label 1 or -1. If two examples have the same RT we need to discard
    # them.
    y_marg = y_marg[y_sign != 0]
    y_sign = y_sign[y_sign != 0]
    assert len(y_marg) == len(y_sign)

    # The full dataset output quite a lot of examples, we draw a random sample with maximum 10000 examples
    rnd_idc = np.random.RandomState(21932).choice(
        np.arange(len(y_marg)), size=np.minimum(len(y_marg), 10000), replace=False
    )
    y_marg = y_marg[rnd_idc]
    y_sign = y_sign[rnd_idc]

    platt_estimator = PlattProbabilities(maxiter=200).fit(y_marg, y_sign)
    LOGGER.debug("Platt estimator: A=%f, B=%f" % (platt_estimator.A, platt_estimator.B))

    return -platt_estimator.A, -platt_estimator.B


if __name__ == "__main__":
    args = get_cli_arguments()

    # ------------------------------------------------------------------------------------------------------------------
    # Set up predictor
    if args.predictor == "ranksvm":
        # RankSVM
        model = KernelRankSVC(
            random_state=args.eval_set_id, kernel="precomputed", pair_generation="random"
        )

        # Set up the model parameters
        model_parameters = {"C": args.C_grid}
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
           "     AND column_type IS 'RP'" % args.molecule_identifier

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

    # Get the spectrum objects and their labels which are in the evaluation (test) set
    spectra_eval, labels_eval, cv_labels_eval = zip(*[
        (spectrum, spectrum.get("molecule_id"), spectrum.get("cv_molecule_id"))
        for spectrum in spectra
        if spectrum.get("spectrum_id") in spec_ids_eval
    ])

    # Get the spectrum objects and their labels for training: All molecules (InChIKey1) that are not in the evaluation set!
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

    # The RankSVM class takes an object of class Labels (essentially a list of tuples = (rt, dataset)) to represent
    # the retention times and their association with a particular dataset.
    y_train = Labels(
        [spec.get("retention_time") for spec in spectra_train],
        [spec.get("dataset") for spec in spectra_train]
    )
    y_eval = Labels([spec.get("retention_time") for spec in spectra_eval], eval_ds)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Train the RankSVM prediction model
    inner_cv = GroupShuffleSplit(
        n_splits=args.n_splits_C_grid, train_size=0.5, random_state=((eval_set_id + 1) * 2 + 10)
    )

    # 1) Build the kernel matrix between all training examples (LIMITED to tanimoto kernel)
    # 2) Run the grid-search for the regularization parameter.
    predictor = GridSearchCV(model, model_parameters, cv=inner_cv, n_jobs=args.n_jobs) \
        .fit(minmax_kernel(X_train, X_train), y_train, groups=cv_labels_train)

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

    for p, s in zip(df_params["C"], df_params["score"]):
        LOGGER.info("C = %f -- s = %f" % (p, s))

    # Extract the best estimator
    predictor = predictor.best_estimator_

    # Predict preference values for the evaluation set
    y_eval_pred = predictor.predict_pointwise(minmax_kernel(X_eval, X_train))

    plt.figure()
    plt.scatter(y_eval.get_rts(), y_eval_pred)
    plt.title(
        "%s: kendaltau=%.2f, cor=%.2f, n_train=%.0f\nmolecule-feature=%s" % (
            eval_ds, kendalltau(y_eval.get_rts(), y_eval_pred)[0], pearsonr(y_eval.get_rts(), y_eval_pred)[0],
            len(spectra_train), args.molecule_features
        )
    )
    plt.xlabel("Retention time")
    plt.ylabel("Predicted preference value")
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
            "kendalltau": kendalltau(y_eval.get_rts(), y_eval_pred)[0],
            "pearsonr": pearsonr(y_eval.get_rts(), y_eval_pred)[0],
        },
        index=[0]  # needs to be passes as the dataframe has only one row
    ) \
        .to_csv(
            os.path.join(
                odir,
                dict2fn({"spl": eval_spl_idx}, pref="ranksvm_performance", ext="tsv")
            ),
            index=False,
            sep="\t"
        )
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Estimate the Platt parameters used as -k for the sigmoid
    s = time.time()
    sigmoid_k, sigmoid_x_0 = estimate_sigmoid_parameters(X_train, y_train, cv_labels_train, inner_cv, predictor)
    LOGGER.info(
        "Sigmoid parameter (k = %.5f, x_0 = %.5f) estimation took: %.3fs" % (sigmoid_k, sigmoid_x_0, time.time() - s)
    )

    # Define the order probability (edge potential) function with its estimated parameters
    def _make_order_prob(x, loc):
        return sigmoid(x, k=sigmoid_k, x_0=sigmoid_x_0)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Set up the candidate DB to train the optimal beta (we use a random subset here)
    candidates_train = RandomSubsetCandSQLiteDB_Massbank(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, init_with_open_db_conn=False,
        number_of_candidates=args.max_n_candidates_training, random_state=eval_set_id,
        include_correct_candidate=True
    )

    # Find the optimal beta
    opt_beta, beta_scores = find_optimal_beta(
        X_train, y_train, spectra_train, eval_ds, predictor, args.beta_grid, candidates_train,
        n_tuples_sequences_ranking=args.n_tuples_sequences_ranking, n_trees=args.n_trees_for_scoring
    )
    LOGGER.info("Optimal MS2 weight: %.2f" % opt_beta)

    plt.figure()
    _g = sns.pointplot(data=beta_scores, x="beta", y="score")
    _g.set_xlabel("MS2 score weight (beta)")
    _g.set_ylabel("Ranking performance Top20AUC")

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

    # --------------------------------------------------------------------------------------------------------------
    # Fit the prediction model using all training data
    predictor = clone(predictor).fit(minmax_kernel(X_train, X_train), y_train)  # type: Pipeline
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    LOGGER.info("Re-score candidates:")
    sequence_eval = LabeledSequence(spectra_eval, candidates, ms_scorer=args.ms2scorer, sort_sequence_by_rt=True)
    d_scores__onlyms = {}
    d_scores__msplusrt = {}

    for s, (spec, correct_label) in enumerate(sequence_eval):
        # Get all potential molecular structures
        with sequence_eval.candidates:
            labelspace = sequence_eval.get_labelspace(s, return_inchikeys=True)

        ikeys = labelspace["inchikey"]
        ikeys1 = labelspace["inchikey1"]
        labelspace = labelspace["molecule_identifier"]

        # Get the MS2 scores
        with sequence_eval.candidates:
            ms2scores = sequence_eval.get_ms_scores(s, scale_scores_to_range=False, return_as_ndarray=True)

        # Collect all information for the top-k accuracy calculation (Only MS)
        d_scores__onlyms[s] = {
            "n_cand": len(labelspace),
            "score": ms2scores,
            "label": labelspace,
            "inchikey": ikeys,
            "inchikey1": ikeys1,
            "index_of_correct_structure": labelspace.index(correct_label)
        }
        assert len(ms2scores) == len(d_scores__onlyms[s]["label"])

    # Compute max-margins to integrate MS2 and RT information
    _all_ms2_scores = np.hstack([_d["score"] for _d in d_scores__onlyms.values()])
    c1, c2 = CandSQLiteDB_Massbank.get_normalization_parameters_c1_and_c2(_all_ms2_scores)
    _all_ms2_scores = None

    for s, (spec, correct_label) in enumerate(sequence_eval):
        LOGGER.info("\tpre-process spectrum = %s" % spec.get("spectrum_id"))

        # Load candidates
        with sequence_eval.candidates:
            X_cnd = sequence_eval.get_molecule_features_for_candidates(features=args.molecule_features, s=s)

        assert len(X_cnd) == len(d_scores__onlyms[s]["label"])

        # Collect information for the margin computation
        d_scores__msplusrt[s] = {
            "n_cand": d_scores__onlyms[s]["n_cand"],
            "label": d_scores__onlyms[s]["label"],
            "retention_time": spec.get("retention_time"),
            # Logarithmized MS2 scores
            "log_score": np.log(CandSQLiteDB_Massbank.normalize_scores(d_scores__onlyms[s]["score"], c1, c2)),
            # RankSVM preference scores
            "pref_score": predictor.predict_pointwise(minmax_kernel(X_cnd, X_train)),
        }

    # Run the max-margin computation
    st = time.time()

    mmarg = Parallel(n_jobs=args.n_jobs_scoring_eval)(
        delayed(_max_margin_wrapper)(
            candidates=d_scores__msplusrt, make_order_prob=_make_order_prob, D=(1 - opt_beta),
            random_state=(ldx + 1) * 202
        )
        for ldx in range(args.n_trees_for_scoring)
    )

    for s in range(len(sequence_eval)):
        # Average the marginals
        for ldx in range(0, args.n_trees_for_scoring):
            if ldx == 0:
                d_scores__msplusrt[s]["score"] = mmarg[ldx][s]
            else:
                d_scores__msplusrt[s]["score"] += mmarg[ldx][s]

        # Remove scores not needed for top-k accuracy computation
        del d_scores__msplusrt[s]["log_score"]
        del d_scores__msplusrt[s]["pref_score"]

        # Add label information for the score aggregation
        d_scores__msplusrt[s]["inchikey"] = d_scores__onlyms[s]["inchikey"]
        d_scores__msplusrt[s]["inchikey1"] = d_scores__onlyms[s]["inchikey1"]

        # Add index of correct structure
        d_scores__msplusrt[s]["index_of_correct_structure"] = d_scores__onlyms[s]["index_of_correct_structure"]

    mmarg = None

    LOGGER.debug("Margin computation and averaging took: %.3fs" % (time.time() - st))

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
