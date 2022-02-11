import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import os

from typing import Union, Optional, List
from scipy.stats import uniform, pearsonr, randint

from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import loguniform
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from rdkit.Chem import Descriptors, MolFromSmiles

from ssvm.kernel_utils import generalized_tanimoto_kernel as minmax_kernel
from ssvm.data_structures import CandSQLiteDB_Massbank


# Set of descriptors used by Bouwmeester et al. (2019), before feature selection
BOUWMEESTER_DESCRIPTOR_SET = frozenset({
    "fr_C_O_noCOO", "PEOE_VSA3", "Chi4v", "fr_Ar_COO", "fr_SH", "Chi4n", "SMR_VSA10", "fr_para_hydroxylation",
    "fr_barbitur", "fr_Ar_NH", "fr_halogen", "fr_dihydropyridine", "fr_priamide", "SlogP_VSA4", "fr_guanido",
    "MinPartialCharge", "fr_furan", "fr_morpholine", "fr_nitroso", "NumAromaticCarbocycles", "fr_COO2", "fr_amidine",
    "SMR_VSA7", "fr_benzodiazepine", "ExactMolWt", "fr_Imine", "MolWt", "fr_hdrzine", "fr_urea", "NumAromaticRings",
    "fr_quatN", "NumSaturatedHeterocycles", "NumAliphaticHeterocycles", "fr_benzene", "fr_phos_acid", "fr_sulfone",
    "VSA_EState10", "fr_aniline", "fr_N_O", "fr_sulfonamd", "fr_thiazole", "TPSA", "EState_VSA8", "PEOE_VSA14",
    "PEOE_VSA13", "PEOE_VSA12", "PEOE_VSA11", "PEOE_VSA10", "BalabanJ", "fr_lactone", "fr_Al_COO", "EState_VSA10",
    "EState_VSA11", "HeavyAtomMolWt", "fr_nitro_arom", "Chi0", "Chi1", "NumAliphaticRings", "MolLogP", "fr_nitro",
    "fr_Al_OH", "fr_azo", "NumAliphaticCarbocycles", "fr_C_O", "fr_ether", "fr_phenol_noOrthoHbond", "fr_alkyl_halide",
    "NumValenceElectrons", "fr_aryl_methyl", "fr_Ndealkylation2", "MinEStateIndex", "fr_term_acetylene",
    "HallKierAlpha", "fr_C_S", "fr_thiocyan", "fr_ketone_Topliss", "VSA_EState4", "Ipc", "VSA_EState6", "VSA_EState7",
    "VSA_EState1", "VSA_EState2", "VSA_EState3", "fr_HOCCN", "fr_phos_ester", "BertzCT", "SlogP_VSA12", "EState_VSA9",
    "SlogP_VSA10", "SlogP_VSA11", "fr_COO", "NHOHCount", "fr_unbrch_alkane", "NumSaturatedRings", "MaxPartialCharge",
    "fr_methoxy", "fr_thiophene", "SlogP_VSA8", "SlogP_VSA9", "MinAbsPartialCharge", "SlogP_VSA5", "SlogP_VSA6",
    "SlogP_VSA7", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "NumRadicalElectrons", "fr_NH2", "fr_piperzine",
    "fr_nitrile", "NumHeteroatoms", "fr_NH1", "fr_NH0", "MaxAbsEStateIndex", "LabuteASA", "fr_amide", "Chi3n",
    "fr_imidazole", "SMR_VSA3", "SMR_VSA2", "SMR_VSA1", "Chi3v", "SMR_VSA6", "Kappa3", "Kappa2", "EState_VSA6",
    "EState_VSA7", "SMR_VSA9", "EState_VSA5", "EState_VSA2", "EState_VSA3", "fr_Ndealkylation1", "EState_VSA1",
    "fr_ketone", "SMR_VSA5", "MinAbsEStateIndex", "fr_diazo", "SMR_VSA4", "fr_Ar_N", "fr_Nhpyrrole", "fr_ester",
    "VSA_EState5", "EState_VSA4", "NumHDonors", "fr_prisulfonamd", "fr_oxime", "SMR_VSA8", "fr_isocyan", "Chi2n",
    "Chi2v", "HeavyAtomCount", "fr_azide", "NumHAcceptors", "fr_lactam", "fr_allylic_oxid", "VSA_EState8", "fr_oxazole",
    "VSA_EState9", "fr_piperdine", "fr_Ar_OH", "fr_sulfide", "fr_alkyl_carbamate", "NOCount", "Chi1n", "PEOE_VSA8",
    "PEOE_VSA7", "PEOE_VSA6", "PEOE_VSA5", "PEOE_VSA4", "MaxEStateIndex", "PEOE_VSA2", "PEOE_VSA1",
    "NumSaturatedCarbocycles", "fr_imide", "FractionCSP3", "Chi1v", "fr_Al_OH_noTert", "fr_epoxide", "fr_hdrzone",
    "fr_isothiocyan", "NumAromaticHeterocycles", "fr_bicyclic", "Kappa1", "Chi0n", "fr_phenol", "MolMR", "PEOE_VSA9",
    "fr_aldehyde", "fr_pyridine", "fr_tetrazole", "RingCount", "fr_nitro_arom_nonortho", "Chi0v", "fr_ArN",
    "NumRotatableBonds", "MaxAbsPartialCharge"
})


def get_cli_arguments() -> argparse.Namespace:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--target_dataset", type=str, default=None)
    arg_parser.add_argument("--n_jobs", type=int, default=1)
    arg_parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    arg_parser.add_argument(
        "--db_fn",
        type=str,
        help="Path to the MassBank database.",
        default="/home/bach/Documents/doctoral/projects/massbank2db_FILES/db/massbank__with_cfm_id.sqlite"
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
        default="inchikey1",
        choices=["inchikey1", "inchikey"],
        help="Identifier used to distinguish molecules"
    )
    arg_parser.add_argument("--do_not_load_test_splits_from_db", action="store_true")
    arg_parser.add_argument("--std_threshold", type=float, default=0.01)
    arg_parser.add_argument("--corr_threshold", type=float, default=0.98)
    arg_parser.add_argument("--n_gridsearch_parameters", type=int, default=250)
    arg_parser.add_argument("--predictor", type=str, default="svr", choices=["svr", "xgboost", "ann"])
    arg_parser.add_argument(
        "--molecule_features",
        type=str,
        default="bouwmeester_db",
        choices=[
            "bouwmeester_compute", "bouwmeester_db", "ECFP__count__all", "FCFP__count__all", "estate_idc"
        ]
    )
    arg_parser.add_argument("--use_gamma_heuristic", action="store_true")
    arg_parser.add_argument("--n_thread_xgboost", default=1, type=int)
    arg_parser.add_argument("--no_plot", action="store_true")

    return arg_parser.parse_args()


def get_outer_cv(n_examples: int) -> Union[KFold, ShuffleSplit]:
    """
    Get outer cross-validation splitter used to generate the (train, test) splits.

    :param n_examples: scalar, number of examples in the dataset which should be evaluated.

    :return: BaseCrossValidator, scikit-learn cross-validation splitter
    """
    if n_examples <= 75:
        cv = None
    elif n_examples <= 250:
        cv = GroupShuffleSplit(n_splits=15, test_size=50, random_state=n_examples)
    else:
        cv = GroupKFold(n_splits=n_examples // 50)

    return cv


def get_inner_cv(n_examples: int, random_state: Optional[int] = None):
    if n_examples <= 100:
        cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=random_state)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    return cv


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load retention time data

    :param args: argparse.Namespace, command line interface arguments

    :return: Pandas DataFrame, retention time data with molecule structure information
    """
    # Prepare DB data query statement

    stmt = "SELECT accession, %s AS molecule, retention_time, dataset, smiles_can AS smiles FROM scored_spectra_meta" \
           "   INNER JOIN datasets d ON scored_spectra_meta.dataset = d.name" \
           "   INNER JOIN molecules m ON scored_spectra_meta.molecule = m.cid" \
           "   WHERE retention_time >= 3 * column_dead_time_min" \
           "     AND column_type IS 'RP'" % args.molecule_identifier

    if args.target_dataset is not None:
        stmt += "\n AND dataset IS '%s'" % args.target_dataset

    # DB Connections
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)

    # Load data
    try:
        data = pd.read_sql(stmt, db)
    finally:
        db.close()

    return data


def get_test_sets(db_fn: str, ds: str, accs: List[str]) -> List[List[int]]:
    """
    Construct the test sets based on the pre-defined splits in the DB.

    :param db_fn:
    :param ds:
    :param accs:

    :return:
    """

    # DB Connections
    db = sqlite3.connect("file:" + db_fn + "?mode=ro", uri=True)

    # Determine the number of test splits for the target dataset
    n_splits = db.execute(
        "SELECT COUNT(DISTINCT split_id) FROM lcms_data_splits WHERE dataset IS ?", (ds, )
    ).fetchone()[0]

    test_sets = []
    for split_id in range(n_splits):
        # Get the accessions belonging to the particular split
        accs_split = db.execute(
            "SELECT accession FROM lcms_data_splits WHERE dataset IS ? AND split_id IS ?", (ds, split_id)
        )

        test_sets.append(
            sorted(
                [
                    accs.index(acc) for acc, in accs_split
                ]
            )
        )

    return test_sets


# ======================================================================================================
# Functions to reproduce the feature extraction, selection and scaling done by Bouwmeester et al. (2019)
def get_gamma_quantiles(X, standardize=False, qtl_list=0.5):
    """
    Calculate the 0.1th, 0.5th and 0.9th (default) distance quantiles.
    """
    if qtl_list is None:
        qtl_list = [0.1, 0.5, 0.9]

    # Standardize data: zero mean, unit variance
    if standardize:
        X = StandardScaler().fit_transform(X)

    # Calculate pairwise distances
    X_pwd = pairwise_distances(X).flatten()

    # Calculate the quantiles
    qtls = np.quantile(X_pwd, qtl_list)

    # Sigma --> Gamma
    qtls = [1 / (2 * s**2) for s in np.atleast_1d(qtls)]

    if len(qtls) == 1:
        qtls = qtls[0]

    return qtls


class BouwmeesterRDKitFeatures(TransformerMixin, BaseEstimator):
    def __init__(
            self, use_feature_subset: bool = True, bouwmeester_descriptor_set: frozenset = BOUWMEESTER_DESCRIPTOR_SET
    ):
        """

        :param use_feature_subset:
        :param bouwmeester_descriptor_set:
        """
        self.use_feature_subset = use_feature_subset
        self.bouwmeester_descriptor_set = bouwmeester_descriptor_set

    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        self.desc_functions_ = sorted(Descriptors.descList)

        if self.use_feature_subset:
            self.desc_functions_ = [
                (name, fun) for name, fun in self.desc_functions_
                if name in self.bouwmeester_descriptor_set
            ]

        return self

    def transform(self, X, y=None, **fit_params):
        """

        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        # Feature matrix
        Z = np.full((len(X), len(self.desc_functions_)), fill_value=np.nan)

        for i, smi in enumerate(X):
            # Parse SMILES to Mol-object
            if not (mol := MolFromSmiles(smi)):
                raise ValueError("Could not parse SMILES: '%s'" % smi)

            # Compute descriptors
            for d, (d_name, d_fun) in enumerate(self.desc_functions_):
                Z[i, d] = d_fun(mol)

        # Check for missing values
        if np.any(np.bitwise_or(np.isnan(Z), np.isinf(Z))):
            print("Some feature where not computed.")

            # Replace infs with nans
            Z[np.isinf(Z)] = np.nan

            # Use simple imputation to fill nan-features
            Z = SimpleImputer(copy=False).fit_transform(Z)

        return Z


class RemoveCorrelatedFeatures(SelectorMixin, BaseEstimator):
    def __init__(self, corr_threshold: float = 0.98):
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        """
        Fit the Bouwmeester feature selection based on the feature correlation
        """
        # Find highly correlated features and keep only one feature
        R = np.abs(np.corrcoef(X.T))  # Absolute correlation between features

        G = nx.from_numpy_array(R > self.corr_threshold)  # Graph connecting the highly correlated features

        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for cc in nx.connected_components(G):
            # Keep one node / feature per group of correlated features
            self.support_mask_[cc.pop()] = True

        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.support_mask_


# ======================================================================================================

if __name__ == "__main__":
    args = get_cli_arguments()

    # Set up the feature processing pipeline --> fitting happens on all example available for one dataset
    if args.molecule_features.startswith("bouwmeester"):
        feature_pipeline = Pipeline([
            ("feature_removal_low_variance_features", VarianceThreshold(threshold=(args.std_threshold ** 2))),
            ("feature_removal_correlated_features", RemoveCorrelatedFeatures(corr_threshold=args.corr_threshold)),
            ("feature_scaling", StandardScaler())
        ])
    else:
        feature_pipeline = None

    if args.predictor == "svr":
        if args.molecule_features.startswith("bouwmeester"):
            # Set up the model training pipeline
            model = SVR(kernel="rbf", tol=1e-8, max_iter=1e8)

            # Set up the model parameters
            model_parameters = {
                "svr__C": uniform(0.01, 300.0),
                "svr__epsilon": uniform(0.01, 10.0),
                "svr__gamma": loguniform(0.001, 1)
            }
        else:
            # Set up the model training pipeline
            model = SVR(kernel="precomputed", tol=1e-8, max_iter=1e8)

            # Set up the model parameters
            model_parameters = {
                "svr__C": uniform(0.01, 300.0),
                "svr__epsilon": uniform(0.01, 10.0),
            }

    elif args.predictor == "xgboost":
        # Set up the model training pipeline
        model = XGBRegressor(nthread=args.n_thread_xgboost)

        # Set up the model parameters (see: https://xgboost.readthedocs.io/en/latest/parameter.html)
        model_parameters = {
            'xgboost__n_estimators': randint(10, 150),      # Number of boosting steps
            'xgboost__max_depth': randint(1, 12),           # Maximum depth of a tree. Default = 6
            'xgboost__learning_rate': uniform(0.01, 0.35),  # Step size shrinkage used in update to prevents overfitting. Default = 0.3
            'xgboost__gamma': uniform(0.0, 10.0),           # Minimum loss reduction required
            'xgboost__reg_alpha': uniform(0.0, 10.0),       # L1 regularization term on weights. Default = 0.0
            'xgboost__reg_lambda': uniform(0.0, 10.0)       # L2 regularization term on weights. Default = 1.0
        }

    elif args.predictor == "ann":
        # Set up the model training pipeline
        model = MLPRegressor(random_state=2232, solver="adam")

        # Set up the model parameters
        model_parameters = {
            "ann__alpha": np.logspace(-6.0, 1.0),
            "ann__hidden_layer_sizes": randint(5, 100),
            "ann__max_iter": randint(75, 200),
        }

    else:
        raise ValueError("Invalid predictor: '%s'" % args.predictor)

    # ------------------------------------------------------------------------------------------------------------------
    # Load the RT data
    data = load_data(args)
    # ------------------------------------------------------------------------------------------------------------------

    for ds in data["dataset"].unique():
        print("Process dataset: '%s'" % ds)

        # Get data subset
        data_ds = data[data["dataset"] == ds]

        # Get the retention times and molecule identifier
        y = data_ds["retention_time"].values
        mols = data_ds["molecule"].values
        accs = data_ds["accession"].values

        # Load the molecule features
        if args.molecule_features == "bouwmeester_compute":
            X = BouwmeesterRDKitFeatures().fit_transform(data_ds["smiles"])
        else:
            candidates = CandSQLiteDB_Massbank(db_fn=args.db_fn, molecule_identifier=args.molecule_identifier)

            if args.molecule_features == "bouwmeester_db":
                molecule_features = "bouwmeester__smiles_can"
            else:
                molecule_features = args.molecule_features

            X = candidates.get_molecule_features_by_molecule_id(
                mols.tolist(), features=molecule_features, return_dataframe=False
            )

        # For the fingerprint features we pre-compute the kernel matrices
        if not args.molecule_features.startswith("bouwmeester"):
            X = minmax_kernel(X)
            kernel_precomputed = True
        else:
            kernel_precomputed = False

        # The RBF kernel scale can be computed using the "median-trick" (or gamma heuristic)
        if args.molecule_features.startswith("bouwmeester") and args.use_gamma_heuristic:
            raise NotImplementedError()
            model_parameters["%s__gamma" % args.rt_model] = [get_gamma_quantiles(X, standardize=True)]

        # Get the outer cross-validation splitter
        if args.do_not_load_test_splits_from_db:
            cv = get_outer_cv(len(data_ds))
            if cv is None:
                test_sets = []
            else:
                test_sets = [test for _, test in cv.split(X, groups=mols)]
        else:
            test_sets = get_test_sets(args.db_fn, ds, accs.tolist())

        n_splits = len(test_sets)
        if n_splits < 2:
            print("Dataset '%s' has not enough splits: %d" % (ds, n_splits))
            continue

        # --------------------------------------------------------------------------------------------------------------
        # Model Training, scoring and prediction
        df_stats = {
            "predictor": args.predictor,
            "dataset": ds,
            "molecule_features": args.molecule_features if not args.use_gamma_heuristic
                                                        else args.molecule_features + "__gamma_heuristic",
            "split": [],
            "test_score": [],
            "n_train": [],
            "n_test": [],
            "relative_error": [],
            "pearsonr": []
        }
        for k in model_parameters:
            df_stats[k] = []

        Y_pred = np.full((len(data_ds), n_splits), fill_value=np.nan)

        for idx, test in enumerate(test_sets):
            print("Process split: %d/%d" % (idx + 1, n_splits))
            df_stats["n_test"].append(len(test))
            df_stats["split"].append(idx)

            # Get the test set molecular descriptors
            mols_test = [mols[i] for i in test]

            # All molecules, that are not in the test set, are used for training
            train = [i for i in range(len(data_ds)) if mols[i] not in mols_test]
            assert len(set(train) & set(test)) == 0, "Training and test set overlap."
            df_stats["n_train"].append(len(train))

            # Get the inner cross-validation splitter
            cv_inner = get_inner_cv(len(train), random_state=idx)

            # Build parameter grid searcher
            X_train = X[np.ix_(train, train)] if kernel_precomputed else X[train]

            if feature_pipeline is not None:
                pipeline = Pipeline([("feature_processing", feature_pipeline), (args.predictor, model)])
            else:
                pipeline = Pipeline([(args.predictor, model)])

            rt_model = RandomizedSearchCV(
                pipeline, model_parameters, cv=cv_inner, scoring="neg_mean_absolute_error", n_jobs=args.n_jobs,
                random_state=idx, n_iter=args.n_gridsearch_parameters
            ).fit(X_train, y[train])

            # Track the best parameters
            print("Best parameters:")
            for k, v in rt_model.best_params_.items():
                df_stats[k].append(v)
                print("\t{}: {}".format(k, v))

            # Compute score on test set
            X_test = X[np.ix_(test, train)] if kernel_precomputed else X[test]
            df_stats["test_score"].append(- rt_model.score(X_test, y[test]))
            print("Test set score (MAE): %.3f" % df_stats["test_score"][-1])

            # Track predicted values for visual analysis
            Y_pred[test, idx] = rt_model.predict(X_test)

            df_stats["pearsonr"].append(pearsonr(y[test], Y_pred[test, idx])[0])
            df_stats["relative_error"].append(mean_absolute_percentage_error(y[test], Y_pred[test, idx]))
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # Inspect model performance and write out results
        mfeat = args.molecule_features if not args.use_gamma_heuristic else args.molecule_features + "__gamma_heuristic"
        odir = os.path.join(args.output_dir, ds, args.predictor, mfeat)
        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        df_stats = pd.DataFrame(df_stats)
        df_stats.to_csv(os.path.join(odir, "stats.tsv"), sep="\t", index=False)
        print(df_stats)

        # Make scatter plot
        y_pred = np.nanmean(Y_pred, axis=1)

        # Some examples might have never been used for testing
        y = y[~np.isnan(y_pred)]
        mols = mols[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]

        plt.scatter(y, y_pred)
        plt.title(
            "%s: MAE=%.2f, rel-err=%.2f%%, cor=%.2f, n_train=%.0f\nmolecule-feature=%s" % (
                ds, mean_absolute_error(y, y_pred), 100 * mean_absolute_percentage_error(y, y_pred),
                pearsonr(y, y_pred)[0], np.median(df_stats["n_train"]).item(), args.molecule_features
            )
        )
        plt.xlabel("Retention time")
        plt.ylabel("Predicted retention time")
        x1, x2 = plt.xlim()
        y1, y2 = plt.ylim()
        plt.plot([np.minimum(x1, y1), np.maximum(x2, y2)], [np.minimum(x1, y1), np.maximum(x2, y2)], 'k--')
        plt.savefig(os.path.join(odir, "scatter.png"))
        plt.savefig(os.path.join(odir, "scatter.pdf"))
        if not args.no_plot:
            plt.show()

        pd.DataFrame({"molecule": mols, "rt": y, "rt_pred": y_pred}) \
            .to_csv(os.path.join(odir, "predictions.tsv"), sep="\t", index=False)
        # --------------------------------------------------------------------------------------------------------------
