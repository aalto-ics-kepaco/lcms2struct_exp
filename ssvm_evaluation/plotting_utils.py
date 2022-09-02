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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging

from matplotlib import cm
from copy import deepcopy
from typing import List, Optional, Union

from scipy.stats import wilcoxon, ttest_rel

# ================
# Setup the Logger
LOGGER = logging.getLogger("plotting_utils")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def _get_topk(x, k, method):
    """
    Task: Pandas aggregation function to compute the top-k acc.
    """
    out = 0.0

    if method == "average":
        for xi in x:
            out += (np.mean(xi) <= k)
    elif method == "csi":
        for xi in x:
            y = np.arange(xi[0], xi[1] + 1)
            for yi in y:
                if yi <= k:
                    out += (1.0 / len(y))
    else:
        raise ValueError("Invalid method: '%s'" % method)

    # Get accuracy as percentages
    out /= len(x)
    out *= 100

    return out


def plot__02__b(
        results: pd.DataFrame, ks: Optional[Union[List[int], int]] = None, min_class_support: int = 50,
        sharey: str = "all", n_samples: int = 50, topk_method: str = "csi", fig_width: int = 6, fig_height: int = 4,
        label_rot_angle: float = 90
):
    """
    Bar plots indicating the top-k improvements per class in the PubChemLite classification based on PubChem's TOC.

    :param results: pd.DataFrame, containing the Only-MS2 and LC-MS2Struct ranks, PubChemLite classes, and further
        information for the spectra in our experiments. Please check, "gather_ranl_changes__csi.py" for the details on
        the data-structure. The results for different MS2-scorers are concatenated.

    :param ks: scalar or list of scalars, k for which the top-k ranking performance improvements should be analysed.

    :param min_class_support: scalar, minimum number of unique molecular structures per PubChemLite class.

    :param sharey: string or boolean, indicating whether (and how) the y-axes ranges are synchronized.

    :param n_samples: scalar, number of random samples to estimate the top-k accuracy from.

    :param topk_method: deprecated

    :param fig_width: scalar, width of the figure

    :param fig_height: scalar, height of the figure

    :param label_rot_angle: scalar, rotation angle of the x-labels

    :return:
    """
    assert topk_method == "csi", "Only CSI:FingerID top-k accuracy computation is supported."

    # Make a deep copy of the input data, e.g. to allow modifications
    results = deepcopy(results)

    # Get relevant subset
    pl_columns = [s for s in results.columns.tolist() if s.startswith("pubchemlite")]
    info_columns = [
        "correct_structure", "molecule_identifier", "rank_onlyms", "rank_msplrt", "n_cand", "n_isomers", "ms2scorer"
    ]
    results = results \
        .filter(items=pl_columns + info_columns, axis=1) \
        .rename(mapper={c: c.split("_")[1] for c in pl_columns}, axis=1)

    # --- Columns in the subplot ---
    # k for the top-k that are plotted as columns in the subplots
    if ks is None:
        ks = [1, 20]
    elif isinstance(ks, int):
        ks = [ks]
    else:
        assert isinstance(ks, list)

    n_k = len(ks)

    # --- Rows in the subplot correspond to the MS2 scoring methods in the input data ---
    if "ms2scorer" not in results.columns:
        results = results.assign(ms2scorer="MS$^2$ Scorer")

    l_ms2scorer = [ms2scorer for ms2scorer, _ in results.groupby("ms2scorer")]
    d_row2scorer = {s: i for i, s in enumerate(l_ms2scorer)}

    n_scorer = len(l_ms2scorer)

    # Create the Axes-array for plotting
    fig, axrr = plt.subplots(
        n_scorer, n_k, figsize=(fig_width * n_k, fig_height * n_scorer), squeeze=False, sharey=sharey, sharex="all"
    )

    # Plot
    results_out = []
    for ms2scorer, res_sub in results.groupby("ms2scorer"):
        for ax_col_idx, k in enumerate(ks):
            # Get the axis to draw in
            ax = axrr[d_row2scorer[ms2scorer], ax_col_idx]

            _res_sub = []
            for rep in range(n_samples):
                _res = res_sub \
                    .sample(frac=1, random_state=rep) \
                    .drop_duplicates("correct_structure") \
                    .melt(info_columns, var_name="pubchemlite_class", value_name="membership_count")

                # We can drop the rows where a molecule is not member of a particular class
                _res = _res[_res["membership_count"] > 0]  # type: pd.DataFrame

                # Compute the top-k accuracies for Only MS and MS + RT
                _res = _res \
                    .groupby("pubchemlite_class") \
                    .agg({
                        "rank_onlyms": lambda x: _get_topk(x, k, topk_method),
                        "rank_msplrt": lambda x: _get_topk(x, k, topk_method),
                        "n_cand": np.median,
                        "n_isomers": np.median,
                        "molecule_identifier": len
                    }) \
                    .rename({
                        "rank_onlyms": "top_k_p_onlyms",
                        "rank_msplrt": "top_k_p_msplrt",
                        "molecule_identifier": "n_class_support"
                    }, axis=1) \
                    .reset_index()

                _res_sub.append(_res)

            _res_sub = pd.concat(_res_sub, ignore_index=True)

            # Add the top-k improvement in percentage-points
            _res_sub = _res_sub.assign(top_k_p_improvement=(_res_sub["top_k_p_msplrt"] - _res_sub["top_k_p_onlyms"]))

            # Filter classes without enough support
            _res_sub = _res_sub[_res_sub["n_class_support"] >= min_class_support]
            if len(_res_sub) == 0:
                raise ValueError("No class has enough support.")

            sns.barplot(
                data=_res_sub, x="pubchemlite_class", y="top_k_p_improvement", ax=ax
            )
            ax.grid(axis="y")
            ax.hlines(0, ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1, color='k', linestyle="--")
            ax.set_title("%s - top-%d" % (ms2scorer, k), fontweight="bold")
            ax.bar_label(
                ax.containers[0],
                labels=[
                    "%.1f" % _l
                    for _l in _res_sub.groupby("pubchemlite_class")["top_k_p_onlyms"].mean().tolist()
                ],
                rotation=90, horizontalalignment="center", fmt="%.1f", label_type="edge", padding=10, fontsize=12
            )

            if d_row2scorer[ms2scorer] == (n_scorer - 1):
                ax.set_xticklabels(
                    [
                        plt.Text(
                            _tl.get_position()[0], _tl.get_position()[1],
                            "%s (n=%d)" %
                            (
                                _tl.get_text(),
                                _res_sub[_res_sub["pubchemlite_class"] == _tl.get_text()]["n_class_support"].iloc[0]
                            )
                        )
                        for _tl in ax.get_xticklabels()
                    ],
                    rotation=label_rot_angle, horizontalalignment="center", fontsize=12
                )
                ax.set_xlabel("PubChemLite classification", fontsize=12)
            else:
                ax.set_xlabel("")

            if ax_col_idx == 0:
                ax.set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)
            else:
                ax.set_ylabel("")

            results_out.append(
                _res_sub
                    .groupby("pubchemlite_class")
                    .agg({
                        "top_k_p_onlyms": np.mean,
                        "top_k_p_msplrt": np.mean,
                        "top_k_p_improvement": np.mean,
                        "n_cand": lambda x: x.iloc[0],
                        "n_isomers": lambda x: x.iloc[0],
                        "n_class_support": lambda x: x.iloc[0],
                     })
                    .assign(k=k, ms2scorer=ms2scorer)
                    .reset_index()
            )

            # Compute the average improvement into actual counts
            results_out[-1]["improvement_in_n"] = \
                (results_out[-1]["n_class_support"] * results_out[-1]["top_k_p_improvement"]) / 100

    # Adjust y-axis range to provide enough space for the labels
    _y_add = {1: 1.0, 5: 0.5, 20: 1.75}
    for ax_col_idx, _k in enumerate(ks):
        for ax in axrr[:, ax_col_idx]:
            _y_min, _y_max = ax.get_ylim()
            ax.set_ylim(_y_min - _y_add.get(_k, 0.0), _y_max)

    plt.tight_layout()

    return pd.concat(results_out, ignore_index=True)


# Same color-map as used in the supplementary material when plotting the classyfire class distribution
MY_CLASSYFIRE_CLASSES_COLORMAP = {
    'Alkaloids and derivatives': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    'Benzenoids': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
    'Lignans, neolignans and related compounds': (0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0),
    'Lipids and lipid-like molecules': (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    'Nucleosides, nucleotides, and analogues': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    'Organic acids and derivatives': (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
    'Organic nitrogen compounds': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0),
    'Organic oxygen compounds': (1.0, 0.596078431372549, 0.5882352941176471, 1.0),
    'Organohalogen compounds': (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    'Organoheterocyclic compounds': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    'Other': (0.586082276047674, 0.586082276047674, 0.586082276047674, 1.0),
    'Phenylpropanoids and polyketides': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0)
}


def plot__02__a(
        results: pd.DataFrame, ks: Optional[Union[List[int], int]] = None, min_class_support: int = 50,
        colormap_name: str = "fixed", sharey: str = "all", cf_level: str = "superclass", n_samples: int = 50,
        topk_method: str = "csi", fig_width: int = 6, fig_height: int = 4, label_rot_angle: float = 90
):
    """
    Bar plots indicating the top-k improvements per ClassyFire compound class.

    :param results: pd.DataFrame, containing the Only-MS2 and LC-MS2Struct ranks, ClassyFire classes, and further
        information for the spectra in our experiments. Please check, "gather_ranl_changes__csi.py" for the details on
        the data-structure. The results for different MS2-scorers are concatenated.

    :param ks: scalar or list of scalars, k for which the top-k ranking performance improvements should be analysed.

    :param min_class_support: scalar, minimum number of unique molecular structures per ClassyFire class.

    :param colormap_name: string, either the name of a matplotlib color-map, or "fixed". If "fixed" than pre-defined
        colors are used for the ClassyFire super-classes.

    :param sharey: string or boolean, indicating whether (and how) the y-axes ranges are synchronized.

    :param cf_level: string, Classyfire level to analyse.

    :param n_samples: scalar, number of random samples to estimate the top-k accuracy from.

    :param topk_method: deprecated

    :param fig_width: scalar, width of the figure

    :param fig_height: scalar, height of the figure

    :param label_rot_angle: scalar, rotation angle of the x-labels

    :return:
    """
    def _aggregate_and_filter_classyfire_classes(df, min_class_support, cf_level):
        """
        Task: Group and aggregate the results by the ClassyFire class-level and determine the support for each class.
              Then, remove all classes with too little support. Purpose is to get the "relevant" class and superclass
              relationships to determine the colors and orders for the plotting.
        """
        # We consider only unique molecular structures to compute the CF class support
        tmp = df.drop_duplicates("correct_structure")

        # Group by the ClassyFire level
        tmp = tmp.groupby("classyfire_%s" % cf_level)

        if cf_level == "class":
            tmp = tmp.aggregate({
                "molecule_identifier": lambda x: len(x),
                "classyfire_superclass": lambda x: x.iloc[0]
            })
        elif cf_level == "superclass":
            tmp = tmp.aggregate({
                "molecule_identifier": lambda x: len(x),
                "classyfire_class": lambda x: ",".join([xi for xi in x if not pd.isna(xi)])
            })
        else:
            raise ValueError("Invalid ClassyFire level: '%s'" % cf_level)

        tmp = tmp \
            .rename({"molecule_identifier": "n_class_support"}, axis=1) \
            .reset_index() \
            .sort_values(by="classyfire_superclass")

        return tmp[tmp["n_class_support"] >= min_class_support]

    assert cf_level in ["superclass", "class"], "Invalid or unsupported ClassyFire class level: '%s'." % cf_level
    assert topk_method == "csi", "Only CSI:FingerID top-k accuracy computation is supported."

    # Make a deep copy of the input data, e.g. to allow modifications
    results = deepcopy(results)

    # Drop the rows for which the desired ClassyFire class has no value (NaN), e.g. some examples might not have a
    # 'class'-level annotation.
    results = results.dropna(subset=["classyfire_%s" % cf_level])

    # --- Columns in the subplot ---
    # k for the top-k that are plotted as columns in the subplots
    if ks is None:
        ks = [1, 20]
    elif isinstance(ks, int):
        ks = [ks]
    else:
        assert isinstance(ks, list)

    n_k = len(ks)

    # --- Rows in the subplot correspond to the MS2 scoring methods in the input data ---
    if "ms2scorer" not in results.columns:
        results = results.assign(ms2scorer="MS2 Scorer")

    l_ms2scorer = [ms2scorer for ms2scorer, _ in results.groupby("ms2scorer")]
    d_row2scorer = {s: i for i, s in enumerate(l_ms2scorer)}

    n_scorer = len(l_ms2scorer)

    # Create the Axes-array for plotting
    fig, axrr = plt.subplots(
        n_scorer, n_k, figsize=(fig_width * n_k, fig_height * n_scorer), squeeze=False, sharey=sharey, sharex="all"
    )

    # Get class-level colors based on superclass-level
    cf_cls_stats = _aggregate_and_filter_classyfire_classes(results, min_class_support, cf_level)

    LOGGER.debug(
        "n_superclass = %d, n_class = %d" %
        (cf_cls_stats["classyfire_superclass"].nunique(), cf_cls_stats["classyfire_class"].nunique())
    )

    superlevel = {}
    palette = {}
    order = []

    if cf_level == "class":
        for idx, (cf_sc, tmp) in enumerate(cf_cls_stats.groupby("classyfire_superclass")):
            for cf_c in sorted(tmp["classyfire_class"].unique()):
                if colormap_name == "fixed":
                    palette[cf_c] = MY_CLASSYFIRE_CLASSES_COLORMAP[cf_sc]
                else:
                    palette[cf_c] = cm.get_cmap(colormap_name)(idx)

                order.append(cf_c)
                superlevel[cf_c] = cf_sc
    elif cf_level == "superclass":
        for idx, (cf_sc, _) in enumerate(cf_cls_stats.groupby("classyfire_superclass")):
            if colormap_name == "fixed":
                palette[cf_sc] = MY_CLASSYFIRE_CLASSES_COLORMAP[cf_sc]
            else:
                palette[cf_sc] = cm.get_cmap(colormap_name)(idx)

            order.append(cf_sc)
    else:
        raise ValueError("Invalid ClassyFire level: '%s'" % cf_level)

    # Plot
    results_out = []
    for ms2scorer, res_sub in results.groupby("ms2scorer"):
        for ax_col_idx, k in enumerate(ks):
            # Get the axis to draw in
            ax = axrr[d_row2scorer[ms2scorer], ax_col_idx]

            # Compute the top-k accuracies for Only MS and MS + RT
            _res_sub = []
            for rep in range(n_samples):
                _res_sub.append(
                    res_sub
                        .sample(frac=1, random_state=rep)
                        .drop_duplicates("correct_structure")
                        .groupby("classyfire_%s" % cf_level)
                        .agg({
                            "rank_onlyms": lambda x: _get_topk(x, k, topk_method),
                            "rank_msplrt": lambda x: _get_topk(x, k, topk_method),
                            "n_cand": np.median,
                            "n_isomers": lambda x: "min=%d, max=%d, avg=%.1f, med=%.1f" % (
                                np.min(x), np.max(x), np.mean(x), np.median(x)
                            ),
                            "molecule_identifier": len
                         })
                        .rename({
                            "rank_onlyms": "top_k_p_onlyms",
                            "rank_msplrt": "top_k_p_msplrt",
                            "molecule_identifier": "n_class_support"
                         }, axis=1)
                        .reset_index()
                )
            _res_sub = pd.concat(_res_sub, ignore_index=True)

            # Add the top-k improvement in percentage-points
            _res_sub = _res_sub.assign(top_k_p_improvement=(_res_sub["top_k_p_msplrt"] - _res_sub["top_k_p_onlyms"]))

            # Filter classes without enough support
            _res_sub = _res_sub[_res_sub["n_class_support"] >= min_class_support]
            if len(_res_sub) == 0:
                raise ValueError("No class has enough support.")

            ax = sns.barplot(
                data=_res_sub, x="classyfire_%s" % cf_level, y="top_k_p_improvement", ax=ax, palette=palette,
                order=order, seed=1020
            )
            ax.grid(axis="y")
            ax.hlines(0, ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1, color='k', linestyle="--")
            ax.set_title("%s - top-%d" % (ms2scorer, k), fontweight="bold")
            ax.bar_label(
                ax.containers[0],
                labels=[
                    "%.1f" % _l
                    for _l in _res_sub.groupby("classyfire_%s" % cf_level)["top_k_p_onlyms"].mean().tolist()
                ],
                rotation=90, horizontalalignment="center", fmt="%.1f", label_type="edge", padding=10, fontsize=12
            )

            if d_row2scorer[ms2scorer] == (n_scorer - 1):
                ax.set_xticklabels(
                    [
                        plt.Text(
                            _tl.get_position()[0], _tl.get_position()[1],
                            "%s (n=%d)" %
                            (
                                _tl.get_text(),
                                _res_sub[_res_sub["classyfire_%s" % cf_level] == _tl.get_text()]["n_class_support"].iloc[0]
                            )
                        )
                        for _tl in ax.get_xticklabels()
                    ],
                    rotation=label_rot_angle, horizontalalignment="center", fontsize=12
                )
                ax.set_xlabel("ClassyFire: %s" % {"superclass": "Super-class", "class": "Class"}[cf_level], fontsize=12)
            else:
                ax.set_xlabel("")

            if ax_col_idx == 0:
                ax.set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)
            else:
                ax.set_ylabel("")

            results_out.append(
                _res_sub
                    .groupby("classyfire_%s" % cf_level)
                    .agg({
                        "top_k_p_onlyms": np.mean,
                        "top_k_p_msplrt": np.mean,
                        "top_k_p_improvement": np.mean,
                        "n_cand": lambda x: x.iloc[0],
                        "n_isomers": lambda x: x.iloc[0],
                        "n_class_support": lambda x: x.iloc[0],
                     })
                    .assign(k=k, ms2scorer=ms2scorer)
                    .reset_index()
            )

            # Compute the average improvement into actual counts
            results_out[-1]["improvement_in_n"] = \
                (results_out[-1]["n_class_support"] * results_out[-1]["top_k_p_improvement"]) / 100

    # Adjust y-axis range to provide enough space for the labels
    _y_add = {1: 1.25, 5: 0.9, 20: 1.5}
    for ax_col_idx, _k in enumerate(ks):
        for ax in axrr[:, ax_col_idx]:
            _y_min, _y_max = ax.get_ylim()
            ax.set_ylim(_y_min - _y_add.get(_k, 0.0), _y_max)

    plt.tight_layout()

    return pd.concat(results_out, ignore_index=True), superlevel


def _get_res_set(df: pd.DataFrame):
    return set((
        (row["eval_indx"], row["dataset"])
        for index, row in df.loc[:, ["eval_indx", "dataset"]].drop_duplicates().iterrows()
    ))


def _restrict_df(df: pd.DataFrame, res_set: set):
    if df is None:
        return None

    df_out = [row for _, row in df.iterrows() if (row["eval_indx"], row["dataset"]) in res_set]

    return pd.DataFrame(df_out)


def _process_dfs__01(res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020, raise_on_missing_results):
    n_scorer = len(res__baseline)

    res_sets = []
    for i in range(n_scorer):
        restrict_results = False

        # Only MS2
        _res_set_baseline = _get_res_set(res__baseline[i])
        res_sets.append(_res_set_baseline)

        # SSVM
        _res = _get_res_set(res__ssvm[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("SSVM has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        # RT filtering
        _res = _get_res_set(res__rtfilter[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("RT filtering has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        # XLogP3
        _res = _get_res_set(res__xlogp3[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("XLogP3 has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        # Bach et al. (2020)
        _res = _get_res_set(res__bach2020[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("Bach et al. (2020) has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        if restrict_results:
            res__baseline[i] = _restrict_df(res__baseline[i], res_sets[i])
            res__ssvm[i] = _restrict_df(res__ssvm[i], res_sets[i])
            res__rtfilter[i] = _restrict_df(res__rtfilter[i], res_sets[i])
            res__xlogp3[i] = _restrict_df(res__xlogp3[i], res_sets[i])
            res__bach2020[i] = _restrict_df(res__bach2020[i], res_sets[i])

    # Sort results so that the rows would match
    for i in range(n_scorer):
        res__baseline[i] = res__baseline[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__ssvm[i] = res__ssvm[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__rtfilter[i] = res__rtfilter[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__xlogp3[i] = res__xlogp3[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__bach2020[i] = res__bach2020[i].sort_values(by=["dataset", "eval_indx", "k"])

    return res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020


def plot__01__a(
        res__baseline:  List[pd.DataFrame],
        res__ssvm:      List[pd.DataFrame],
        res__rtfilter:  List[pd.DataFrame],
        res__xlogp3:    List[pd.DataFrame],
        res__bach2020:  List[pd.DataFrame],
        aspect: str = "landscape",
        max_k: int = 20,
        weighted_average: bool = False,
        raise_on_missing_results: bool = True,
        verbose: bool = False,
        sharey: Union[bool, str] = False
):
    """
    Plot comparing the top-k accuracy performance for k in {1, ..., max_k} of the different scoring methods:

        - baseline: Only MS2 information is used
        - ssvm: Proposed Structured Support Vector Regression (SSVM) model
        - rtfilter: Candidate filtering using retention time errors
        - xlogp3: Candidate re-ranking using predicted XLogP3 values
        - bach2020: Retention order and MS2 score integration framework by Bach et al. 2020

    The for each scoring method a list of dataframes is provided. Each DataFrame has the following structure:

    k 	top_k_method 	scoring_method 	correct_leq_k 	seq_length 	n_models 	eval_indx 	dataset 	top_k_acc 	ds 	    lloss_mode 	    mol_feat 	            mol_id 	ms2scorer 	    ssvm_flavor
    1 	csi 	        Only-MS2        3.000000 	    50 	        8 	        0 	        AC_003 	    6.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    2 	csi 	        Only-MS2        5.000000  	    50 	        8 	        0 	        AC_003 	    10.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    3 	csi 	        Only-MS2        7.000000 	    50 	        8 	        0 	        AC_003 	    14.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    4 	csi 	        Only-MS2        9.000000 	    50 	        8 	        0 	        AC_003 	    18.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    5 	csi 	        Only-MS2        11.000000 	    50 	        8 	        0 	        AC_003 	    22.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    ...

    whereby the "scoring_method" differs.

    Each list element corresponds to a different MS2 base-scorer, e.g. CFM-ID, MetFrag, ...

    :param res__baseline: list of dataframe, containing the ranking results for Only MS2.

    :param res__ssvm: list of dataframe, containing the ranking results for the SSVM approach.

    :param res__rtfilter: list of dataframe, containing the RT filtering results.

    :param res__xlogp3: list of dataframe, containing the XLogP3 re-scoring results.

    :param res__bach2020: list of dataframe, containing the results achieved by Bach et al.'s method.

    :param aspect: string, indicating which layout for the plot should be used:

        "landscape":
                        CFMID  METFRAG    SIRIUS
                         ____   ____       ____
                        |    | |    | ... |    |   Top-k
                        |____| |____|     |____|
                         ____   ____       ____
                        |    | |    | ... |    |   Top-k improvement over the baseline
                        |____| |____|     |____|

        "portrait":
                        Top-k   Top-k improvement over the baseline
                         ____   ____
                CFMID   |    | |    |
                        |____| |____|
                         ____   ____
                MEFRAG  |    | |    |
                        |____| |____|
                              .
                              .
                              .
                         ____   ____
                SIRIUS  |    | |    |
                        |____| |____|

    :param max_k: scalar, what is the maximum k value for the top-k curve plot.

    :param weighted_average: boolean, indicating whether the average the top-k accuracy should be first computed within
        each dataset and subsequently averaged across the datasets. If False, than all samples are treated equally and
        simply averaged directly across all datasets.

    :param raise_on_missing_results: boolean, indicating whether an error should be raised if results are missing. If
        False, than only those results which are available for all scoring methods of a particular MS2 base-scorer are
        considered for the plots.

    :param verbose: boolean, indicating whether all kinds of stuff should be printed, which somehow can be helpful for
        debugging.

    :return: pd.DataFrame, data shown in the plot for publication.
    """
    def _acc_info_printer(baseline, other, k):
        print(
            "\ttop-%d: baseline = %.1f%%, other = %.1f%%, improvement = %.1f%%p, gain = %.1f%%, n = %.1f" %
            (
                k,
                baseline["top_k_acc"][other["k"] == k],
                other["top_k_acc"][other["k"] == k],
                (other["top_k_acc"] - baseline["top_k_acc"])[other["k"] == k],
                ((other["top_k_acc"] / baseline["top_k_acc"])[other["k"] == k] - 1) * 100,
                (other["correct_leq_k"] - baseline["correct_leq_k"])[other["k"] == k]
            )
        )

    assert aspect in ["landscape", "portrait"], "Invalid aspect value: '%s'" % aspect

    # Number of MS2 scorers must be equal
    assert len(res__baseline) == len(res__ssvm)
    assert len(res__baseline) == len(res__rtfilter)
    assert len(res__baseline) == len(res__xlogp3)
    assert len(res__baseline) == len(res__bach2020)
    n_scorer = len(res__baseline)

    # There should be only one scoring and one top-k accuracy computation method in each dataframe
    for k in ["scoring_method", "top_k_method", "ms2scorer"]:
        for i in range(n_scorer):
            assert res__baseline[i][k].nunique() == 1
            assert res__ssvm[i][k].nunique() == 1
            assert res__rtfilter[i][k].nunique() == 1
            assert res__xlogp3[i][k].nunique() == 1
            assert res__bach2020[i][k].nunique() == 1

    # For the SSVM all results should be 8 SSVM models
    for i in range(n_scorer):
        assert np.all(res__ssvm[i]["n_models"] == 8), "There seems to be SSVM models missing."

    # Get all available results and restrict them if needed by only using the result intersection
    res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020 = _process_dfs__01(
        res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020, raise_on_missing_results
    )

    # Get a new figure
    if aspect == "portrait":
        _n_rows = n_scorer
        _n_cols = 2
        _figsize = (9, 3 * n_scorer)
    else:  # landscape
        _n_rows = 2
        _n_cols = n_scorer
        _figsize = (4.5 * n_scorer, 5.75)

    fig, axrr = plt.subplots(_n_rows, _n_cols, figsize=_figsize, sharex="all", sharey=sharey, squeeze=False)

    # Set some plot properties
    k_ticks = np.arange(0, max_k + 1, 5)
    k_ticks[0] = 1

    # For Machine Intelligence we need to provide the raw-data for the plot
    res_out = []

    # Plot Top-k curve
    if verbose:
        print("We expect 17500 result rows")

    for idx, (a, b, c, d, e) in enumerate(zip(res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020)):
        assert a["ms2scorer"].unique().item() == b["ms2scorer"].unique().item()
        assert a["ms2scorer"].unique().item() == c["ms2scorer"].unique().item()
        assert a["ms2scorer"].unique().item() == d["ms2scorer"].unique().item()
        assert a["ms2scorer"].unique().item() == e["ms2scorer"].unique().item()

        if verbose:
            print("Rows (MS2-scorer='%s'):" % a["ms2scorer"].unique().item())
            print("Number of samples: %d" % (a["k"] == 1).sum())

        # Get current axis and set labels
        if aspect == "portrait":
            # first column
            ax = axrr[idx, 0]
            ax.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            ax.set_ylabel("Top-k accuracy (%)", fontsize=12)

            # second column
            ax2 = axrr[idx, 1]
            ax2.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            ax2.set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)
        else:
            # first row
            ax = axrr[0, idx]
            ax.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            axrr[0, 0].set_ylabel("Top-k accuracy (%)", fontsize=12)

            # second row
            ax2 = axrr[1, idx]
            axrr[1, 0].set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)

        # Baseline
        if verbose:
            print("Baseline: ", len(a))

        if weighted_average:
            bl = a[a["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            bl = a[a["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(bl["k"], bl["top_k_acc"], where="post", label=a["scoring_method"].unique().item(), color="black")
        ax2.hlines(0, 1, max_k, colors="black", label=a["scoring_method"].unique().item())

        res_out += list(zip(
            bl["k"], bl["top_k_acc"], [a["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # SSVM
        if verbose:
            print("SSVM: ", len(b))

        if weighted_average:
            tmp = b[b["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            tmp = b[b["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(tmp["k"], tmp["top_k_acc"], where="post", label=b["scoring_method"].unique().item(), color="blue")

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=b["scoring_method"].unique().item(),
            color="blue"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [b["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # RT filtering
        if verbose:
            print("RT filtering: ", len(c))

        if weighted_average:
            tmp = c[c["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            tmp = c[c["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(tmp["k"], tmp["top_k_acc"], where="post", label=c["scoring_method"].unique().item(), color="red")

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=c["scoring_method"].unique().item(),
            color="red"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [c["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # XLogP3
        if verbose:
            print("XLogP3: ", len(d))

        if weighted_average:
            tmp = d[d["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            tmp = d[d["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(tmp["k"], tmp["top_k_acc"], where="post", label=d["scoring_method"].unique().item(), color="green")

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=d["scoring_method"].unique().item(),
            color="green"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [d["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # Bach et al. (2020)
        if verbose:
            print("Bach et al. (2020)", len(e))

        if weighted_average:
            tmp = e[e["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby(
                "k").mean().reset_index()
        else:
            tmp = e[e["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(tmp["k"], tmp["top_k_acc"], where="post", label=e["scoring_method"].unique().item(), color="orange")

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=e["scoring_method"].unique().item(),
            color="orange"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [e["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # Set some further axis properties
        ax.set_xticks(k_ticks)
        ax2.set_xticks(k_ticks)

        ax.grid(axis="y")
        ax2.grid(axis="y")

        if (aspect == "portrait") and (idx == (n_scorer - 1)):
            ax.set_xlabel("k")
            ax2.set_xlabel("k")
        elif aspect == "landscape":
            ax2.set_xlabel("k")

    # There should be only a single legend in the figure
    # TODO: Would be nice to get that one below the plots
    axrr[0, 0].legend()

    plt.tight_layout()

    return pd.DataFrame(res_out, columns=["k", "avg_top_k_acc", "scoring_method", "ms2scorer"])


def _compute_color(baseline, other, ctype):
    if ctype.startswith("gain"):
        cvalue = (other / baseline) - 1

        if ctype.endswith("perc"):
            cvalue *= 100
    elif ctype == "improvement":
        cvalue = other - baseline
    else:
        raise ValueError("Invalid ctype: '%s'." % ctype)

    return cvalue


def _reshape_output(df_d):
    return [
        _df.melt(ignore_index=False, value_name="top_k_acc").reset_index().assign(ms2scorer=_ms2scorer_i, k=_k)
        for (_ms2scorer_i, _k), _df in df_d.items()
    ]


def plot__01__b(
    res__baseline:  List[pd.DataFrame],
    res__ssvm:      List[pd.DataFrame],
    res__rtfilter:  List[pd.DataFrame],
    res__xlogp3:    List[pd.DataFrame],
    res__bach2020:  List[pd.DataFrame],
    ks: Optional[List[int]] = None,
    ctype: str = "improvement",
    weighted_average: bool = False,
    raise_on_missing_results: bool = True,
    label_format: str = ".0f",
    verbose: bool = False
):
    """
    Plot to illustrate the performance difference between Only MS2 and the four (4) different score integration
    approaches. The input structure is the same as for "plot__01__a".

    :param res__baseline: list of dataframe, containing the ranking results for Only MS2.

    :param res__ssvm: list of dataframe, containing the ranking results for the SSVM approach.

    :param res__rtfilter: list of dataframe, containing the RT filtering results.

    :param res__xlogp3: list of dataframe, containing the XLogP3 re-scoring results.

    :param res__bach2020: list of dataframe, containing the results achieved by Bach et al.'s method.

    :param ks: list of scalars, top-k values to plot. By default, the variable is set to [1, 20], which means that the
        top-1 and top-20 values will be plotted.

    :param ctype: string, which statistic should be encoded using the color of the heatmap plot. Choises are:

        "improvement": Difference between top-k (score integration) and top-k (baseline) in percentage points.
        "gain": Performance gain of top-k (score integration) over top-k (baseline)
        "gain_perc": Performance gain of top-k (score integration) over top-k (baseline) in percentages

    :param weighted_average: boolean, indicating whether the average the top-k accuracy should be first computed within
        each dataset and subsequently averaged across the datasets. If False, than all samples are treated equally and
        simply averaged directly across all datasets.

    :param raise_on_missing_results: boolean, indicating whether an error should be raised if results are missing. If
        False, than only those results which are available for all scoring methods of a particular MS2 base-scorer are
        considered for the plots.

    :param label_format: string, format string for the labels. Default: Rounded to full number.

    :param verbose: boolean, indicating whether all kinds of stuff should be printed, which somehow can be helpful for
        debugging.

    :return: pd.DataFrame, data shown in the plot for publication.
    """
    assert ctype in ["improvement", "gain", "gain_perc"], "Invalid ctype value: '%s'" % ctype

    ctype_labels = {
        "improvement": "Top-k acc. improvement (%p)",
        "gain": "Performance gain",
        "gain_perc": "Performance gain (%)"
    }

    # Total number of scoring methods in our manuscript
    n_methods = 5

    # Number of MS2 scorers must be equal
    assert len(res__baseline) == len(res__ssvm)
    assert len(res__baseline) == len(res__rtfilter)
    assert len(res__baseline) == len(res__xlogp3)
    assert len(res__baseline) == len(res__bach2020)
    n_scorer = len(res__baseline)

    # There should be only one scoring and one top-k accuracy computation method in each dataframe
    for k in ["scoring_method", "top_k_method", "ms2scorer"]:
        for i in range(n_scorer):
            assert res__baseline[i][k].nunique() == 1
            assert res__ssvm[i][k].nunique() == 1
            assert res__rtfilter[i][k].nunique() == 1
            assert res__xlogp3[i][k].nunique() == 1
            assert res__bach2020[i][k].nunique() == 1

    # For the SSVM all results should be 8 SSVM models
    for i in range(n_scorer):
        assert np.all(res__ssvm[i]["n_models"] == 8), "There seems to be SSVM models missing."

    # Get all available results and restrict them if needed by only using the result intersection
    res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020 = _process_dfs__01(
        res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020, raise_on_missing_results
    )

    # Get number of datasets
    datasets = [res__baseline[i]["dataset"].unique().tolist() for i in range(n_scorer)]

    # Get a new figure
    fig, axrr = plt.subplots(n_scorer, len(ks), figsize=(20, 5 * n_scorer), sharex=False, sharey="row", squeeze=False)

    # Plot Top-k curve
    if verbose:
        print("We expect 17500 result rows")

    # For Machine Intelligence we need to write out the content of the figure
    _label_df = {}
    _color_df = {}

    # Do the plotting ...
    for i, _res in enumerate(zip(res__baseline, res__ssvm, res__rtfilter, res__xlogp3, res__bach2020)):
        _ms2scorer_i = _res[0]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[1]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[2]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[3]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[4]["ms2scorer"].unique().item()

        if verbose:
            print("Rows (MS2-scorer='%s'):" % _ms2scorer_i)
            print("Number of samples: %d" % (_res[0]["k"] == 1).sum())

        # Top-k accuracy matrices: (1) label matrix and (2) color encoding matrix
        lab_val_mat = np.full((len(ks), n_methods, len(datasets[i]) + 1), fill_value=np.nan)
        col_val_mat = np.full((len(ks), n_methods, len(datasets[i]) + 1), fill_value=np.nan)
        # shape = (
        #   number_of_ks_to_plot,
        #   number_of_score_integration_methods,
        #   number_of_datasets_plus_avg
        # )
        lab_val_d = {}

        for j, k in enumerate(ks):
            # Get current axis
            ax = axrr[i, j]
            # i: Each MS2 scorer is plotted into its own row
            # j: Each top-k is plotted into its own column

            for l, ds in enumerate(datasets[i]):
                # Top-k accuracy as label
                for m in range(n_methods):
                    # Get the top-k values for the current dataset (= MassBank group) and the current value of k
                    # (top-k). This might be several values, depending on the number of evaluation samples in each
                    # dataset.
                    _top_k_values = _res[m][(_res[m]["dataset"] == ds) & (_res[m]["k"] == k)]["top_k_acc"].values

                    # As label, we use the average performance (WITHIN DATASET).
                    lab_val_mat[j, m, l] = np.mean(_top_k_values)

                    if not weighted_average:
                        lab_val_d[(j, m, l)] = _top_k_values

                # Performance gain or improvement as color
                for m in range(n_methods):
                    # Note: The first score integration method is Only MS2 (= baseline)
                    col_val_mat[j, m, l] = _compute_color(
                        baseline=lab_val_mat[j, 0, l], other=lab_val_mat[j, m, l], ctype=ctype
                    )

            # Compute average performance (ACROSS THE DATASETS)
            if weighted_average:
                lab_val_mat[j, :, -1] = np.mean(lab_val_mat[j, :, :-1], axis=1)
            else:
                for m in range(n_methods):
                    lab_val_mat[j, m, -1] = np.mean(
                        np.concatenate(
                            [lab_val_d[(j, m, l)] for l in range(len(datasets[i]))]
                        )
                    )

            for m in range(n_methods):
                col_val_mat[j, m, -1] = _compute_color(
                    baseline=lab_val_mat[j, 0, -1], other=lab_val_mat[j, m, -1], ctype=ctype
                )

            # Wrap the matrices into dataframes
            _index = pd.Index(
                data=[_res[m]["scoring_method"].unique().item() for m in range(n_methods)],
                name="scoring_method"
            )
            _columns = pd.Index(data=datasets[i] + ["AVG."], name="dataset")
            _label_df[(_ms2scorer_i, k)] = pd.DataFrame(lab_val_mat[j], index=_index, columns=_columns)
            _color_df[(_ms2scorer_i, k)] = pd.DataFrame(col_val_mat[j], index=_index, columns=_columns)

            # Plot the heatmap
            sns.heatmap(
                data=_color_df[(_ms2scorer_i, k)],
                # -- Label design --
                annot=_label_df[(_ms2scorer_i, k)],
                fmt=label_format,
                annot_kws={"fontweight": "normal"},
                # -- Color design --
                center=0,
                cmap="PiYG",
                cbar_kws={
                    "location": "bottom",
                    "orientation": "horizontal",
                    "shrink": 0.5,
                    "label": ctype_labels[ctype]
                },
                linewidths=0.75,
                ax=ax,
                square=True
            )

            # Visually separate the baseline and average cells from the rest
            ax.hlines(1, 0, col_val_mat.shape[2], color="black", linewidth=0.75)
            ax.vlines(len(datasets[i]), 0, n_methods, color="black", linewidth=0.75)

            ax.set_title(
                "%s - top-%d accuracy" % (_res[0]["ms2scorer"].unique().item(), k), fontsize=14, fontweight="bold",
                pad=16
            )

            if i == (n_scorer - 1):
                ax.set_xlabel("Dataset (MassBank group)")
            else:
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("Scoring method")
            else:
                ax.set_ylabel("")

    plt.tight_layout()

    # Reshape output for the figure reproduction and return
    return pd.concat(_reshape_output(_label_df), ignore_index=True), \
           pd.concat(_reshape_output(_color_df), ignore_index=True)


def _process_dfs__03(res__baseline, res__ssvm__2D, res__ssvm__3D, raise_on_missing_results):
    n_scorer = len(res__baseline)

    res_sets = []
    for i in range(n_scorer):
        restrict_results = False

        # Only MS2
        _res_set_baseline = _get_res_set(res__baseline[i])
        res_sets.append(_res_set_baseline)

        # SSVM (2D fingerprints)
        _res = _get_res_set(res__ssvm__2D[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("SSVM has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        # SSVM (3D fingerprints)
        _res = _get_res_set(res__ssvm__3D[i])
        if _res != _res_set_baseline:
            if raise_on_missing_results:
                raise ValueError("RT filtering has missing results!")
            else:
                res_sets[-1] &= _res
                restrict_results = True

        if restrict_results:
            res__baseline[i] = _restrict_df(res__baseline[i], res_sets[i])
            res__ssvm__2D[i] = _restrict_df(res__ssvm__2D[i], res_sets[i])
            res__ssvm__3D[i] = _restrict_df(res__ssvm__3D[i], res_sets[i])

    # Sort results so that the rows would match
    for i in range(n_scorer):
        res__baseline[i] = res__baseline[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__ssvm__2D[i] = res__ssvm__2D[i].sort_values(by=["dataset", "eval_indx", "k"])
        res__ssvm__3D[i] = res__ssvm__3D[i].sort_values(by=["dataset", "eval_indx", "k"])

    return res__baseline, res__ssvm__2D, res__ssvm__3D


def plot__03__a(
        res__baseline:  List[pd.DataFrame],
        res__ssvm__2D:  List[pd.DataFrame],
        res__ssvm__3D:  List[pd.DataFrame],
        aspect: str = "landscape",
        max_k: int = 20,
        weighted_average: bool = False,
        raise_on_missing_results: bool = True,
        verbose: bool = False
):
    """
    Plot comparing the top-k accuracy performance for k in {1, ..., max_k} of the different scoring methods:

        - baseline: Only MS2 information is used
        - ssvm__2D: Proposed Structured Support Vector Regression (SSVM) model with 2D fingerprints
        - ssvm__3D: Proposed Structured Support Vector Regression (SSVM) model with 3D fingerprints

    The for each scoring method a list of dataframes is provided. Each DataFrame has the following structure:

    k 	top_k_method 	scoring_method 	correct_leq_k 	seq_length 	n_models 	eval_indx 	dataset 	top_k_acc 	ds 	    lloss_mode 	    mol_feat 	            mol_id 	ms2scorer 	    ssvm_flavor
    1 	csi 	        Only-MS2        3.000000 	    50 	        8 	        0 	        AC_003 	    6.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    2 	csi 	        Only-MS2        5.000000  	    50 	        8 	        0 	        AC_003 	    10.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    3 	csi 	        Only-MS2        7.000000 	    50 	        8 	        0 	        AC_003 	    14.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    4 	csi 	        Only-MS2        9.000000 	    50 	        8 	        0 	        AC_003 	    18.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    5 	csi 	        Only-MS2        11.000000 	    50 	        8 	        0 	        AC_003 	    22.000000 	AC_003 	mol_feat_fps 	FCFP__binary__all__2D 	cid 	CFM-ID (v4) 	default
    ...

    whereby the "scoring_method" differs.

    Each list element corresponds to a different MS2 base-scorer, e.g. CFM-ID, MetFrag, ...

    :param res__baseline: list of dataframe, containing the ranking results for Only MS2.

    :param res__ssvm__2D: list of dataframe, containing the ranking results for the SSVM approach with 2D fingerprints.

    :param res__ssvm__3D: list of dataframe, containing the ranking results for the SSVM approach with 3D fingerprints.

    :param aspect: string, indicating which layout for the plot should be used:

        "landscape":
                        CFMID  METFRAG    SIRIUS
                         ____   ____       ____
                        |    | |    | ... |    |   Top-k
                        |____| |____|     |____|
                         ____   ____       ____
                        |    | |    | ... |    |   Top-k improvement over the baseline
                        |____| |____|     |____|

        "portrait":
                        Top-k   Top-k improvement over the baseline
                         ____   ____
                CFMID   |    | |    |
                        |____| |____|
                         ____   ____
                MEFRAG  |    | |    |
                        |____| |____|
                              .
                              .
                              .
                         ____   ____
                SIRIUS  |    | |    |
                        |____| |____|

    :param max_k: scalar, what is the maximum k value for the top-k curve plot.

    :param weighted_average: boolean, indicating whether the average the top-k accuracy should be first computed within
        each dataset and subsequently averaged across the datasets. If False, than all samples are treated equally and
        simply averaged directly across all datasets.

    :param raise_on_missing_results: boolean, indicating whether an error should be raised if results are missing. If
        False, than only those results which are available for all scoring methods of a particular MS2 base-scorer are
        considered for the plots.

    :param verbose: boolean, indicating whether all kinds of stuff should be printed, which somehow can be helpful for
        debugging.

    :return: pd.DataFrame, data shown in the plot for publication.
    """
    def _acc_info_printer(baseline, other, k):
        print(
            "\ttop-%d: baseline = %.1f%%, other = %.1f%%, improvement = %.1f%%p, gain = %.1f%%, n = %.1f" %
            (
                k,
                baseline["top_k_acc"][other["k"] == k],
                other["top_k_acc"][other["k"] == k],
                (other["top_k_acc"] - baseline["top_k_acc"])[other["k"] == k],
                ((other["top_k_acc"] / baseline["top_k_acc"])[other["k"] == k] - 1) * 100,
                (other["correct_leq_k"] - baseline["correct_leq_k"])[other["k"] == k]
            )
        )

    assert aspect in ["landscape", "portrait"], "Invalid aspect value: '%s'" % aspect

    # Number of MS2 scorers must be equal
    assert len(res__baseline) == len(res__ssvm__2D)
    assert len(res__baseline) == len(res__ssvm__3D)
    n_scorer = len(res__baseline)

    # There should be only one scoring and one top-k accuracy computation method in each dataframe
    for k in ["scoring_method", "top_k_method", "ms2scorer"]:
        for i in range(n_scorer):
            assert res__baseline[i][k].nunique() == 1
            assert res__ssvm__2D[i][k].nunique() == 1
            assert res__ssvm__3D[i][k].nunique() == 1

    # For the SSVM all results should be 8 SSVM models
    for i in range(n_scorer):
        assert np.all(res__ssvm__2D[i]["n_models"] == 8), "2D: There seems to be SSVM models missing."
        assert np.all(res__ssvm__3D[i]["n_models"] == 8), "2D: There seems to be SSVM models missing."

    # Get all available results and restrict them if needed by only using the result intersection
    res__baseline, res__ssvm__2D, res__ssvm__3D, = _process_dfs__03(
        res__baseline, res__ssvm__2D, res__ssvm__3D, raise_on_missing_results
    )

    # Get a new figure
    if aspect == "portrait":
        _n_rows = n_scorer
        _n_cols = 2
        _figsize = (9, 3 * n_scorer)
    else:  # landscape
        _n_rows = 2
        _n_cols = n_scorer
        _figsize = (4.5 * n_scorer, 5.75)

    fig, axrr = plt.subplots(_n_rows, _n_cols, figsize=_figsize, sharex="all", sharey=False, squeeze=False)

    # Set some plot properties
    k_ticks = np.arange(0, max_k + 1, 5)
    k_ticks[0] = 1

    # For Machine Intelligence we need to provide the raw-data for the plot
    res_out = []

    # Plot Top-k curve
    if verbose:
        print("We expect 4700 result rows")

    for idx, (a, b, c) in enumerate(zip(res__baseline, res__ssvm__2D, res__ssvm__3D)):
        assert a["ms2scorer"].unique().item() == b["ms2scorer"].unique().item()
        assert a["ms2scorer"].unique().item() == c["ms2scorer"].unique().item()

        if verbose:
            print("Rows (MS2-scorer='%s'):" % a["ms2scorer"].unique().item())
            print("Number of samples: %d" % (a["k"] == 1).sum())

        # Get current axis and set labels
        if aspect == "portrait":
            # first column
            ax = axrr[idx, 0]
            ax.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            ax.set_ylabel("Top-k accuracy (%)", fontsize=12)

            # second column
            ax2 = axrr[idx, 1]
            ax2.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            ax2.set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)
        else:
            # first row
            ax = axrr[0, idx]
            ax.set_title(a["ms2scorer"].unique().item(), fontweight="bold")
            axrr[0, 0].set_ylabel("Top-k accuracy (%)", fontsize=12)

            # second row
            ax2 = axrr[1, idx]
            axrr[1, 0].set_ylabel("Top-k accuracy\nimprovement (%p)", fontsize=12)

        # Baseline
        if verbose:
            print("Baseline: ", len(a))

        if weighted_average:
            bl = a[a["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            bl = a[a["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(bl["k"], bl["top_k_acc"], where="post", label=a["scoring_method"].unique().item(), color="black")
        ax2.hlines(0, 1, max_k, colors="black", label=a["scoring_method"].unique().item())

        res_out += list(zip(
            bl["k"], bl["top_k_acc"], [a["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # SSVM (2D)
        if verbose:
            print("SSVM (2D): ", len(b))

        if weighted_average:
            tmp = b[b["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            tmp = b[b["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(
            tmp["k"], tmp["top_k_acc"], where="post", label=b["scoring_method"].unique().item(), color="blue",
            linestyle="dashed"
        )

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=b["scoring_method"].unique().item(),
            color="blue", linestyle="dashed"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [b["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # SSVM (3D)
        if verbose:
            print("SSVM (3D): ", len(c))

        if weighted_average:
            tmp = c[c["k"] <= max_k].groupby(["dataset", "k"]).mean().reset_index().groupby("k").mean().reset_index()
        else:
            tmp = c[c["k"] <= max_k].groupby("k").mean().reset_index()

        ax.step(
            tmp["k"], tmp["top_k_acc"], where="post", label=c["scoring_method"].unique().item(), color="blue",
            linestyle="dotted"
        )

        assert np.all(tmp["k"] == bl["k"])
        ax2.step(
            tmp["k"], tmp["top_k_acc"] - bl["top_k_acc"], where="post", label=c["scoring_method"].unique().item(),
            color="blue", linestyle="dotted"
        )

        if verbose:
            for _k in [1, 20]:
                _acc_info_printer(bl, tmp, _k)

        res_out += list(zip(
            tmp["k"], tmp["top_k_acc"], [c["scoring_method"].unique().item()] * max_k,
            [a["ms2scorer"].unique().item()] * max_k
        ))
        # ---

        # Set some further axis properties
        ax.set_xticks(k_ticks)
        ax2.set_xticks(k_ticks)

        ax.grid(axis="y")
        ax2.grid(axis="y")

        if (aspect == "portrait") and (idx == (n_scorer - 1)):
            ax.set_xlabel("k")
            ax2.set_xlabel("k")
        elif aspect == "landscape":
            ax2.set_xlabel("k")

    # There should be only a single legend in the figure
    # TODO: Would be nice to get that one below the plots
    axrr[0, 0].legend()

    plt.tight_layout()

    return pd.DataFrame(res_out, columns=["k", "avg_top_k_acc", "scoring_method", "ms2scorer"])


def plot__03__b(
    res__baseline:  List[pd.DataFrame],
    res__ssvm__2D:      List[pd.DataFrame],
    res__ssvm__3D:  List[pd.DataFrame],
    ks: Optional[List[int]] = None,
    ctype: str = "improvement",
    weighted_average: bool = False,
    raise_on_missing_results: bool = True,
    label_format: str = ".0f",
    verbose: bool = False
):
    """
    Plot to illustrate the performance difference between Only MS2 and the four (4) different score integration
    approaches. The input structure is the same as for "plot__03__a".

    :param res__baseline: list of dataframe, containing the ranking results for Only MS2.

    :param res__ssvm__2D: list of dataframe, containing the ranking results for the SSVM approach with 2D fingerprints.

    :param res__ssvm__3D: list of dataframe, containing the ranking results for the SSVM approach with 3D fingerprints.

    :param ks: list of scalars, top-k values to plot. By default, the variable is set to [1, 20], which means that the
        top-1 and top-20 values will be plotted.

    :param ctype: string, which statistic should be encoded using the color of the heatmap plot. Choises are:

        "improvement": Difference between top-k (score integration) and top-k (baseline) in percentage points.
        "gain": Performance gain of top-k (score integration) over top-k (baseline)
        "gain_perc": Performance gain of top-k (score integration) over top-k (baseline) in percentages

    :param weighted_average: boolean, indicating whether the average the top-k accuracy should be first computed within
        each dataset and subsequently averaged across the datasets. If False, than all samples are treated equally and
        simply averaged directly across all datasets.

    :param raise_on_missing_results: boolean, indicating whether an error should be raised if results are missing. If
        False, than only those results which are available for all scoring methods of a particular MS2 base-scorer are
        considered for the plots.

    :param label_format: string, format string for the labels. Default: Rounded to full number.

    :param verbose: boolean, indicating whether all kinds of stuff should be printed, which somehow can be helpful for
        debugging.

    :return: pd.DataFrame, data shown in the plot for publication.
    """
    assert ctype in ["improvement", "gain", "gain_perc"], "Invalid ctype value: '%s'" % ctype

    ctype_labels = {
        "improvement": "Top-k acc. improvement (%p)",
        "gain": "Performance gain",
        "gain_perc": "Performance gain (%)"
    }

    # Total number of scoring methods in our manuscript
    n_methods = 3

    # Number of MS2 scorers must be equal
    assert len(res__baseline) == len(res__ssvm__2D)
    assert len(res__baseline) == len(res__ssvm__3D)
    n_scorer = len(res__baseline)

    # There should be only one scoring and one top-k accuracy computation method in each dataframe
    for k in ["scoring_method", "top_k_method", "ms2scorer"]:
        for i in range(n_scorer):
            assert res__baseline[i][k].nunique() == 1
            assert res__ssvm__2D[i][k].nunique() == 1
            assert res__ssvm__3D[i][k].nunique() == 1

    # For the SSVM all results should be 8 SSVM models
    for i in range(n_scorer):
        assert np.all(res__ssvm__2D[i]["n_models"] == 8), "2D: There seems to be SSVM models missing."
        assert np.all(res__ssvm__3D[i]["n_models"] == 8), "3D: There seems to be SSVM models missing."

    # Get all available results and restrict them if needed by only using the result intersection
    res__baseline, res__ssvm__2D, res__ssvm__3D, = _process_dfs__03(
        res__baseline, res__ssvm__2D, res__ssvm__3D, raise_on_missing_results
    )

    # Get number of datasets
    datasets = [res__baseline[i]["dataset"].unique().tolist() for i in range(n_scorer)]

    # Get a new figure
    fig, axrr = plt.subplots(n_scorer, len(ks), figsize=(20, 5 * n_scorer), sharex=False, sharey="row", squeeze=False)

    # Plot Top-k curve
    if verbose:
        print("We expect 4700 result rows")

    # For Machine Intelligence we need to write out the content of the figure
    _label_df = {}
    _color_df = {}

    # Do the plotting ...
    for i, _res in enumerate(zip(res__baseline, res__ssvm__2D, res__ssvm__3D)):
        _ms2scorer_i = _res[0]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[1]["ms2scorer"].unique().item()
        assert _ms2scorer_i == _res[2]["ms2scorer"].unique().item()

        if verbose:
            print("Rows (MS2-scorer='%s'):" % _ms2scorer_i)
            print("Number of samples: %d" % (_res[0]["k"] == 1).sum())

        # Top-k accuracy matrices: (1) label matrix and (2) color encoding matrix
        lab_val_mat = np.full((len(ks), n_methods, len(datasets[i]) + 1), fill_value=np.nan)
        col_val_mat = np.full((len(ks), n_methods, len(datasets[i]) + 1), fill_value=np.nan)
        # shape = (
        #   number_of_ks_to_plot,
        #   number_of_score_integration_methods,
        #   number_of_datasets_plus_avg
        # )
        lab_val_d = {}

        for j, k in enumerate(ks):
            # Get current axis
            ax = axrr[i, j]
            # i: Each MS2 scorer is plotted into its own row
            # j: Each top-k is plotted into its own column

            for l, ds in enumerate(datasets[i]):
                # Top-k accuracy as label
                for m in range(n_methods):
                    # Get the top-k values for the current dataset (= MassBank group) and the current value of k
                    # (top-k). This might be several values, depending on the number of evaluation samples in each
                    # dataset.
                    _top_k_values = _res[m][(_res[m]["dataset"] == ds) & (_res[m]["k"] == k)]["top_k_acc"].values

                    # As label, we use the average performance (WITHIN DATASET).
                    lab_val_mat[j, m, l] = np.mean(_top_k_values)

                    if not weighted_average:
                        lab_val_d[(j, m, l)] = _top_k_values

                # Performance gain or improvement as color
                for m in range(n_methods):
                    # Note: The first score integration method is Only MS2 (= baseline)
                    col_val_mat[j, m, l] = _compute_color(
                        baseline=lab_val_mat[j, 0, l], other=lab_val_mat[j, m, l], ctype=ctype
                    )

            # Compute average performance (ACROSS THE DATASETS)
            if weighted_average:
                lab_val_mat[j, :, -1] = np.mean(lab_val_mat[j, :, :-1], axis=1)
            else:
                for m in range(n_methods):
                    lab_val_mat[j, m, -1] = np.mean(
                        np.concatenate(
                            [lab_val_d[(j, m, l)] for l in range(len(datasets[i]))]
                        )
                    )

            for m in range(n_methods):
                col_val_mat[j, m, -1] = _compute_color(
                    baseline=lab_val_mat[j, 0, -1], other=lab_val_mat[j, m, -1], ctype=ctype
                )

            # Wrap the matrices into dataframes
            _index = pd.Index(
                data=[_res[m]["scoring_method"].unique().item() for m in range(n_methods)],
                name="scoring_method"
            )
            _columns = pd.Index(data=datasets[i] + ["AVG."], name="dataset")
            _label_df[(_ms2scorer_i, k)] = pd.DataFrame(lab_val_mat[j], index=_index, columns=_columns)
            _color_df[(_ms2scorer_i, k)] = pd.DataFrame(col_val_mat[j], index=_index, columns=_columns)

            # Plot the heatmap
            sns.heatmap(
                data=_color_df[(_ms2scorer_i, k)],
                # -- Label design --
                annot=_label_df[(_ms2scorer_i, k)],
                fmt=label_format,
                annot_kws={"fontweight": "normal", "fontsize": 12},
                # -- Color design --
                center=0,
                cmap="PiYG",
                cbar_kws={
                    "location": "bottom",
                    "orientation": "horizontal",
                    "shrink": 0.5,
                    "label": ctype_labels[ctype]
                },
                linewidths=0.75,
                ax=ax,
                square=True
            )

            # Visually separate the baseline and average cells from the rest
            ax.hlines(1, 0, col_val_mat.shape[2], color="black", linewidth=0.75)
            ax.vlines(len(datasets[i]), 0, n_methods, color="black", linewidth=0.75)

            ax.set_title(
                "%s - top-%d accuracy" % (_res[0]["ms2scorer"].unique().item(), k), fontsize=14, fontweight="bold",
                pad=16
            )

            if i == (n_scorer - 1):
                ax.set_xlabel("Dataset (MassBank group)")
            else:
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("Scoring method")
            else:
                ax.set_ylabel("")

    plt.tight_layout()

    # Reshape output for the figure reproduction and return
    return pd.concat(_reshape_output(_label_df), ignore_index=True), \
           pd.concat(_reshape_output(_color_df), ignore_index=True)


def table__top_k_acc_per_dataset_with_significance(
        results: pd.DataFrame, p_level: float = 0.05, ks: Optional[List[int]] = None, top_k_method: str = "csi",
        test: str = "ttest", decimals: int = 1
) -> pd.DataFrame:
    """
    Function to generate the table comparing "Only MS" with "MS + RT". Test for significance is performed and indicated,
    if "MS + RT" significantly outperforms "Only MS".

    :param results: pd.DataFrame, results
    :param p_level:
    :param ks:
    :param top_k_method:
    :param test:
    :return:
    """
    if ks is None:
        ks = [1, 5, 10, 20]

    # Check that all needed columns are provide
    for column in ["k", "top_k_method", "scoring_method", "dataset", "eval_indx", "top_k_acc"]:
        if column not in results.columns:
            raise ValueError("Column {} is missing from the data-frame {}".format(column, results.columns))

    # Collect all "MS + RT" settings, e.g., "MS + RT" or ["MS + RT (global)", "MS + RT (local)"], ...
    ms_p_rt_labels = results[results["scoring_method"] != "Only MS"]["scoring_method"].unique().tolist()
    if len(ms_p_rt_labels) < 1:
        raise ValueError(
            "There must be at least one other scoring method other than 'Only MS': {}".format(
                results["scoring_method"].unique().tolist()
            )
        )

    # Subset results for the specified top-ks and the top-k determination method (casmi or csi)
    _results = results[(results["k"].isin(ks)) & (results["top_k_method"] == top_k_method)]

    results_out = pd.DataFrame()

    for (k, ds), res in _results.groupby(["k", "dataset"]):
        # Separate the "Only MS" setting
        _only_ms = res[res["scoring_method"] == "Only MS"].sort_values(by="eval_indx")

        # New row(s) for the output data-frame
        _df = {
            "k": k,
            "dataset": ds,
            "n_samples": len(_only_ms),
            "scoring_method": ["Only MS"],
            "top_k_acc": [np.mean(_only_ms["top_k_acc"]).item()],
            "p_value": [1.0]
        }

        # Load the "MS + RT" for each label
        for l in ms_p_rt_labels:
            _ms_p_rt = res[res["scoring_method"] == l].sort_values(by="eval_indx")

            _df["scoring_method"].append(l)
            _df["top_k_acc"].append(np.mean(_ms_p_rt["top_k_acc"]).item())

            if len(_ms_p_rt) <= 1:
                # We need to have more than one value to perform a significance test
                _p = np.nan
            else:
                # Perform the significance test
                if test == "wilcoxon":
                    _p = wilcoxon(_only_ms["top_k_acc"], _ms_p_rt["top_k_acc"], alternative="less")[1]
                elif test == "ttest":
                    _p = ttest_rel(_only_ms["top_k_acc"], _ms_p_rt["top_k_acc"], alternative="less")[1]
                else:
                    raise ValueError("Invalid significance test: %s" % test)

            _df["p_value"].append(_p)

        # Convert to accuracy strings
        _df["top_k_acc__as_labels"] = []
        for idx, s in enumerate(_df["top_k_acc"]):
            _is_best = (
                    np.round(_df["top_k_acc"][idx], decimals=decimals)
                    ==
                    np.max(np.round(_df["top_k_acc"], decimals=decimals))
            )
            _is_sign = False if np.isnan(_df["p_value"][idx]) else (_df["p_value"][idx] <= p_level)

            _lab = "{}".format(np.round(_df["top_k_acc"][idx], decimals=decimals))

            if _is_best:
                _lab = "| " + _lab
            if _is_sign:
                _lab = _lab + " *"

            _df["top_k_acc__as_labels"].append(_lab)

        results_out = pd.concat((results_out, pd.DataFrame(_df)), ignore_index=True)

    return results_out
