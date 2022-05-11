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
import os
import re
import pandas as pd
import glob

from typing import Dict, Optional, Tuple

from pandas.api.types import is_float, is_bool, is_integer

# ----------------------------------------------------------------------------------------------------------------------
# Helper scripts to load the result of our experiments
# ----------------------------------------------------------------------------------------------------------------------


def dict2fn(d: Dict, sep: str = "__", pref: Optional[str] = None, ext: Optional[str] = None) -> str:
    """
    :param d: dictionary, input (key, value)-pairs

    :param sep: string, used to separate the key=value pairs in the filename

    :param pref: string, added as prefix to the filename. The prefix and rest of the filename is separated by the string
        specified in the 'sep' parameter

    :param ext: string, added as extension to the filename. The extension is separated by the systems file-extension
        separator, e.g. '.'.

    :return: string, compiled filename
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


def fn2dict(params: str) -> Tuple[Dict, bool, Optional[str], Optional[str]]:
    """
    Roughly the inverse function of "dict2fn".

    :param params: string, filename to split up

    :return:
    """
    param_name_pattern = re.compile(r'(?:[a-zA-Z0-9]+_?)+=')

    # Debugging outputs do have an additional prefix which we need to remove
    if params.startswith("debug__"):
        params = params.removeprefix("debug__")
        is_debug = True
        pref = "debug"
    else:
        # FIXME: we only support "debug" as prefix, here
        is_debug = False
        pref = None

    # Get the file extension (assume that there is no "." in the filename that does NOT separate the extension)
    _ext_idx = params.find(os.extsep)
    if _ext_idx >= 0:
        ext = params[(_ext_idx + 1):]
    else:
        ext = None

    # Split the filename and extract the (key, value)-pairs
    ks = [m.removesuffix("=") for m in param_name_pattern.findall(params)]
    vs = [v.removesuffix("__") for v in param_name_pattern.split(params) if len(v) > 0]
    assert len(ks) == len(vs)

    # Construct the output dictionary
    out = {}
    for k, v in zip(ks, vs):
        if is_bool(v):
            out[k] = bool(v)
        elif is_integer(v):
            out[k] = int(v)
        elif is_float(v):
            out[k] = int(v)
        else:
            assert isinstance(v, str)
            out[k] = v

    return out, is_debug, pref, ext


def load_topk__publication(
        setting: Dict, agg_setting: Dict, basedir: str = ".", top_k_method: str = "csi",
        load_max_model_number: bool = False
) -> pd.DataFrame:
    """
    Load the Top-k accuracies in the "publications" folder. These are the results for the SSVM (exp_ver >= 3)

    :param setting: dictionary, specifying the parameters of the experiments for which the results should be loaded.

    :param agg_setting: dictionary, specifying the parameter of the margin score aggregation, that means how are the
        molecular candidates identified when choosing the margin score.

    :param basedir: string, directory containing the results

    :param top_k_method: string, specifies which top-k accuracy calculation method should be used. We always use the one
        used in the original SIRIUS publication [1].

    :param load_max_model_number: boolean, indicating whether only the results of the averaged marginals for the maximum
        number of available SSVM models should be loaded.

    :return: dataframe, containing all results (row-wise concatenation) that match the "setting" and "agg_setting"

    :references:

    [1] Dührkop, Kai / Shen, Huibin / Meusel, Marvin / Rousu, Juho / Böcker, Sebastian
        Searching molecular structure databases with tandem mass spectra using CSI:FingerID
        2015
    """
    assert top_k_method == "csi", "We always use the top-k accuracy calculated as in the original SIRIUS publication."

    # Collect result data-frames. There might be several, e.g. if the "setting" dictionary contains wildcards (*) for
    # some parameters, such as the dataset.
    df = []

    # Prefix of the result file depending on whether only the results for the maximum number of SSVM models should be
    # loaded.
    _top_k_fn = "top_k__max_models" if load_max_model_number else "top_k"

    # Iterate over all result files matching the parameters
    for ifn in sorted(glob.glob(
        os.path.join(
            basedir, dict2fn(setting), dict2fn(agg_setting, pref="combined"), os.extsep.join([_top_k_fn, "tsv"])
        )
    )):
        # Parse the actual parameters from the basename (the setting might contain wildcards)
        params, _, _, _ = fn2dict(ifn.split(os.sep)[-3])  # /path/to/PARAMS/not/file.tsv --> PARAMS

        # Read the top-k performance results
        _df = pd.read_csv(ifn, sep="\t", dtype={"scoring_method": str, "top_k_method": str})

        # Restrict the results to the one with the specified top-k accuracy method
        _df = _df[_df["top_k_method"] == top_k_method]

        assert _df["top_k_method"].nunique() == 1, "There should be only two different top-k accuracy method."

        # Add the parameters to the dataframe
        for k, v in params.items():
            if k not in _df.columns:
                _df[k] = v

        df.append(_df)

    # All loaded results are concatenated into a single dataframe
    df = pd.concat(df, ignore_index=True)

    return df


def load_topk__comparison(setting: Dict, agg_setting: Dict, basedir: str = ".", top_k_method: str = "csi") \
        -> pd.DataFrame:
    """
    Load the Top-k accuracies in the "comparison" folder. These are the results for the comparison methods, i.e.
    RT filtering, LogP scoring and RO score integration approaches.

    :param setting: dictionary, specifying the parameters of the experiments for which the results should be loaded.

    :param agg_setting: dictionary, specifying the parameter of the margin score aggregation, that means how are the
        molecular candidates identified when choosing the margin score.

    :param basedir: string, directory containing the results

    :param top_k_method: string, specifies which top-k accuracy calculation method should be used. We always use the one
        used in the original SIRIUS publication [1].

    :return: dataframe, containing all results (row-wise concatenation) that match the "setting" and "agg_setting"

    :references:

    [1] Dührkop, Kai / Shen, Huibin / Meusel, Marvin / Rousu, Juho / Böcker, Sebastian
        Searching molecular structure databases with tandem mass spectra using CSI:FingerID
        2015
    """
    spl_pattern = re.compile(r"spl=([0-9]+)")

    assert top_k_method == "csi", "We always use the top-k accuracy calculated as in the original SIRIUS publication."

    # Collect result data-frames. There might be several, e.g. if the "setting" dictionary contains wildcards (*) for
    # some parameters, such as the dataset.
    df = []

    # Input directory
    idir = os.path.join(basedir, dict2fn(setting))

    for ifn in sorted(glob.glob(
        os.path.join(idir, dict2fn({"spl": "*", "cand_agg_id": agg_setting["cand_agg_id"]}, pref="top_k", ext="tsv"))
    )):
        # Parse the actual parameters from the basename (the setting might contain wildcards)
        params, _, _, _ = fn2dict(ifn.split(os.sep)[-2])  # /path/to/PARAMS/file.tsv --> PARAMS

        # Read the top-k performance results
        _df = pd.read_csv(ifn, sep="\t", converters={"scoring_method": str})

        # Restrict the specified top-k method
        _df = _df[_df["top_k_method"] == top_k_method]
        assert _df["top_k_method"].nunique() == 1

        # Add the parameters to the dataframe
        for k, v in params.items():
            if k not in _df.columns:
                _df[k] = v

        # The scoring method label should include the "score_int_app" which distinguishes the different filtering
        # approaches using the predicted retention times
        _df.loc[_df["scoring_method"] == "MS + RT", "scoring_method"] = "MS + RT (%s)" % params["score_int_app"]

        # Add the evaluation split index
        eval_indx = spl_pattern.findall(os.path.basename(ifn).removesuffix(os.extsep + "tsv"))
        assert len(eval_indx) == 1
        _df["eval_indx"] = int(eval_indx[0])

        df.append(_df)

    if len(df) > 0:
        df = pd.concat(df, ignore_index=True).rename({"ds": "dataset"}, axis=1)

        # Compute the accuracy and add it as column
        df["top_k_acc"] = (df["correct_leq_k"] / df["seq_length"]) * 100
    else:
        # Empty dataframe
        df = pd.DataFrame(df)

    return df


def load_topk__cand_set_info(setting: Dict, basedir: str = ".") -> pd.DataFrame:
    """
    Load the Top-k accuracies in the "comparison" folder. These are the results for the comparison methods, i.e.
    RT filtering, LogP scoring and RO score integration approaches.
    """
    df = []

    for ifn in sorted(glob.glob(
            os.path.join(basedir, dict2fn(setting), dict2fn({"spl": "*"}, pref="cand_set_info", ext="tsv"))
    )):
        # Parse the actual parameters from the basename (the setting might contain wildcards)
        params, _, _, _ = fn2dict(ifn.split(os.sep)[-2])  # /path/to/PARAMS/file.tsv --> PARAMS

        # Read the top-k performance results
        _df = pd.read_csv(ifn, sep="\t")

        # Add the parameters to the dataframe
        for k, v in params.items():
            if k not in _df.columns:
                _df[k] = v

        # Add the evaluation split index
        _df["eval_indx"] = int(
            os.path.basename(ifn).removesuffix(os.extsep + "tsv").removeprefix("top_k__").split("=")[1]
        )

        df.append(_df)

    df = pd.concat(df, ignore_index=True)

    return df


if __name__ == "__main__":
    out = fn2dict(
        "debug__ds=AU_003__mol_feat=bouwmeester__smiles_can__mol_id=inchikey1__ms2scorer=sirius__sd__correct_mf__norm__rt_predictor=svr__score_int_app=filtering__global"
    )

    print(out)