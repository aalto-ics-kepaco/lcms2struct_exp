import argparse
import sqlite3
import os
import sys

from typing import Dict, Optional, Tuple


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


def parse_eval_set_id_information(args) -> Tuple[str, int, int, int]:
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
    eval_set_idx = int(eval_set_id[:2])

    # Extract dataset name from the DB corresponding the evaluation set
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)
    try:
        ds, = db.execute("SELECT name FROM datasets ORDER BY name").fetchall()[eval_set_idx]
    finally:
        db.close()

    # Get the sub-sample set index
    spl_idx = int(eval_set_id[2:])

    return ds, spl_idx, args.eval_set_id, eval_set_idx


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("eval_set_id", type=int)
    arg_parser.add_argument("--db_fn", type=str)
    arg_parser.add_argument("--ms2scorer", type=str)
    arg_parser.add_argument("--output_dir", type=str)
    arg_parser.add_argument("--score_integration_approach", type=str)
    arg_parser.add_argument("--molecule_features", type=str)
    arg_parser.add_argument("--molecule_identifier", type=str)
    arg_parser.add_argument("--predictor", type=str)
    arg_parser.add_argument("--debug", type=int, default=0)
    arg_parser.add_argument("--verbose", type=int, default=0)
    args = arg_parser.parse_args()

    eval_ds, eval_spl_idx, eval_set_id, _eval_set_idx = parse_eval_set_id_information(args)

    odir_res = get_odir_res(eval_ds, args)
    if args.verbose:
        print(odir_res)

    # Print the basic information about the experiment
    msg = "eval_set_id=%04d, ds=%s,eval_set=%d,eval_spl_idx=%d,feat=%s,ms2scorer=%s" % (
        args.eval_set_id, eval_ds, _eval_set_idx, eval_spl_idx, args.molecule_features, args.ms2scorer
    )

    # We check for the last file that is written
    exists = os.path.exists(
        os.path.join(odir_res, dict2fn({"spl": eval_spl_idx, "cand_agg_id": "inchikey1"}, pref="top_k", ext="tsv"))
    )

    if exists:
        if args.verbose:
            print("EXISTS:%s" % msg)
        sys.exit(0)
    else:
        if args.verbose:
            print("MISSING:%s" % msg)
        sys.exit(1)
