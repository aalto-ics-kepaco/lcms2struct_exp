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

import scipy.stats as sp_stats

from run_with_gridsearch import dict2fn

from ssvm.data_structures import CandSQLiteDB_Massbank

# ================
# Setup the Logger
LOGGER = logging.getLogger("gather_rank_changes")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================

SPL_PATTERN = re.compile(r".*=([0-9]+)\.pkl.gz")
DS_PATTERN = re.compile(r"ds=([A-Z]{2,3}_[0-9]{3})")


def get_cli_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("base_dir", help="Directory containing the results for the different tree indices.")
    arg_parser.add_argument("ms2scorer", choices=["metfrag__norm", "sirius__norm", "cfmid4__norm"], type=str)
    arg_parser.add_argument("--molecule_identifier", choices=["cid"], default="cid", type=str)
    arg_parser.add_argument(
        "--mol_feat_retention_order",
        type=str, default="FCFP__binary__all__2D", choices=["FCFP__binary__all__2D", "FCFP__binary__all__3D"]

    )
    arg_parser.add_argument(
        "--margin_aggregation_fun", default="average", choices=["average"]
    )
    arg_parser.add_argument(
        "--candidate_aggregation_identifier",
        default="inchikey1", choices=["inchikey1", "inchikey"]
    )
    arg_parser.add_argument(
        "--db_fn",
        default="/home/bach/Documents/doctoral/projects/lcms2struct_experiments/data/massbank.sqlite"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()

    # Open the candidate DB wrapper
    cands_db = CandSQLiteDB_Massbank(
        args.db_fn, init_with_open_db_conn=True, molecule_identifier=args.molecule_identifier
    )

    # Find all 'ds=*__*/tree_index=*/marginals__aggfun=average__spl=*.tsv'.
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
    agg_setting = {
        "marg_agg_fun": args.margin_aggregation_fun,
        "cand_agg_id": str(args.candidate_aggregation_identifier)  # Make None --> 'None'
    }

    margs_fns = glob.iglob(
        os.path.join(
            args.base_dir,
            dict2fn(setting),
            dict2fn(agg_setting, pref="combined"),
            dict2fn({"spl": "*"}, pref="marginals", ext="pkl.gz")
        )
    )

    for fn in margs_fns:
        _tree, _topk = fn.split(os.sep)[-2:]
        _ds = DS_PATTERN.findall(fn)[0]
        avail_res_df.append([_ds, _tree, _topk, os.path.dirname(fn)])

    # DataFrame storing the (triton) result directories for each (sample, tree) combination
    avail_res_df = pd.DataFrame(avail_res_df, columns=["dataset", "tree", "sample", "fn"])

    # Collect ranks and meta-data for each dataset individually
    for ds in avail_res_df["dataset"].unique():
        LOGGER.info("Dataset: %s" % ds)

        _avail_res_df = avail_res_df[avail_res_df["dataset"] == ds].drop("dataset", axis=1)

        # Define output directory relative to the input directory
        output_dir = os.path.join(os.path.dirname(_avail_res_df["fn"].iloc[0]), dict2fn(agg_setting, pref="combined"))
        LOGGER.info(output_dir)
        assert os.path.exists(output_dir)

        # For each sample we need to aggregate the tree marginals separately
        it_groupby_spl_idx = _avail_res_df.groupby(["sample"])["fn"].apply(list).iteritems()

        df_out = []

        for filename, model_dirs in it_groupby_spl_idx:
            # train / test sample index extracted from, e.g. marginal__spl=3.pkl.gz
            sample_idx = SPL_PATTERN.findall(filename)[0]

            LOGGER.info("Sample index: {} (Number of models = {})".format(sample_idx, len(model_dirs)))

            # Load the dict containing the marginal scores for all MS-tuples and candidates
            with gzip.open(os.path.join(model_dirs[0], filename)) as cands_file:
                cands = pickle.load(cands_file)

            # Iterate over the MS-tuples and determine the rank of the correct structure (MS + RT & Only MS)
            for cand in cands.values():
                # Get the rank given the marginal scores (MS + RT)
                ranks_msplrt = sp_stats.rankdata(- cand["score"], method="ordinal")
                rank_msplrt = ranks_msplrt[cand["score"] == cand["score"][cand["index_of_correct_structure"]]]
                rank_msplrt = [np.min(rank_msplrt), np.max(rank_msplrt)]

                # Get the rank given the MS2 scores (Only MS)
                ms_scores = cand["ms_score"]
                ranks_onlyms = sp_stats.rankdata(- ms_scores, method="ordinal")
                rank_onlyms = ranks_onlyms[ms_scores == ms_scores[cand["index_of_correct_structure"]]]
                rank_onlyms = [np.min(rank_onlyms), np.max(rank_onlyms)]

                # Load the Classyfire class labels from the DB (superclass and class level)
                acc = cand["spectrum_id"].get("spectrum_id")
                super_clf, class_clf, smi_iso_gt = cands_db.db.execute(
                    " \
                     SELECT superclass, class, smiles_iso FROM classyfire_classes \
                        INNER JOIN scored_spectra_meta ssm ON classyfire_classes.molecule = ssm.molecule \
                        INNER JOIN molecules mol ON classyfire_classes.molecule = mol.cid \
                        WHERE accession IS ? \
                    ", (acc, )
                ).fetchone()

                # Load the PubChemLite classes from the DB
                annoTypeCount, agroChemInfo, bioPathway, drugMedicInfo, foodRelated, pharmacoInfo, safetyInfo, toxicityInfo, knownUse, disorderDisease, identification = \
                    cands_db.db.execute(
                        "SELECT AnnoTypeCount, AgroChemInfo, BioPathway, DrugMedicInfo, FoodRelated, PharmacoInfo, SafetyInfo,"
                        "       ToxicityInfo, KnownUse, DisorderDisease, Identification "
                        "   FROM pubchemlite_categories"
                        "   INNER JOIN scored_spectra_meta ssm on pubchemlite_categories.molecule = ssm.molecule"
                        "   WHERE accession IS ?",
                        (acc, )
                    ).fetchone()

                # Get the number of isomers for the correct candidate
                n_isomers, = cands_db.db.execute(
                    " \
                     SELECT COUNT(DISTINCT m2.inchikey) FROM ( \
                        SELECT * FROM scored_spectra_meta INNER JOIN molecules m on m.cid = scored_spectra_meta.molecule \
                     ) t1 \
                      INNER JOIN molecules m2 on m2.inchikey1 = t1.inchikey1 \
                      WHERE accession is ? \
                    ", (acc, )
                ).fetchone()

                # Get the structures of the top-ranked candidates (do not need to be the correct ones)
                top_ranked_structure_msplrt = np.random.RandomState(cand["n_cand"]).choice(
                    np.array(cand["label"])[cand["score"] == np.max(cand["score"])]
                )
                top_ranked_structure_onlyms = np.random.RandomState(cand["n_cand"]).choice(
                    np.array(cand["label"])[ms_scores == np.max(ms_scores)]
                )

                smi_can_top_msplrt, smi_iso_top_msplrt = cands_db.db.execute(
                    "SELECT smiles_can, smiles_iso FROM molecules WHERE %s IS ?" % args.candidate_aggregation_identifier,
                    (top_ranked_structure_msplrt, )
                ).fetchone()

                smi_can_top_onlyms, smi_iso_top_onlyms = cands_db.db.execute(
                    "SELECT smiles_can, smiles_iso FROM molecules WHERE %s IS ?" % args.candidate_aggregation_identifier,
                    (top_ranked_structure_onlyms, )
                ).fetchone()

                # Does the correct structure has 3D information
                try:
                    has_3D = cand["correct_structure"].split("-")[1] != "UHFFFAOYSA"
                except IndexError:
                    has_3D = False

                df_out.append(
                    (
                        ds,                         # dataset
                        acc,                        # accession
                        sample_idx,                 # MS-tuple set random-sample ID
                        args.ms2scorer,             # MS2-scoring method
                        args.candidate_aggregation_identifier,   # Identifier used to distinguish molecules
                        cand["correct_structure"],  # Identifier of the correct molecular structure
                        has_3D,                     # Does the correct molecular structure encodes 3D information?
                        smi_iso_gt,                 # Isomeric SMILES
                        super_clf,                  # ClassyFire superclass-level
                        class_clf,                  # ClassyFire class-level
                        # ---
                        # PubChemLite class labels
                        agroChemInfo,
                        bioPathway,
                        drugMedicInfo,
                        foodRelated,
                        pharmacoInfo,
                        safetyInfo,
                        toxicityInfo,
                        knownUse,
                        disorderDisease,
                        identification,
                        1 if annoTypeCount == 0 else 0,
                        # ---
                        cand["spectrum_id"].get("retention_time"),  # Retention time
                        cand["n_cand"],             # Number of associated candidates
                        n_isomers,                  # Number of stereo-isomers
                        rank_onlyms,                # Rank of the correct structure using Only MS
                        rank_msplrt,                # -- " -- (MS + RT)
                        np.mean(rank_onlyms) - np.mean(rank_msplrt),  # Rank improvement of the correct structure (MS + RT over Only MS)
                        np.max(rank_onlyms) == 1,   # Correct structure is correctly ranked at top-1 (Only MS)
                        np.max(rank_msplrt) == 1,   # -- " -- (MS + RT)
                        top_ranked_structure_onlyms,
                        top_ranked_structure_msplrt,
                        smi_iso_top_onlyms,         # Isomeric SMILES of the top-ranked structure (Only MS)
                        smi_iso_top_msplrt,         # Isomeric SMILES of the top-ranked structure (MS + RT)
                    )
                )

        df_out = pd.DataFrame(
            df_out,
            columns=[
                "dataset",
                "spectrum",
                "sample_idx",
                "ms2scorer",
                "molecule_identifier",
                "correct_structure",
                "has_3D",
                "correct_structure_smiles_iso",
                "classyfire_superclass",
                "classyfire_class",
                "pubchemlite_agroChemInfo",
                "pubchemlite_bioPathway",
                "pubchemlite_drugMedicInfo",
                "pubchemlite_foodRelated",
                "pubchemlite_pharmacoInfo",
                "pubchemlite_safetyInfo",
                "pubchemlite_toxicityInfo",
                "pubchemlite_knownUse",
                "pubchemlite_disorderDisease",
                "pubchemlite_identification",
                "pubchemlite_noClassification",
                "retention_time",
                "n_cand",
                "n_isomers",
                "rank_onlyms",
                "rank_msplrt", "rank_improvement",
                "is_top1_correct_onlyms",
                "is_top1_correct_msplrt",
                "top_ranked_structure_onlyms",
                "top_ranked_structure_msplrt",
                "top_ranked_structure_onlyms_smiles_iso",
                "top_ranked_structure_msplrt_smiles_iso"
            ]
        ) \
            .to_csv(os.path.join(output_dir, "rank_improvements__csi.tsv"), index=False, sep="\t")

    sys.exit(0)
