####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
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
import argparse
import numpy as np
import more_itertools as mit

from typing import Callable, List, Tuple
from joblib.parallel import Parallel, delayed
from sklearn.impute import SimpleImputer

from rdkit.Chem import Descriptors
from rdkit import __version__ as rdkit_version

from rosvm.feature_extraction.featurizer_cls import FeaturizerMixin
from rosvm import __version__ as rosvm_version

from utils import get_backup_db


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


def filter_descriptors(l_rdkit_desc) -> List[Tuple[str, Callable]]:
    """
    Only keep the descriptors used by Bouwmeester et al. (2019)
    """
    return [(dname, dfun) for dname, dfun in l_rdkit_desc if dname in BOUWMEESTER_DESCRIPTOR_SET]


def get_descriptors(data_batch):
    # Get the list of descriptors available in RDKit
    l_rdkit_desc = sorted(Descriptors.descList)

    # Filter the descriptors used by Bouwmeester et al. (2019)
    l_rdkit_desc = filter_descriptors(l_rdkit_desc)

    # Matrix storing all descriptor values
    X = np.zeros((len(data_batch), len(l_rdkit_desc)))

    cids = []
    for idx, (cid, smi) in enumerate(data_batch):
        # Get RDKit mol-objects
        mol = FeaturizerMixin.sanitize_mol(smi)

        # Compute descriptors
        for jdx, (_, dfun) in enumerate(l_rdkit_desc):
            X[idx, jdx] = dfun(mol)

        cids.append(cid)

    # Find and replace np.inf - values
    X[np.isinf(X)] = np.nan

    # How many molecules did have problems with the descriptor computation?
    n_x_with_nan = np.sum(np.any(np.isnan(X), axis=1)).item()
    print("%d / %d molecules with nan-descriptors." % (n_x_with_nan, len(data_batch)))

    # Impute missing values
    if n_x_with_nan > 0:
        X = SimpleImputer(copy=False).fit_transform(X)

    # Convert descriptor matrix to list of strings
    dvals = [",".join(x_i) for x_i in X.astype(str)]

    return cids, dvals


if __name__ == "__main__":
    # Read CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("massbank_db_fn", help="Filepath of the Massbank database.")
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=2**14,
        help="Size of the batches in which the descriptors are computed and inserted to the DB."
    )
    arg_parser.add_argument("--n_jobs", type=int, default=4)
    arg_parser.add_argument("--molecule_representation", type=str, default="smiles_iso", choices=["smiles_iso"])
    arg_parser.add_argument(
        "--backup_db_exists", type=str, default="raise", choices=["overwrite", "raise", "reuse"]
    )
    args = arg_parser.parse_args()

    # Open connection to database
    # conn = sqlite3.connect(args.massbank_db_fn)
    conn = get_backup_db(args.massbank_db_fn, exists=args.backup_db_exists, postfix="with_descriptors")
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

        # Create meta and data tables
        with conn:
            desc_name = "bouwmeester__%s" % args.molecule_representation

            conn.execute(
                "CREATE TABLE IF NOT EXISTS descriptors_meta("
                "   name                    VARCHAR NOT NULL PRIMARY KEY,"
                "   molecule_representation VARCHAR NOT NULL,"
                "   length                  INTEGER NOT NULL,"
                "   mode                    VARCHAR NOT NULL,"
                "   timestamp               VARCHAR NOT NULL,"
                "   library                 VARCHAR NOT NULL,"
                "   used_descriptors        VARCHAR NOT NULL"
                ")"
            )

            conn.execute(
                "CREATE TABLE IF NOT EXISTS descriptors_data__%s("
                "   molecule    INTEGER NOT NULL PRIMARY KEY,"
                "   desc_vals   VARCHAR NOT NULL,"
                "   FOREIGN KEY (molecule) REFERENCES molecules(cid)"
                ")" % desc_name
            )

            # Get the list of descriptors available in RDKit
            l_rdkit_desc = filter_descriptors(sorted(Descriptors.descList))
            conn.execute(
                "INSERT OR REPLACE INTO descriptors_meta "
                "   VALUES ('%s', '%s', %d, 'real', DATETIME('now', 'localtime'), 'rosvm: %s, RDKit: %s', '%s')"
                % (
                    desc_name, args.molecule_representation, len(l_rdkit_desc), rosvm_version, rdkit_version,
                    ",".join([name for name, _ in l_rdkit_desc])
                )
            )

        # Load all molecular descriptors
        print("Load molecules ...")
        data = conn.execute("SELECT cid, %s FROM molecules" % args.molecule_representation).fetchall()

        # Shuffle data in-place to avoid biases in the nan-value imputation
        np.random.RandomState(1989).shuffle(data)

        # Compute descriptors in batches
        print("Compute descriptors ...")
        batches = list(mit.chunked(data, args.batch_size))
        res = Parallel(n_jobs=args.n_jobs, verbose=11)(delayed(get_descriptors)(batch) for batch in batches)

        # Insert descriptors
        print("Insert descriptors ...")
        with conn:
            for cids, dvals in res:
                conn.executemany(
                    "INSERT OR REPLACE INTO descriptors_data__%s VALUES (?, ?)" % desc_name,
                    zip(cids, dvals)
                )

    finally:
        conn.close()
