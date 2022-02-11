# Data used in "Joint structural annotation of small molecules using liquid chromatography retention order and tandem mass spectrometry data"

The experimental data including:

- (aggregated) MassBank records
- Molecular candidate sets
- Molecule features (e.g. FCFP fingerprints)
- MS² matching scores between MassBank spectra and corresponding candidates
- Data splits used in evaluation
- and further meta data ...

is stored in the SQLite database (DB) file provided in this repository. In the following we provide a brief summary 
of the content of each SQLite table in the DB.

## General remarks

- This documentation is for the **massbank.sqlite.gz** file.
- Molecules in the DB are identified by their PubChem ID (cid) across all tables.
- If you have any questions regarding the data, please contact: **eric.bach [at] aalto.fi**.

## Table descriptions

### "candidate_spectra"

Association table of the internal spectrum identifier (e.g. AC01111385) and molecular candidates identified by their 
Pubchem IDs (cid). The molecular candidate sets where generated with the [SIRIUS software](https://bio.informatik.uni-jena.de/software/sirius/)
using the ground truth molecular formula of each spectrum. 

### "classyfire_classes"

[Classyfire](http://classyfire.wishartlab.com/) molecule classification for each ground truth structure associated 
with the MassBank records in this DB. The classification has not been performed for all molecular candidate 
structures.

### "datasets"

Meta information for all datasets (in the paper referred as "MassBank groups") used in our evaluation. The table, 
for example, specifies the experimental conditions of the LC-MS² setup.

### "descriptors_data__bouwmeester__smiles_iso"

Molecular descriptor as used by [Bouwmeester et al. (2019)](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.8b05820) 
for each molecular candidate structure. 

### "descriptors_meta"

Meta information for all molecular descriptors provided in the DB.

### "fingerprints_data__FCFP__binary__all__2D"

FCFP fingerprints for molecular candidate structures in their *binarized* format. Binarized here means that the 
counting fingerprints where converted into an equivalent binary format (see [Ralaivola et al. (2005)](https://www.sciencedirect.com/science/article/pii/S0893608005001693))
allowing for faster kernel computation in practice. The fingerprints where computed based on the isomeric SMILES 
representation of the molecules. In RDKit the option "use_chirality" was set to False. 

### "fingerprints_data__FCFP__binary__all__3D"

Same as above, but the "use_chirality" option was set to True.

### "fingerprints_data__FCFP__count__all__2D"

Explicit integer vector representation of "fingerprints_data__FCFP__binary__all__2D".

### "fingerprints_data__FCFP__count__all__3D"

Explicit integer vector representation of "fingerprints_data__FCFP__binary__all__3D".

### "fingerprints_data__substructure_count__smiles_iso"

Substructure counting fingerprint vectors (integer vector representation) as used by [Bach et al. (2020)](https://academic.oup.com/bioinformatics/article/37/12/1724/6007259).
The fingerprints where calculated using CDK and ismomeric SMILES.

### "fingerprints_meta"

Meta information for all molecular fingerprints provided in the DB.

### "lcms_data_splits"

Assignment of the spectra to the evaluation splits. Each row stores the spectrum id, the corresponding dataset, the 
evaluation set id (staring with 0 within each dataset) and the experiment type. Latter ones is referred in the paper 
as ONLYSTEREO and FULLDATA. 

### "merged_accessions"

As described in the paper, we merge different MassBank records into one record for internal use (e.g. "AC000298, 
AC000299, AC000300, AC000301, AC000302 --> AC01111385"). The original records correspond to measurements of with 
different collision energies in MassBank. For the purpose of MS² score prediction, we group those collision energies 
and feed them jointly to the different MS² scoring methods. This table contains the information which orginal record 
ids where grouped and assigned a new id. 

### "molecules"

Table containing all molecular structures used for our experiments, i.e. candidates and ground truth structures. 
Each molecule is identified by its PubChem cid and multiple structure representations and further properties are 
stored for each structure. 

### "pubchemlite_categories"

[PubChemLite](https://zenodo.org/record/4183801/) molecule classification for each ground truth structure associated 
with the MassBank records in this DB. The classification has not been performed for all molecular candidate 
structures.

### "scored_spectra_meta"

Meta information for each merged MassBank record used in our experiments (e.g. its ground truth molecular structure).
We only considered records that could be scored by the SIRIUS software, hence the term "scored_".

### "scoring_methods"

Meta information about the three (3) MS² scoring methods used in our experiments.

### "spectra_candidate_scores"

Association table assigning to each candidate and spectrum the MS² score of each method. Scores are normalized to [0,
1] *within* each candidate set. Please note that for the paper we slightly modify the scores to be within (0,1].

### "spectra_meta"

Meta information for each *original* MassBank record. This table contains a lot more information than the other meta 
information table for the merged spectra. 

### "spectra_peaks"

Spectra peaks and intensities for the original MassBank records. You can recover the measured MS² spectra from here. 

### "spectra_rts"

Measured retention times for each original spectrum. 
