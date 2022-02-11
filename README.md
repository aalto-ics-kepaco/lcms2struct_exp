# Data, experiments and results for the LC-MS²Struct

This repository contains the data, experiment scripts and results for the manuscript:
```bibtex
\article{TODO}
```
Use the citation information of you want to reference our work. 

## Directory structure

The repository is organized in different sub-directories who's content is described in the following.

### Data (```data```)

Directory containing our MassBank DB (see Methods "Training data generation using MassBank") and all scripts needed to 
get from ```data/massbank__2020.11__v0.6.1.sqlite``` to ```data/massbank_db.sqlite```. The former DB file is the 
direct output of [massbank2db Python package](https://github.com/bachi55/massbank2db) applied to the [2020.11 
MassBank](https://github.com/MassBank/MassBank-data/releases/tag/2020.11) release. It is added to the repository. The 
latter DB file contains all pre-computed MS² scores, candidate sets and molecular features. It is required for our 
experiments and will be available from **ZENODO LINK**. 

The directory furthermore includes the input files for all three MS² scoring tools (see Methods "Pre-computing the 
MS² matching scores"). 

For further details on the data pre-processing please refer the README file in ```data```.

### Miscellaneous scripts (```misc_scripts```) 

The directory contains various Jupyter notebooks for data inspection and MassBank database (DB) statistic extraction. 

#### Database statistic tables

The ```various_db_statistics.ipynb``` notebook can be used to (re-)generate tables in the manuscript:
- summarize the MassBank DB as well as our groupings
- extract meta information from our DB 
- compute the baseline MS² scorer performances for our MassBank subset


