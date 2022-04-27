# Experiments and result analysis scripts for the LC-MS²Struct

This repository contains the scripts to reproduce the experiments and analyse the results for the manuscript: 

**"Joint structural annotation of small molecules using liquid chromatography retention order and tandem mass spectrometry data"**,

*Eric Bach, Emma L. Schymanski and Juho Rousu*, 2022

## Outputs used to produce the results in the manuscript

The raw outputs of the LC-MS²Struct for the experiments presented in the manuscript are available on Zenodo:

- **ALLDATA**: https://zenodo.org/record/6451016
- **ONLYSTEREO**: https://zenodo.org/record/6037629

Download the tar-files and unpack them in the repository.

## LC-MS²Struct library

The library implementing the Structured Support Vector Machine (SSVM) model working in the backend of LC-MS²Struct 
can be found [here](https://github.com/aalto-ics-kepaco/msms_rt_ssvm). 

## Directory structure

The repository is organized in different sub-directories who's content is described in the following.

### Data (```data```)

Directory containing our MassBank DB (see Methods "Training data generation using MassBank") and all scripts needed to 
get from ```data/massbank__2020.11__v0.6.1.sqlite``` to ```data/massbank_db.sqlite```. The former DB file is the 
direct output of [massbank2db Python package](https://github.com/bachi55/massbank2db) applied to the [2020.11 
MassBank](https://github.com/MassBank/MassBank-data/releases/tag/2020.11) release. It is added to the repository. The 
latter DB file contains all pre-computed MS² scores, candidate sets, molecular features, etc. It is required to 
re-run the experiments. All databases and further data files can be downloaded from Zenodo: https://zenodo.org/record/5854661. 
For further details on the data pre-processing please refer the README file in ```data```.

### Miscellaneous scripts (```misc_scripts```) 

The directory contains various Jupyter notebooks for data inspection and MassBank database (DB) statistic extraction. 

#### Database statistic tables

The ```various_db_statistics.ipynb``` notebook can be used to (re-)generate tables in the manuscript:
- summarize the MassBank DB as well as our groupings
- extract meta information from our DB 
- compute the baseline MS² scorer performances for our MassBank subset

# Citation information

Use the citation information of you want to reference our work:

```bibtex
\article{TODO}
```
