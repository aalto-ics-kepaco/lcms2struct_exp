# Experiments and result analysis scripts for the LC-MS²Struct

This repository contains the scripts to reproduce the experiments and analyse the results for the manuscript: 

**"Joint structural annotation of small molecules using liquid chromatography retention order and tandem mass spectrometry data"**,

*Eric Bach, Emma L. Schymanski and Juho Rousu*, 2022

## Reproducibility

If you wish to re-produce our results please follow the instructions given below. All our experiments where 
performed using Linux as operating system and Python (version 3.8 and 3.9 are supported). Other operating systems are 
not officially supported. Detailed software requirements are given alongside the reproducibility instructions.

### Manuscript figures

The raw outputs of the LC-MS²Struct for the experiments presented in the manuscript are available on Zenodo. Download 
both [tar-files](https://en.wikipedia.org/wiki/Tar_(computing)) and unpack them in the repository's root directory.

- **ALLDATA**: https://zenodo.org/record/6451016
- **ONLYSTEREO**: https://zenodo.org/record/6037629

Detailed instructions how to re-produce the figures of the manuscript can be found [here](results_processed/publication/massbank/ssvm_lib=v2__exp_ver=4/README.md). 

### Re-run the experiments

If you wish to re-run our experiments, you will have to install the [LC-MS²Struct library](https://github.com/aalto-ics-kepaco/msms_rt_ssvm), 
which provides an implementation of the presented Structured Support Vector Machine (SSVM). Furthermore, you will 
have to download our MassBank database containing all features, candidate sets, etc. from [Zenodo](https://zenodo.org/record/5854661).
Subsequently, follow the instructions given [here](run_scripts/README.md). 

### Re-building the MassBank DB

If you wish to re-build the MassBank DB used in our experiments, please follow the instructions given [here](data/README.md)

## Repository structure

The repository is organized in different subdirectories those content is described in the following.

- ```misc_scripts```: Contains a script to extract different MassBank database statistics
- ```run_scripts```: Contains all scripts needed to re-run our experiments
- ```results_raw```: Contains the "raw" result files as produces by our experimental scripts (download from Zenodo)
- ```results_processed```: Contains the result files as used to produce the figures in our manuscript, e.g. 
  aggregated max-margins, comparison methods, ... (download from Zenodo)
- 

# Citation information

Use the citation information of you want to reference our work:

```bibtex
\article{TODO}
```
