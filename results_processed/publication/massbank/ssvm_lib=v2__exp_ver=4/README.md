# Jupyter notebooks to generate manuscript figures

Here you can find the jupyter notebooks used to produce the figures in our manuscript. The relevant notebooks for 
the latest version of our manuscript are:

| **Notebook name** | **Content**                                                                                                                                                                                                                    |
| --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```NEW__exp_04__using_3D_features_for_inchikey1_aggregation.ipynb``` | Here we compare the ranking performance in the ALLDATA setting when using 2D and 3D FCFP fingerprints in combination with LC-MS²Struct. The generated figures can be also found in the supplementary material of the manuscript. |
| ```NEW__exp_05__comparison_of_different_score_integration_approaches_3D.ipynb``` | Here we compare the ranking performance of LC-MS²Struct with the different comparison methods.                                                                                                                                 |
| ```NEW__exp_06__performance_analysis_using_molecule_classification_3D.ipynb``` | Here we analyze the performance improvements using LC-MS²Struct for different ClassyFire and PubChemLite molecular classes.                                                                                                    | 
| ```exp_03__stereochemistry.ipynb```| Here we assess the performance of LC-MS²Struct in the ONLYSTEREO setting.                                                                                                                                                      |

## Instructions to reproduce the figures

You can reproduce figures in our manuscript simply running the aforementioned notebooks using the instructions below.

1) Make sure you downloaded the result archives from Zenodo ([ALLDATA](https://zenodo.org/record/6451016) and 
   [ONLYSTEREO](https://zenodo.org/record/6037629)) and extracted them in the root directory of this repository.
2) Create a conda environment: 
```bash
conda env create -f environment.yml
conda activate lcms2struct_manuscript_figures
```
3) Make the environment available in your JupyterLab
```bash
python -m ipykernel install --user --name=lcms2struct_manuscript_figures
```
4) Install the figure helper-tools
```bash
cd ../../../../

pwd
## Expected output is the git-repository's root-directory 

pip install .
```
5) Change back to the notebooks' directory and start JupyterLab
```bash
cd results_processed/publication/massbank/ssvm_lib=v2__exp_ver=4/

jupyter lab
```
6) Choose any notebook and execute it. Ensure that it is running on the kernel corresponding to the conda environment.