# Script to run the experiments presented in our manuscript

The scripts are contained in the two directories:

- ```publication```: Experiments (mainly model training) with the [LC-MS²Struct SSVM model](https://github.com/aalto-ics-kepaco/msms_rt_ssvm)
- ```comparison```: Experiments with the comparison methods (e.g. RT filtering, XLogP3, ...)

There is a subdirectory ```massbank``` for each setting which contains the run-scripts.

## LC-MS²Struct experiments

All experiments with the LC-MS²Struct implementation can be performed with the script [```run_with_gridsearch.py```](publication/massbank/run_with_gridsearch.py). 
The input parameters are specified in [```load_experiment_parameter.sh```](publication/massbank/load_experiment_parameters.sh) (for the manuscript experiment version 4 was used). For each conducted experiment the complete list of command line interface parameters can be found, for example,
[here](example_parameters_ALLDATA.list) (ALLDATA setting) or [here](example_parameters_ONLYSTEREO.list) (ONLYSTEREO).
Such parameter files are outputted by each experiment can be found alongside the results in ```results_raw``` folder 
at the root of this repository (please note that for that you need to download the result files from the Zenodo: 
[ALLDATA](https://zenodo.org/record/6451016) and [ONLYSTEREO](https://zenodo.org/record/6037629)).

### Instructions to re-run our experiments

For our manuscript, we ran the experiments on a cluster. If you want to reproduce the full setting, then this can be 
done by adapting the [```run_massbank__with_deps.sh```](publication/massbank/run_massbank__with_deps.sh) to your cluster. This script trains eight (8) SSVM models for each
evaluation set ([```eval_set_id```](publication/massbank/run_with_gridsearch.py#L75)). In total this are about 350 jobs for ALLDATA and about 100 jobs for ONLYSTEREO. 
The script furthermore produces the files in ```results_raw```, which we uploaded to Zenodo:

- Predicted max-marginal scores for each MassBank-subset and sampled LC-MS² dataset  
- Experimental parameters
- List of spectra ids used for evaluation
- ...

You can compare your output to files on Zenodo. 

### Small example to test our implementation

Here we will give a minimal example, which can serve as a starting point to re-produce our experiments. If you want to 
apply a trained LC-MS²Struct model to new data, you can train a model as done in [```run_with_gridsearch.py```](publication/massbank/run_with_gridsearch.py), but you need to implement your [own candidate set wrapper](https://github.com/aalto-ics-kepaco/msms_rt_ssvm/blob/master/ssvm/README.md#own-candidate-db-wrappers). 

To run our minimal example, please follow the following instructions. We assume that you are at the root of this 
repository. Please note that all code was developed and tested in a **Linux** environment. Other operating systems are 
not supported.

1) Install the [LC-MS²Struct](https://github.com/aalto-ics-kepaco/msms_rt_ssvm) described in the repository
2) Make sure that the conda environment you installed LC-MS²Struct into is activated
3) Download our [massbank.sqlite.gz](https://zenodo.org/record/5854661) database file and unpack it:
```bash
gunzip /path/to/your/massbank.sqlite.gz

## Will produce the file: /path/to/your/massbank__with_pubchemlite.sqlite
```
4) Change to directory containing the run-script: 
```bash
cd run_scripts/publication/massbank
```
5) Run the training and max-margin prediction for **two** SSVM model:
```bash
NUMBA_NUM_THREADS=2;OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2;

python run_with_gridsearch.py 0 0 \
  --n_jobs=2 \
  --n_threads_test_prediction=4 \
  --debug=1 \
  --n_samples_train=128 \
  --max_n_train_candidates=25 \
  --ms2scorer=cfmid4__norm \
  --C_grid 64 \
  --n_epochs=2 \
  --db_fn=/path/to/your/massbank__with_pubchemlite.sqlite \
  --training_dataset=massbank
```
and 
```bash
NUMBA_NUM_THREADS=2;OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2;

python run_with_gridsearch.py 0 1 \
  --n_jobs=2 \
  --n_threads_test_prediction=4 \
  --debug=1 \
  --n_samples_train=128 \
  --max_n_train_candidates=25 \
  --ms2scorer=cfmid4__norm \
  --C_grid 64 \
  --n_epochs=2 \
  --db_fn=/path/to/your/massbank__with_pubchemlite.sqlite \
  --training_dataset=massbank
```

Running this scripts will take about 25 minutes on a modern 4 core machine. You might need to adapt the 
```*_NUM_THREADS``` variables and [```n_jobs```](publication/massbank/run_with_gridsearch.py#L89) setting with 
respect to the actual number of your **physical** cores, i.e. ```*_NUM_THREADS``` x ```n_jobs``` = number of physical 
cores available. Please also have a look on the cluster scripts ([```run_massbank__with_deps.sh```](publication/massbank/run_massbank__with_deps.sh) and [```_run_massbank.sh```](publication/massbank/_run_massbank.sh))
to see we configured the thread settings given a specific amount of resources.

6) The output of the above runs is stored in ```debugging/massbank```
```bash
ls debugging/massbank

## Expected output
# debug__ds=AC_003__lloss_mode=mol_feat_fps__mol_feat=FCFP__binary__all__3D__mol_id=cid__ms2scorer=cfmid4__norm__ssvm_flavor=default
```

7) The max-marginals are stored in the respective SSVM models' subdirectories (```ssvm_model_idx```). The aggregated 
   margins can be computed using:
```bash
python combine_margins_from_different_ssvm_models.py \ 
  /path/to/this/repository/run_scripts/publication/massbank/debugging/massbank \
  cfmid4__norm \
  --candidate_aggregation_identifier=inchikey1 \
  --write_out_averaged_margins \
  --debug
```
   Note that you need to give the full path to the location of the max-marginals' base-directory. 

8) After running the above command the directory structure will look like:
```bash
debugging/
└── massbank
    └── debug__ds=AC_003__lloss_mode=mol_feat_fps__mol_feat=FCFP__binary__all__3D__mol_id=cid__ms2scorer=cfmid4__norm__ssvm_flavor=default
        ├── combined__cand_agg_id=inchikey1__marg_agg_fun=average
        │   ├── marginals__spl=0.pkl.gz
        │   ├── top_k__max_models.tsv
        │   └── top_k.tsv
        ├── ssvm_model_idx=0
        │   ├── eval_spec_ids__spl=0.list
        │   ├── grid_search_results__spl=0.tsv
        │   ├── marginals__spl=0.pkl.gz
        │   ├── parameters__spl=0.list
        │   ├── top_k__spl=0.tsv
        │   └── train_stats__spl=0.tsv
        └── ssvm_model_idx=1
            ├── eval_spec_ids__spl=0.list
            ├── grid_search_results__spl=0.tsv
            ├── marginals__spl=0.pkl.gz
            ├── parameters__spl=0.list
            ├── top_k__spl=0.tsv
            └── train_stats__spl=0.tsv
```
   Here, ```combined``` refers to the [aggregated max-margins](publication/massbank/combine_margins_from_different_ssvm_models.py#L44) (averaged across the SSVM models) and the candidate scores
   being collapsed based on ```cand_agg_id```. The top-k accuracies (comparing Only MS² and LC-MS²Struct (MS + RT)) 
   performance are stored as well (```top_k.tsv```), whereby all evaluation subsets for the particular MassBank 
   subset are stored in a single file.

## Comparison

Scripts containing the implementations of the methods we compare with in our manuscript.  

