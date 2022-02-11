#!/bin/bash

# -----------------------
# Ensure that all parameters are set

DEBUG=0
N_THREADS=${1}
EVAL_SET_ID=${2}
MS2SCORER=${3}
MOL_FEATURES=${4}
MOL_IDENTIFIER=${5}
C_GRID=( "$(echo ${6} | tr "," "\n")" )  # convert to array
BETA_GRID=( "$(echo ${7} | tr "," "\n")" )  # convert to array
N_TREES_FOR_SCORING=${8}
MAX_N_CAND_TRAIN=${9}
N_TUPLES_SEQUENCES_RANKING=${10}
EXP_VERSION=${11}
SSVM_LIBRARY_VERSION=${12}
USE_RAM_DISK=${13}
N_JOBS_SCORING_EVAL=${14}
COMPARISON_METHOD=${15}

# -----------------------

# ----------------------------------------------
# Create temporary directories on computing node

# For the database
LOCAL_DB_DIR="/dev/shm/$SLURM_JOB_ID"
mkdir "$LOCAL_DB_DIR" || exit 1

# For conda environment
if [ "$USE_RAM_DISK" -eq 1 ] ; then
  LOCAL_DIR="/dev/shm/$SLURM_JOB_ID"
else
  LOCAL_DIR="/tmp/$SLURM_JOB_ID"
fi
# LOCAL_DB_DIR="$LOCAL_DIR"
LOCAL_CONDA_DIR="$LOCAL_DIR/miniconda/"
mkdir -p "$LOCAL_CONDA_DIR" || exit 1

# Set up trap to remove my results on exit from the local disk
trap "rm -rf $LOCAL_DB_DIR; rm -rf $LOCAL_DIR; exit" TERM EXIT

# ----------------------------------------------

# Which SSVM library version should be used?
module load git

GIT_REPO_URL=git@github.com:aalto-ics-kepaco/msms_rt_ssvm.git
VERSION_TAG=$(git ls-remote --tags "$GIT_REPO_URL" | cut -d'/' -f3 | grep "$SSVM_LIBRARY_VERSION" | sort --version-sort | tail -1 | sed "s/\^{}//g")
echo "Repository URL: $GIT_REPO_URL"
echo "Version-tag: $VERSION_TAG"

# Set some multiprocessing parameters
N_JOBS=$(($SLURM_CPUS_PER_TASK/$N_THREADS))  # e.g. parallel sub-problems solved
echo "n_cpus=$SLURM_CPUS_PER_TASK, n_threads=$N_THREADS, n_jobs=$N_JOBS, n_jobs_scoring_eval=$N_JOBS_SCORING_EVAL"

# Set up some paths
BASE_PROJECT_DIR="/scratch/cs/kepaco/bache1/projects/"
PROJECT_DIR="$BASE_PROJECT_DIR/lcms2struct_experiments/"
case $COMPARISON_METHOD in
  "ruttkies2016_xlogp3")
    SCRIPTPATH="$PROJECT_DIR/run_scripts/comparison/massbank/run_Ruttkies2016_xlogp3.py"
    ;;
  "bach2020_ranksvm")
    SCRIPTPATH="$PROJECT_DIR/run_scripts/comparison/massbank/run_Bach2020_ranksvm.py"
    ;;
  "rt_prediction_svr")
    SCRIPTPATH="$PROJECT_DIR/run_scripts/comparison/massbank/run_rt_prediction_svr.py"
    ;;
  *)
    echo "[ERROR] Invalid comparison method: $COMPARISON_METHOD."
    exit 1
esac
__PPATH="$PROJECT_DIR/run_scripts/publication/massbank/"  # allow import of some functions from the SSVM evaluation
DB_DIR="$PROJECT_DIR/data/"
DB_FN="massbank.sqlite"

# ----------------------------
# Set up the conda environment

module load miniconda

# Clone the SSVM library from the master branch
SSVM_LIB_DIR="$LOCAL_DIR/msms_rt_ssvm"
git clone --branch "$VERSION_TAG" "$GIT_REPO_URL" "$SSVM_LIB_DIR"

# Create the conda environment based on the SSVM library's requirements
eval "$(conda shell.bash hook)"
cd "$LOCAL_CONDA_DIR" || exit 1
conda create --clone msms_rt_ssvm__base --prefix msms_rt_ssvm__local
conda activate "$LOCAL_CONDA_DIR/msms_rt_ssvm__local"
cd - || exit 1

# Install SSVM library
pip install --no-deps "$SSVM_LIB_DIR"

# ----------------------------

# Set up path to output directory
OUTPUT_DIR="$(dirname $SCRIPTPATH | sed "s/run_scripts/results_processed/g")__exp_ver=${EXP_VERSION}/"
mkdir -p "$OUTPUT_DIR" || exit 1
echo "Output dir: $OUTPUT_DIR"

# Copy the DB file to the node's local disk
cp -v "$DB_DIR/$DB_FN" "$LOCAL_DB_DIR" || exit 1

# Copy the run script to the local disk (fortifies it from changes)
LOCAL_SCRIPTPATH=$LOCAL_DIR/$(basename $SCRIPTPATH)
cp -v "$SCRIPTPATH" "$LOCAL_SCRIPTPATH" || exit 1

# ------------------
# Run the experiment
NUMBA_NUM_THREADS=$N_THREADS;OMP_NUM_THREADS=$N_THREADS;OPENBLAS_NUM_THREADS=$N_THREADS;PYTHONPATH=$__PPATH \
    srun python "$LOCAL_SCRIPTPATH" \
    "$EVAL_SET_ID" \
  --n_jobs="$N_JOBS" \
  --n_jobs_scoring_eval="$N_JOBS_SCORING_EVAL" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$OUTPUT_DIR" \
  --n_trees_for_scoring="$N_TREES_FOR_SCORING" \
  --ms2scorer="$MS2SCORER" \
  --C_grid ${C_GRID[*]} \
  --beta_grid ${BETA_GRID[*]} \
  --debug="$DEBUG" \
  --molecule_features="$MOL_FEATURES" \
  --molecule_identifier="$MOL_IDENTIFIER" \
  --max_n_candidates_training="$MAX_N_CAND_TRAIN" \
  --n_tuples_sequences_ranking="$N_TUPLES_SEQUENCES_RANKING" \
  --no_plot
# ------------------
