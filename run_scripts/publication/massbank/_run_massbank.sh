#!/bin/bash

# -----------------------
# Ensure that all parameters are set

DEBUG=0
N_THREADS=$1
MOL_KERNEL=$2
EVAL_SET_ID=$3
SSVM_MODEL_INDEX=$4
N_TRAIN_SEQS=$5
N_EPOCHS=$6
BATCH_SIZE=$7
MS2SCORER=$8
LLOSS_MODE=$9
C_GRID=( "$(echo ${10} | tr "," "\n")" )  # convert to array
LOAD_OPT_C=${11}
SSVM_FLAVOR=${12}
MOL_FEATURES=${13}
MOL_IDENTIFIER=${14}
MAX_N_CAND_TRAIN=${15}
TRAINING_DATASET=${16}
EXP_VERSION=${17}
SSVM_LIBRARY_VERSION=${18}
CLUSTER_SYSTEM=${19}

#if [ -z "$N_THREADS" ] ; then
#  echo "[ERROR] Number of threads (N_THREADS) not specified."
#  exit 1
#fi
#
#if [ -z "$MOL_KERNEL" ] ; then
#  echo "[ERROR] Molecule kernel (MOL_KERNEL) not specified."
#  exit 1
#fi
#
#if [ -z "$EVAL_SET_ID" ] ; then
#  echo "[ERROR] Evaluation set ID (EVAL_SET_ID) not specified."
#  exit 1
#fi
#
#if [ -z "$SSVM_MODEL_INDEX" ] ; then
#  echo "[ERROR] SSMV model index (SSVM_MODEL_INDEX) not specified."
#  exit 1
#fi
#
#if [ -z "$N_TRAIN_SEQS" ] ; then
#  echo "[ERROR] Number of training sequences (N_TRAIN_SEQS) not specified."
#  exit 1
#fi
#
#if [ -z "$N_EPOCHS" ] ; then
#  echo "[ERROR] Number of training epochs (N_EPOCHS) not specified."
#  exit 1
#fi
#
#if [ -z "$BATCH_SIZE" ] ; then
#  echo "[ERROR] Batch-size (BATCH_SIZE) not specified."
#  exit 1
#fi
#
#if [ -z "$MS2SCORER" ] ; then
#  echo "[ERROR] MS2-scoring method (MS2SCORER) not specified."
#  exit 1
#fi
#
#if [ -z "$LLOSS_MODE" ] ; then
#  echo "[ERROR] Label-loss mode (LLOSS_MODE) not specified."
#  exit 1
#fi
#
#if [ -z "$C_GRID" ] ; then
#  echo "[ERROR] C-grid (C_GRID) not specified."
#  exit 1
#fi
#
#if [ -z "$LOAD_OPT_C" ] ; then
#  echo "[ERROR] Not specified whether the optimal C-parameter (LOAD_OPT_C) should be loaded."
#  exit 1
#fi
#
#if [ -z "$SSVM_FLAVOR" ] ; then
#  echo "[ERROR] SSVM flavor (SSVM_FLAVOR) not specified."
#  exit 1
#fi
#
#if [ -z "$MOL_FEATURES" ] ; then
#  echo "[ERROR] Molecule features (MOL_FEATURES) not specified."
#  exit 1
#fi
#
#if [ -z "$MOL_IDENTIFIER" ] ; then
#  echo "[ERROR] Molecule identifier (MOL_IDENTIFIER) not specified."
#  exit 1
#fi
#
#if [ -z "$MAX_N_CAND_TRAIN" ] ; then
#  echo "[ERROR] Maximum number of random candidate during training (MAX_N_CAND_TRAIN) not specified."
#  exit 1
#fi
#
#if [ -z "$TRAINING_DATASET" ] ; then
#  echo "[ERROR] Training dataset (TRAINING_DATASET) not specified."
#  exit 1
#fi

# -----------------------

# ----------------------------------------------
# Create temporary directories on computing node

if [[ "${CLUSTER_SYSTEM}" == "triton" ]] ; then
  LOCAL_SCRATCH="/dev/shm/$SLURM_JOB_ID"
  mkdir "$LOCAL_SCRATCH" || exit 1
fi

if [[ -z "${LOCAL_SCRATCH}" ]] ; then
  echo "[ERROR] Local scratch directory not defined on the cluster system '${CLUSTER_SYSTEM}'."
  exit 1
fi

# Local directories
LOCAL_DIR=${LOCAL_SCRATCH}  # base dir
LOCAL_DB_DIR=${LOCAL_SCRATCH}  # for SQLite DB
LOCAL_CONDA_DIR="$LOCAL_DIR/miniconda/"  # for conda environment
mkdir "$LOCAL_CONDA_DIR" || exit 1

if [[ "${CLUSTER_SYSTEM}" == "triton" ]] ; then
  # Set up trap to remove my results on exit from the local disk
  trap "rm -rf $LOCAL_DB_DIR; rm -rf $LOCAL_DIR; exit" TERM EXIT
fi
# On puhti (CSC) the local disc is automatically wiped

# ----------------------------------------------

# Set some multiprocessing parameters
N_JOBS=$(($SLURM_CPUS_PER_TASK/$N_THREADS))  # e.g. parallel sub-problems solved
echo "n_cpus=$SLURM_CPUS_PER_TASK, n_threads=$N_THREADS, n_jobs=$N_JOBS, mol_kernel=$MOL_KERNEL"

# Set up some paths
BASE_PROJECT_DIR="${SCRATCHHOME}/projects/"
PROJECT_DIR="$BASE_PROJECT_DIR/lcms2struct_experiments/"
SCRIPTPATH="$PROJECT_DIR/run_scripts/publication/massbank/run_with_gridsearch.py"
DB_DIR="$PROJECT_DIR/data/"
DB_FN="massbank.sqlite"

# ----------------------------
# Set up the conda environment

module load git

# Directory where our SSVM library will be stored
SSVM_LIB_DIR="$LOCAL_DIR/msms_rt_ssvm"

if [[ ${CLUSTER_SYSTEM} == "triton" ]] ; then
  module load miniconda

  # Which SSVM library version should be used?
  GIT_REPO_URL=git@github.com:aalto-ics-kepaco/msms_rt_ssvm.git
  VERSION_TAG=$(git ls-remote --tags "$GIT_REPO_URL" | cut -d'/' -f3 | grep "$SSVM_LIBRARY_VERSION" | sort --version-sort | tail -1 | sed "s/\^{}//g")

  # Activate conda
  eval "$(conda shell.bash hook)"
elif [[ ${CLUSTER_SYSTEM} == "puhti" ]] ; then
  # Activate conda
  eval "$(/projappl/${MYPROJECTID}/miniconda3/bin/conda shell.bash hook)"

  # Which SSVM library version should be used?
  GIT_REPO_URL=${SCRATCHHOME}/projects/msms_rt_ssvm
  VERSION_TAG=$(git --git-dir="${GIT_REPO_URL}/.git" tag --list | grep "$SSVM_LIBRARY_VERSION" | sort --version-sort | tail -1)
fi

echo "Repository URL: $GIT_REPO_URL"
echo "Version-tag: $VERSION_TAG"

# Clone the SSVM library from the master branch at github
git clone --branch "$VERSION_TAG" "$GIT_REPO_URL" "$SSVM_LIB_DIR"

# Create the conda environment based on the SSVM library's requirements
cd "$LOCAL_CONDA_DIR" || exit 1
conda create --clone msms_rt_ssvm__base --prefix msms_rt_ssvm__local
conda activate "$LOCAL_CONDA_DIR/msms_rt_ssvm__local"
cd - || exit 1

# Install SSVM library
pip install --no-deps "$SSVM_LIB_DIR"

# ----------------------------

# Set up path to output directory
OUTPUT_DIR="$(dirname $SCRIPTPATH | sed "s/run_scripts/results_raw/g")/ssvm_lib=${SSVM_LIBRARY_VERSION}__exp_ver=${EXP_VERSION}/"
mkdir -p "$OUTPUT_DIR" || exit 1
echo "Output dir: $OUTPUT_DIR"

# Copy the DB file to the node's local disk
cp -v "$DB_DIR/$DB_FN" "$LOCAL_DB_DIR" || exit 1

# Copy the run script to the local disk (fortifies it from changes)
LOCAL_SCRIPTPATH=$LOCAL_DIR/$(basename $SCRIPTPATH)
cp -v "$SCRIPTPATH" "$LOCAL_SCRIPTPATH" || exit 1

# ------------------
# Run the experiment
NUMBA_NUM_THREADS=$N_THREADS;OMP_NUM_THREADS=$N_THREADS;OPENBLAS_NUM_THREADS=$N_THREADS; \
    srun python "$LOCAL_SCRIPTPATH" \
    "$EVAL_SET_ID" "$SSVM_MODEL_INDEX" \
  --n_jobs="$N_JOBS" \
  --n_threads_test_prediction="$SLURM_CPUS_PER_TASK" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$OUTPUT_DIR" \
  --n_samples_train="$N_TRAIN_SEQS" \
  --n_epoch="$N_EPOCHS" \
  --batch_size="$BATCH_SIZE" \
  --mol_kernel="$MOL_KERNEL" \
  --ms2scorer="$MS2SCORER" \
  --lloss_fps_mode="$LLOSS_MODE" \
  --stepsize="linesearch" \
  --C_grid ${C_GRID[*]} \
  --debug="$DEBUG" \
  --load_optimal_C_from_tree_zero="$LOAD_OPT_C" \
  --ssvm_flavor="$SSVM_FLAVOR" \
  --mol_feat_retention_order="$MOL_FEATURES" \
  --molecule_identifier="$MOL_IDENTIFIER" \
  --max_n_train_candidates="$MAX_N_CAND_TRAIN" \
  --training_dataset="$TRAINING_DATASET"
# ------------------

# Export the conda environment.yml when the run was successful
conda env export --no-build > "$OUTPUT_DIR/conda_env__$SLURM_JOB_ID.yml"
