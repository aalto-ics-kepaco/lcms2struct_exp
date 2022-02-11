#!/bin/bash

# Check whether we are running on Triton or Puhti (CSC) and set required resources accordingly
case "$HOSTNAME" in
  *triton*)
    # We are on Triton
    CPUS_PER_TASK=24
    N_THREADS=3
    TIME_BASE_MODEL="22:00:00"
    TIME_OTHER_MODEL="08:00:00"
    NODE_FEATURES=""  # on triton we use the ram-disk for the SQLite DB. No need to request any cluster features.
    ACCOUNT=""  # on triton we do not need an account

    MEM_PER_CPU__NORMAL=5000
    MEM_PER_CPU__LARGE=7000

    CLUSTER_SYSTEM="triton"
    ;;
  *puhti*)
    # We are CSC's Puhti cluster
    CPUS_PER_TASK=20  # nodes have 40 cores
    N_THREADS=2
    TIME_BASE_MODEL="22:00:00"
    TIME_OTHER_MODEL="08:00:00"
    NODE_FEATURES="--gres=nvme:40"  # we reserve 40GB of local storage while working on puhti to store the SQLite DB.
    ACCOUNT="--account=${MYPROJECTID}"

    MEM_PER_CPU__NORMAL=3500
    MEM_PER_CPU__LARGE=6300

    CLUSTER_SYSTEM="puhti"
    ;;
  *)
    echo "[ERROR] Unsupported cluster system: hostname='${HOSTNAME}'."
    exit 1
esac

# Some parameters of the experiment
EXP_VERSION=4
SSVM_LIBRARY_VERSION=v2

MOL_FEATURES_SHORT="fcfp_2D"
MS2SCORER_SHORT="cfmid4"
N_SSVM_MODELS=8
MOL_IDENTIFIER="cid"
TRAINING_DATASET="massbank"
SSVM_FLAVOR="default"

# Resolve short parameter names
# E.g. "metfrag" --> "metfrag__norm"
source ./resolve_short_parameter_names.sh $MS2SCORER_SHORT $MOL_FEATURES_SHORT
if [ -z "$MS2SCORER" ] || [ -z "$MOL_FEATURES" ] ; then
  echo "[ERROR] MS2SCORER or MOL_FEATURES is not defined"
  exit 1
else
  echo "${MS2SCORER_SHORT} --> ${MS2SCORER}"
  echo "${MOL_FEATURES_SHORT} --> ${MOL_FEATURES}"
fi

# Get kernel and label-loss setup
source ./get_kernel_and_label_loss.sh "$MOL_FEATURES"
if [ -z "$MOL_KERNEL" ] || [ -z "$LLOSS_MODE" ] ; then
  echo "[ERROR] MOL_KERNEL or LLOSS_MODE is not defined"
  exit 1
else
  echo "MOL_KERNEL: ${MOL_KERNEL}"
  echo "LLOSS_MODE: ${LLOSS_MODE}"
fi

# Load some experiment parameters depending on the experiment version
source ./load_experiment_parameters.sh "$EXP_VERSION" "$MS2SCORER"
if [ -z "$BATCH_SIZE" ] || [ -z "$C_GRID" ] || [ -z "$LOAD_OPT_C" ] || [ -z "$N_EPOCHS" ] || [ -z "$N_TRAIN_SEQS" ] || [ -z "$MAX_N_CAND_TRAIN" ] ; then
  echo "[ERROR] BATCH_SIZE, C_GRID, LOAD_OPT_C, N_EPOCHS, N_TRAIN_SEQS or MAX_N_CAND_TRAIN is not defined."
  exit 1
else
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "C_GRID: ${C_GRID}"
  echo "LOAD_OPT_C: ${LOAD_OPT_C}"
  echo "N_EPOCHS: ${N_EPOCHS}"
  echo "N_TRAIN_SEQS: ${N_TRAIN_SEQS}"
  echo "MAX_N_CAND_TRAIN: ${MAX_N_CAND_TRAIN}"
  echo "MOL_IDENTIFIER: ${MOL_IDENTIFIER}"
fi

if [ "$MOL_FEATURES_SHORT" = "fcfp_3D" ] ; then
  __MOL_FEAT_STEREO="3D"
elif [ "$MOL_FEATURES_SHORT" = "fcfp_2D" ] ; then
  __MOL_FEAT_STEREO="2D"
fi

# Depending on the training dataset load the number of samples per dataset
case $TRAINING_DATASET in
  "massbank")
    N_SAMPLES=(15 15 1 14 15 15 15 15 1 15 15 6 6 15 15 15 15 1 1 6 7 11 15 14 18 1 15 15 7 15 15 6)  # 32 datasets
    ;;
  "massbank__with_stereo")
    N_SAMPLES=(15 0 0 15 0 0 1 15 0 0 0 1 1 0 0 1 0 0 0 5 1 15 1 0 15 1 1 0 1 0 0 5)  # 32 datasets
    ;;
  *)
    echo "[ERROR] Invalid training dataset: ${TRAINING_DATASET}. Choices are 'massbank' and 'massbank__with_stereo'"
    exit 1
esac

N_DATASETS=${#N_SAMPLES[@]}  # Number of datasets

# Sleep for a couple of seconds. This gives us the change to double check the specified parameters.
sleep 3s

# Start the jobs for all datasets
# for (( i=0; i<N_DATASETS; i++ ))
for (( i=0; i<1; i++ ))
do
  echo "Submit dataset '$i' with '${N_SAMPLES[$i]}' samples."

  # We skip the HILIC dataset
  if [ "$i" -eq 2 ] ; then
    continue
  fi

  # From experience we know that some datasets require more memory. However, to not restrict the computation nodes
  # we only increase the memory for the respective datasets.
  if [ "$i" -eq 17 ] ; then
    MEM_PER_CPU=${MEM_PER_CPU__LARGE}
  else
    MEM_PER_CPU=${MEM_PER_CPU__NORMAL}
  fi

  for (( j=0; j<${N_SAMPLES[$i]}; j++))
  do
    EVAL_SET_ID=$(printf "%02d%02d" $i $j)
    echo "Evaluation set id: $EVAL_SET_ID"

    # We need to check the model-0 availability for each sample separately
    __MODEL_ZERO_AVAILABLE__=0

    for (( k=0; k<N_SSVM_MODELS; k++ ))
    do
      # Check whether the results for the current setup already exists and, if, go to the next setup.
      source ./check_whether_result_exists.sh "$EVAL_SET_ID" "$k" "$MS2SCORER" "$LLOSS_MODE" "$MOL_FEATURES" "$SSVM_FLAVOR" "$MOL_IDENTIFIER" "$TRAINING_DATASET" "$SSVM_LIBRARY_VERSION" "$EXP_VERSION"
      if [ "$__RESULT_MISSING__" -eq 0 ] ; then
        # The result already computed ...

        if [ "$k" -eq 0 ] ; then
          # If the SSVM model with index =0 is available, than we do not need to add any dependency to the subsequent jobs.
          __MODEL_ZERO_AVAILABLE__=1
        fi
        
        continue
      fi

      # Job-name for example: 2901_1_cfmid_2D
      JOB_NAME="${EVAL_SET_ID}_${k}_${MS2SCORER_SHORT}_${__MOL_FEAT_STEREO}"

      if [ "$k" -eq 0 ] ; then
        # Submit the job for the (dataset, sample-idx) tuple for the first SSVM model and get the job's ID
        JOB_ID=$( \
          sbatch --job-name="$JOB_NAME" --time="$TIME_BASE_MODEL" --mem-per-cpu="$MEM_PER_CPU" --cpus-per-task="$CPUS_PER_TASK" --nodes=1 ${NODE_FEATURES} ${ACCOUNT} \
            _run_massbank.sh \
              $N_THREADS "$MOL_KERNEL" "$EVAL_SET_ID" "$k" "$N_TRAIN_SEQS" "$N_EPOCHS" "$BATCH_SIZE" "$MS2SCORER" \
              "$LLOSS_MODE" "$C_GRID" "$LOAD_OPT_C" "$SSVM_FLAVOR" "$MOL_FEATURES" "$MOL_IDENTIFIER" "$MAX_N_CAND_TRAIN" \
              "$TRAINING_DATASET" "$EXP_VERSION" "$SSVM_LIBRARY_VERSION" "$CLUSTER_SYSTEM" | awk '{printf $4}' \
        )
      elif [ "$__MODEL_ZERO_AVAILABLE__" -eq 1 ] ; then
          sbatch --job-name="$JOB_NAME" --time="$TIME_OTHER_MODEL" --mem-per-cpu="$MEM_PER_CPU" --cpus-per-task="$CPUS_PER_TASK" --nodes=1 ${NODE_FEATURES} ${ACCOUNT} \
            _run_massbank.sh \
              $N_THREADS "$MOL_KERNEL" "$EVAL_SET_ID" "$k" "$N_TRAIN_SEQS" "$N_EPOCHS" "$BATCH_SIZE" "$MS2SCORER" \
              "$LLOSS_MODE" "$C_GRID" "$LOAD_OPT_C" "$SSVM_FLAVOR" "$MOL_FEATURES" "$MOL_IDENTIFIER" "$MAX_N_CAND_TRAIN" \
              "$TRAINING_DATASET" "$EXP_VERSION" "$SSVM_LIBRARY_VERSION" "$CLUSTER_SYSTEM"
      else
        # Submit the jobs for all other SSVM models (larger 0) and make them depending on the first SSVM model (= 0)
        sbatch --job-name="$JOB_NAME" --dependency=afterok:"$JOB_ID" --time="$TIME_OTHER_MODEL" --mem-per-cpu="$MEM_PER_CPU" --cpus-per-task="$CPUS_PER_TASK" --nodes=1 ${NODE_FEATURES} ${ACCOUNT} \
          _run_massbank.sh \
              $N_THREADS "$MOL_KERNEL" "$EVAL_SET_ID" "$k" "$N_TRAIN_SEQS" "$N_EPOCHS" "$BATCH_SIZE" "$MS2SCORER" \
              "$LLOSS_MODE" "$C_GRID" "$LOAD_OPT_C" "$SSVM_FLAVOR" "$MOL_FEATURES" "$MOL_IDENTIFIER" "$MAX_N_CAND_TRAIN" \
              "$TRAINING_DATASET" "$EXP_VERSION" "$SSVM_LIBRARY_VERSION" "$CLUSTER_SYSTEM"
      fi

      echo "Submitted: ${JOB_NAME}"
    done
  done
done
