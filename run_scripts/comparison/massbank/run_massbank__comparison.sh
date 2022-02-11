#!/bin/bash

# Some parameters of the experiment
EXP_VERSION=3
SSVM_LIBRARY_VERSION=v2
USE_RAM_DISK=1

# Which comparison method should be run: "ruttkies2016_xlogp3", "bach2020_ranksvm" or "rt_prediction_svr"
COMPARISON_METHOD="ruttkies2016_xlogp3"

MS2SCORER_SHORT="cfmid4"
MOL_IDENTIFIER="cid"
TRAINING_DATASET="massbank"

# Resolve short parameter names
source ./resolve_short_parameter_names.sh $MS2SCORER_SHORT
if [ -z "$MS2SCORER" ] ; then
  echo "[ERROR] MS2SCORER is not defined"
  exit 1
else
  echo "${MS2SCORER_SHORT} --> ${MS2SCORER}"
fi

# Load cluster resource requirements depending on the comparison method
source ./load_cluster_requirements.sh $COMPARISON_METHOD
if [ -z "$CPUS_PER_TASK" ] || [ -z "$N_THREADS" ] || [ -z "$RUN_TIME" ] || [ -z "$MEM_PER_CPU" ] || [ -z "$PARTITION" ] ; then
  echo "[ERROR] CPUS_PER_TASK, N_THREADS, RUN_TIME, MEM_PER_CPU or PARTITION is not defined"
  exit 1
else
  echo "CPUS_PER_TASK: ${CPUS_PER_TASK}"
  echo "N_THREADS: ${N_THREADS}"
  echo "RUN_TIME: ${RUN_TIME}"
  echo "MEM_PER_CPU: ${MEM_PER_CPU}"
  echo "PARTITION: ${PARTITION}"
fi

# Load some experiment parameters depending on the experiment version
source ./load_experiment_parameters.sh "$EXP_VERSION"
if [ -z "$C_GRID" ] || [ -z "$BETA_GRID" ] || [ -z "$N_TUPLES_SEQUENCES_RANKING" ] || [ -z "$MAX_N_CAND_TRAIN" ] || [ -z "$N_TREES_FOR_SCORING" ] ; then
  echo "[ERROR] C_GRID, BETA_GRID, N_TUPLES_SEQUENCES_RANKING, MAX_N_CAND_TRAIN or N_TREES_FOR_SCORING is not defined."
  exit 1
else
  echo "C_GRID: ${C_GRID}"
  echo "BETA_GRID: ${BETA_GRID}"
  echo "N_TUPLES_SEQUENCES_RANKING: ${N_TUPLES_SEQUENCES_RANKING}"
  echo "MAX_N_CAND_TRAIN: ${MAX_N_CAND_TRAIN}"
  echo "N_TREES_FOR_SCORING: ${N_TREES_FOR_SCORING}"
  echo "MOL_IDENTIFIER: ${MOL_IDENTIFIER}"
fi

# Load some more parameters which depend on the comparison method
source ./load_comparison_method_specific_parameters.sh $COMPARISON_METHOD
if [ -z "$N_JOBS_SCORING_EVAL" ] || [ -z "$MOL_FEATURES" ] ; then
  echo "[ERROR] N_JOBS_SCORING_EVAL or MOL_FEATURES is not defined."
  exit 1
else
  echo "N_JOBS_SCORING_EVAL: ${N_JOBS_SCORING_EVAL}"
  echo "MOL_FEATURES: ${MOL_FEATURES}"
fi

# Depending on the training dataset load the number of samples per dataset
case $TRAINING_DATASET in
  "massbank")
    N_SAMPLES=(15 15 1 14 15 15 15 15 1 15 15 6 6 15 15 15 15 1 1 6 7 11 15 14 18 1 15 15 7 15 15 6)  # 32 datasets
    ;;
  *)
    echo "[ERROR] Invalid training dataset: ${TRAINING_DATASET}. Choices are 'massbank'"
    exit 1
esac

N_DATASETS=${#N_SAMPLES[@]}  # Number of datasets

# Sleep for a couple of seconds. This gives us the change to double check the specified parameters.
sleep 3s

# Start the jobs for all datasets
for (( i=0; i<N_DATASETS; i++ ))
do
  echo "Submit dataset '$i' with '${N_SAMPLES[$i]}' samples."

  # We skip the HILIC dataset
  if [ "$i" -eq 2 ] ; then
    continue
  fi

  for (( j=0; j<${N_SAMPLES[$i]}; j++))
  do
    EVAL_SET_ID=$(printf "%02d%02d" $i $j)
    echo "Evaluation set id: $EVAL_SET_ID"

    # Check whether the results for the current setup already exists and, if, go to the next setup.
    source ./check_whether_result_exists.sh "$EVAL_SET_ID" "$MS2SCORER" "$MOL_FEATURES" "$MOL_IDENTIFIER" "$EXP_VERSION" "$COMPARISON_METHOD"
    if [ "$__RESULT_MISSING__" -eq 0 ] ; then
      # The result already computed ...
      continue
    fi

    # Job-name for example (using cfmid): 2901_cf
    JOB_NAME="${EVAL_SET_ID}_${MS2SCORER_SHORT:0:2}_${COMPARISON_METHOD}"

    sbatch --job-name="$JOB_NAME" --time="$RUN_TIME" --mem-per-cpu="$MEM_PER_CPU" --cpus-per-task="$CPUS_PER_TASK" --partition="${PARTITION}" \
      ./_run_massbank.sh \
      "$N_THREADS" "$EVAL_SET_ID" "$MS2SCORER"  "$MOL_FEATURES" "$MOL_IDENTIFIER" "$C_GRID" "$BETA_GRID" \
      "$N_TREES_FOR_SCORING" "$MAX_N_CAND_TRAIN" "$N_TUPLES_SEQUENCES_RANKING" "$EXP_VERSION" "$SSVM_LIBRARY_VERSION" \
      "$USE_RAM_DISK" "$N_JOBS_SCORING_EVAL" "$COMPARISON_METHOD"
  done
done



# FIXME: This code is needed for the RankSVM approach
  # The CE_001 dataset makes some trouble here. It requires too much memory. So we load the MASSBANK DB to the
  # local disk rather the RAM. That also means we need to restrict us to nodes with "gres" feature.
  #  if [ "$i" -eq 17 ] ; then
  #    USE_RAM_DISK=1
  #    # MEM_PER_CPU=10550
  #    MEM_PER_CPU=9380
  #    # NODE_FEATURE="--gres=spindle"
  #    # NODE_FEATURE=""
  #    N_JOBS_SCORING_EVAL=1
  #  else
  #    USE_RAM_DISK=1
  #    MEM_PER_CPU=7800
  #    # NODE_FEATURE=""
  #    N_JOBS_SCORING_EVAL=2
  #  fi
  #
  #  if [ "$USE_RAM_DISK" -eq 0 ] ; then
  #    NODE_FEATURE="--gres=spindle"
  #  else
  #    NODE_FEATURE=""
  #  fi

  # From experience we know that some datasets require more memory. However, to not restrict the computation nodes
  # we only increase the memory for the respective datasets.
  #if [ "$i" -eq 17 ] || [ "$i" -eq 15 ] ; then
  #  # MEM_PER_CPU=7800  # makes 168Gb for 14 cores
  #  MEM_PER_CPU=9000
  #else
  #  MEM_PER_CPU=12500  # makes 119Gb for 14 cores
  #fi
