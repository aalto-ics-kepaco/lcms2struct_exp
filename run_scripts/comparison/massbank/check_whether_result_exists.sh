#
# Script that checks whether the results for the experiment specified by the parameters already exists.
#

if [ $# -ne 6 ] ; then
  echo "[ERROR] Script must be called with six (6) parameters."
  echo "USAGE: $(basename "$0") EVAL_SET_ID MS2SCORER MOL_FEATURES MOL_IDENTIFIER EXP_VERSION COMPARISON_METHOD"
  exit 1
fi

# Which predictor is used for the different comparison methods
case $6 in
  "ruttkies2016_xlogp3")
    PREDICTOR="linear_reg"
    SCORE_INTEGRATION_APPROACH="score_combination"
    ;;
  "bach2020_ranksvm")
    PREDICTOR="ranksvm"
    SCORE_INTEGRATION_APPROACH="msms_pl_rt_score_integration"
    ;;
  "rt_prediction_svr")
    PREDICTOR="svr"
    SCORE_INTEGRATION_APPROACH="filtering__global"
    ;;
  *)
    echo "[ERROR] Invalid comparison method: $COMPARISON_METHOD."
    exit 1
esac

# We hard-code some paths and parameters
__PROJECT_DIR="/scratch/cs/kepaco/bache1/projects/lcms2struct_experiments"
__DB_FN="${__PROJECT_DIR}/data/massbank.sqlite"

# Get the output directory
__ODIR="${__PROJECT_DIR}/results_processed/comparison/massbank__exp_ver=${5}"

__SCRIPTPATH="${__PROJECT_DIR}/run_scripts/comparison/massbank/check_whether_result_exists.py"

python $__SCRIPTPATH "$1" \
  --db_fn="$__DB_FN" \
  --ms2scorer="$2" \
  --score_integration_approach="$SCORE_INTEGRATION_APPROACH" \
  --predictor="$PREDICTOR" \
  --molecule_features="$3" \
  --molecule_identifier="$4" \
  --output_dir="$__ODIR" \
  --verbose=0

# Store the exit code of the python script
__RESULT_MISSING__=$?
