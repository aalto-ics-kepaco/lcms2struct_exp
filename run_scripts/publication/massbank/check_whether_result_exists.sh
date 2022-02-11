#
# Script that checks whether the results for the experiment specified by the parameters already exists.
#

if [ $# -ne 10 ] ; then
  echo "[ERROR] Script must be called with ten (10) parameters."
  echo "USAGE: $(basename "$0") EVAL_SET_ID SSVM_MODEL_IDX MS2SCORER LLOSS_MODE MOL_FEATURES SSVM_FLAVOR MOL_IDENTIFIER TRAINING_DATASET SSVM_LIBRARY_VERSION EXP_VERSION"
  exit 1
fi

# We hard-code some paths and parameters
__PROJECT_DIR="${SCRATCHHOME}/projects/lcms2struct_experiments"
__DB_FN="${__PROJECT_DIR}/data/massbank.sqlite"

# Get the output directory
__ODIR="${__PROJECT_DIR}/results_raw/publication/massbank/ssvm_lib=${9}__exp_ver=${10}"

__SCRIPTPATH="${__PROJECT_DIR}/run_scripts/publication/massbank/check_whether_result_exists.py"

python $__SCRIPTPATH "$1" "$2" \
  --db_fn="$__DB_FN" \
  --ms2scorer="$3" \
  --lloss_fps_mode="$4" \
  --mol_feat_retention_order="$5" \
  --ssvm_flavor="$6" \
  --molecule_identifier="$7" \
  --output_dir="$__ODIR" \
  --training_dataset="$8" \
  --verbose=0

# Store the exit code of the python script
__RESULT_MISSING__=$?
