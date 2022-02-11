#
# Script to load some comparison method specific parameters
#

if [ $# -ne 1 ] ; then
  echo "[ERROR] Script must be called with one (1) parameters."
  echo "USAGE: $(basename "$0") COMPARISON_METHOD"
  exit 1
fi

case $1 in
  "ruttkies2016_xlogp3")
    N_JOBS_SCORING_EVAL=-1  # NOT USED
    MOL_FEATURES="xlogp3"
    ;;
  "bach2020_ranksvm")
    N_JOBS_SCORING_EVAL=2  # Number of parallel jobs that are used during the scoring of the evaluation set
    MOL_FEATURES="substructure_count__smiles_iso"
    ;;
  "rt_prediction_svr")
    N_JOBS_SCORING_EVAL=-1  # NOT USED
    MOL_FEATURES="bouwmeester__smiles_iso"
    ;;
  *)
    echo "[ERROR] Invalid comparison method: $COMPARISON_METHOD."
    exit 1
esac
