#
# Script loading the cluster resource requirements depending on the comparison method to run
#

if [ $# -ne 1 ] ; then
  echo "[ERROR] Script must be called with one (1) parameters."
  echo "USAGE: $(basename "$0") COMPARISON_METHOD"
  exit 1
fi

case $1 in
  "ruttkies2016_xlogp3")
    CPUS_PER_TASK=8
    N_THREADS=1
    RUN_TIME="00:10:00"
    MEM_PER_CPU=5250
    PARTITION="hugemem"
    ;;
  "bach2020_ranksvm")
    CPUS_PER_TASK=6
    N_THREADS=3
    RUN_TIME="08:00:00"
    MEM_PER_CPU=16000
    PARTITION="hugemem"
    ;;
  "rt_prediction_svr")
    CPUS_PER_TASK=8
    N_THREADS=4
    RUN_TIME="00:10:00"
    MEM_PER_CPU=5250
    PARTITION="hugemem"
    ;;
  *)
    echo "[ERROR] Invalid comparison method: $COMPARISON_METHOD."
    exit 1
esac
