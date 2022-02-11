#
# Parameter setup for the different experiment versions.
#
# OUTPUT (variables)
# - "C_GRID": Grid for the SSVM regularization parameter C
# - "BETA_GRID": Grid for the MS2-score weight
# - "N_TUPLES_SEQUENCES_RANKING": Number of (MS2, RT)-tuple sequences used to determine the optimal beta-value
# - "MAX_N_CAND_TRAIN": Maximum number of random candidates during the optimization of beta
# - "N_TREES_FOR_SCORING": umber of random spanning trees used for the max-margin computation

if [ $# -ne 1 ] ; then
  echo "[ERROR] Script must be called with one (1) parameter."
  echo "USAGE: $(basename "$0") EXP_VERSION"
  exit 1
fi

if [ "$1" = 1 ] ; then
  #
  # EXPERIMENT VERSION 1
  #

  echo "[ERROR] The first experiment version is not valid anymore !!!"
  exit 1
elif [ "$1" = 2 ] ; then
  #
  # EXPERIMENT VERSION 2
  #

  # Regularization parameter for the RankSVM
  C_GRID="0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32"

  # Grid for the MS2-score weight
  BETA_GRID="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

  # Number of (MS2, RT)-tuple sequences used to determine the optimal beta-value
  N_TUPLES_SEQUENCES_RANKING=25

  # Maximum number of random candidates during the optimization of beta
  MAX_N_CAND_TRAIN=300

  # Number of random spanning trees used for the max-margin computation (as in Bach et al. 2020)
  N_TREES_FOR_SCORING=128
elif [ "$1" = 3 ] ; then
  #
  # EXPERIMENT VERSION 3
  #

  # Regularization parameter for the RankSVM and the SVR
  C_GRID="0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32"

  # Grid for the MS2-score weight
  BETA_GRID="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0"

  # Number of (MS2, RT)-tuple sequences used to determine the optimal beta-value
  N_TUPLES_SEQUENCES_RANKING=25

  # Maximum number of random candidates during the optimization of beta
  MAX_N_CAND_TRAIN=500

  # Number of random spanning trees used for the max-margin computation (as in Bach et al. 2020)
  N_TREES_FOR_SCORING=128
else
  echo "[ERROR] Invalid experiment version: $1. Choices are '1', '2' and '3'."
  exit 1
fi
