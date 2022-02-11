#
# Parameter setup for the different experiment versions.
#
# OUTPUT (variables)
# - "C_GRID": Grid for the SSVM regularization parameter C
# - "BATCH_SIZE": Batch-size for the stochastic gradient descent
# - "LOAD_OPT_C": Binary indicating whether the optimal C parameter should be loaded from model with index =0
# - "N_EPOCHS": Number of training epochs for the SSVM model
# - "N_TRAIN_SEQS": Number of sequences used for the SSVM model training
# - "MAX_N_CAND_TRAIN": Maximum number of random candidates per spectrum during training
# - "MOL_IDENTIFIER": (only for experiment version 3) Identifier used to distinguish the candidate molecules

if [ $# -ne 2 ] ; then
  echo "[ERROR] Script must be called with two (2) parameters."
  echo "USAGE: $(basename "$0") EXP_VERSION MS2SCORER"
  exit 1
fi

if [ "$1" = 1 ] ; then
  #
  # EXPERIMENT VERSION 1
  #

  C_GRID="0.125,2048,1,512,8,128,32,1024"
  BATCH_SIZE=64
  LOAD_OPT_C=0  # C is optimized for each SSVM model separately
  N_EPOCHS=3
  N_TRAIN_SEQS=768
  MAX_N_CAND_TRAIN=50
elif [ "$1" = 2 ] ; then
  #
  # EXPERIMENT VERSION 2
  #

  # - Optimal C parameter is now determined only for SSVM model with index =0
  # - Reduction of the batch-size from 64 to 32
  # - C-grid is specific for each MS2-scoring method

  case $2 in
    "metfrag__norm" | "cfm-id__summed_up_sim__norm")
      C_GRID="4,32,96,128,256,512,1024"
      ;;
    "sirius__sd__correct_mf__norm")
      C_GRID="0.0625,0.25,1,4,32,128,256"
      ;;
    *)
      echo "[ERROR] Invalid long MS2-scorer ID: $MS2SCORER."
      exit 1
  esac
  BATCH_SIZE=32
  # The optimal parameters are loaded from SSVM model (tree) with index =0 for indices >0
  LOAD_OPT_C=1
  N_EPOCHS=3
  N_TRAIN_SEQS=768
  MAX_N_CAND_TRAIN=50
elif [ "$1" = 3 ] ; then
  #
  # EXPERIMENT VERSION 3
  #

  # - maximum number of candidates increased to 75 (as no grouping is done by inchikey during training)
  # - C-grid for SIRIUS has been altered
  # - molecule identifier now always used "cid"

  case $2 in
    "metfrag__norm" | "cfm-id__summed_up_sim__norm")
      C_GRID="4,32,96,128,256,512,1024"
      ;;
    "sirius__sd__correct_mf__norm")
      C_GRID="0.25,1,4,32,128,192,256"
      ;;
    *)
      echo "[ERROR] Invalid long MS2-scorer ID: $2."
      exit 1
  esac
  BATCH_SIZE=32
  # The optimal parameters are loaded from SSVM model (tree) with index =0 for indices >0
  LOAD_OPT_C=1
  N_EPOCHS=3
  N_TRAIN_SEQS=768
  MAX_N_CAND_TRAIN=75
  MOL_IDENTIFIER="cid"
elif [ "$1" = 4 ] ; then
  #
  # EXPERIMENT VERSION 4
  #

  # - increase the number of epochs to 4

  case $2 in
    "metfrag__norm" | "cfm-id__summed_up_sim__norm" | "cfmid2__norm" | "cfmid4__norm")
      C_GRID="4,32,96,128,256,512,1024"
      ;;
    "sirius__sd__correct_mf__norm" | "sirius__norm")
      C_GRID="0.25,1,4,32,128,192,256"
      ;;
    *)
      echo "[ERROR] Invalid long MS2-scorer ID: $2."
      exit 1
  esac
  BATCH_SIZE=32
  # The optimal parameters are loaded from SSVM model (tree) with index =0 for indices >0
  LOAD_OPT_C=1
  N_EPOCHS=4
  N_TRAIN_SEQS=768
  MAX_N_CAND_TRAIN=75
  MOL_IDENTIFIER="cid"
else
  echo "[ERROR] Invalid experiment version: $1. Choices are '1', '2', '3' and '4'."
  exit 1
fi
