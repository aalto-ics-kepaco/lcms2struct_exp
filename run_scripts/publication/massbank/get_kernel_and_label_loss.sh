#
# Set up the feature kernel and label-loss.
#
# OUTPUT (variables)
# - "MOL_KERNEL": appropriate kernel for the molecule features
# - "LLOSS_MODE": label-loss setup

if [ $# -ne 1 ] ; then
  echo "[ERROR] Script must be called with one (1) parameters."
  echo "USAGE: $(basename "$0") MOL_FEATURES"
  exit 1
fi

case $1 in
  "FCFP__binary__all__3D" | "FCFP__binary__all__2D")
    MOL_KERNEL="tanimoto"
    LLOSS_MODE="mol_feat_fps"
    ;;
  "estate_cnt")
    MOL_KERNEL="minmax"
    LLOSS_MODE="mol_feat_fps"
    ;;
  "estate_idc")
    MOL_KERNEL="generalized_tanimoto"
    LLOSS_MODE="mol_feat_fps"
    ;;
  "bouwmeester__smiles_can")
    MOL_KERNEL="rbf__median"
    LLOSS_MODE="kernel_loss"
    ;;
esac
