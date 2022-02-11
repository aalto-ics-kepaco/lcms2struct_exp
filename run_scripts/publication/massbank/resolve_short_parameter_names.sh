#
# Script translating short parameter names into long ones.
#
# E.g. "metfrag" --> "metfrag__norm"
#
# OUTPUT (variables)
# - "MS2SCORER": long name of the MS2-scorer as used in the SQLite DB
# - "MOL_FEATURES": long name of the molecule features as used in the SQLite DB

if [ $# -ne 2 ] ; then
  echo "[ERROR] Script must be called with two (2) parameters."
  echo "USAGE: $(basename "$0") MS2SCORER_SHORT MOL_FEATURE_SHORT"
  exit 1
fi

# MS2-scorers
case $1 in
  "metfrag")
    MS2SCORER="metfrag__norm"
    ;;
  "sirius")
    MS2SCORER="sirius__norm"
    ;;
  "cfmid2")
    MS2SCORER="cfmid2__norm"
    ;;
  "cfmid4")
    MS2SCORER="cfmid4__norm"
    ;;
  *)
    echo "[ERROR] Invalid short MS2-scorer ID: $1. Choices are 'metfrag', 'sirius', 'cfmid2' and 'cfmid4'."
    exit 1
    ;;
esac

# Molecule features.
case $2 in
  "fcfp_3D")
    MOL_FEATURES="FCFP__binary__all__3D"
    ;;
  "fcfp_2D")
    MOL_FEATURES="FCFP__binary__all__2D"
    ;;
  "bouwmeester")
    MOL_FEATURES="bouwmeester__smiles_iso"
    ;;
  *)
    echo "Invalid molecule feature representation: $2. Choices are 'fcfp_2D', 'fcfp_3D' and 'bouwmeester'."
    exit 1
    ;;
esac
