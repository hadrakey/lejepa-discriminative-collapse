#!/usr/bin/env bash
# Every experiment of the post, in order. Run from the repository root:
#
#   DATA=/path/to/dataTE_rieth GPU=0 bash scripts/reproduce.sh
#
# DATA must hold train/ and test/ as written by `ldc convert`. Runs are
# skipped when their last checkpoint already exists, so the script resumes.
# One 50-epoch run takes about 25 minutes on an A100; the whole script runs
# ~30 trainings. Set SEEDS="42" to skip the seed check.
set -euo pipefail
: "${DATA:?set DATA to the folder with train/ and test/}"
GPU="${GPU:-0}"
RUNS="${RUNS:-runs}"
SEEDS="${SEEDS:-42 43 44}"
DEV="cuda:$GPU"
mkdir -p "$RUNS/results"

ssl() {  # <run dir> <ldc train-ssl args...>
  local out="$1"; shift
  [[ -f "$out/epoch_50.pth" ]] || ldc train-ssl --data-root "$DATA" --device "$DEV" --out "$out" "$@"
}
diagnose() {  # <run dir> <init>
  [[ -f "$1/diagnostics.json" ]] || ldc diagnose --data-root "$DATA" --device "$DEV" --run "$1" --init "$2"
}

# --- 1. supervised reference encoders (one per seed) ---------------------------
for s in $SEEDS; do
  [[ -f "$RUNS/s$s/sup/encoder_init.pth" ]] || ldc train-sup --data-root "$DATA" --device "$DEV" \
    --seed "$s" --out "$RUNS/s$s/sup"
done
INIT42="$RUNS/s42/sup/encoder_init.pth"

# --- the paradox + 4. masks: LeJEPA from scratch, with a shared core of sensors -
for share in 0 25 50 75; do
  ssl "$RUNS/scratch/share$share" --shared-core "$(python -c "print($share/100)")"
  diagnose "$RUNS/scratch/share$share" "$INIT42"
done
[[ -f "$RUNS/results/mask_audit.json" ]] || ldc mask-audit --data-root "$DATA" --out "$RUNS/results/mask_audit.json"

# --- 5-6. continuation from the supervised encoder: which loss term? ------------
for s in $SEEDS; do
  init="$RUNS/s$s/sup/encoder_init.pth"
  ssl "$RUNS/s$s/inv_only" --init "$init" --seed "$s" --alpha 1 --lam 0
  ssl "$RUNS/s$s/both"     --init "$init" --seed "$s" --alpha 1 --lam 0.05
  for run in inv_only both; do diagnose "$RUNS/s$s/$run" "$init"; done
done
ssl "$RUNS/s42/sig_only" --init "$INIT42" --alpha 0 --lam 0.05
diagnose "$RUNS/s42/sig_only" "$INIT42"

# --- 7. class-conditional SIGReg gradient ---------------------------------------
for t in "iso isotropic 1" "power1 power 1" "power2 power 2" "linear linear 1"; do
  set -- $t
  [[ -f "$RUNS/results/grad_$1.json" ]] || ldc grad --data-root "$DATA" --device "$DEV" --init "$INIT42" \
    --sigreg-target "$2" --sigreg-power "$3" --out "$RUNS/results/grad_$1.json"
done

# --- 8. anisotropic SIGReg targets ----------------------------------------------
ssl "$RUNS/s42/sig_power1" --init "$INIT42" --alpha 0 --lam 0.05 --sigreg-target power --sigreg-power 1
ssl "$RUNS/s42/sig_power2" --init "$INIT42" --alpha 0 --lam 0.05 --sigreg-target power --sigreg-power 2
ssl "$RUNS/s42/sig_linear" --init "$INIT42" --alpha 0 --lam 0.05 --sigreg-target linear
for run in sig_power1 sig_power2 sig_linear; do diagnose "$RUNS/s42/$run" "$INIT42"; done

# --- 9. gradient paths, affine control, projector dimension ---------------------
for s in $SEEDS; do
  init="$RUNS/s$s/sup/encoder_init.pth"
  ssl "$RUNS/s$s/detach_h" --init "$init" --seed "$s" --path detach_h
  diagnose "$RUNS/s$s/detach_h" "$init"
  [[ -f "$RUNS/results/affine_s$s.json" ]] || ldc affine --data-root "$DATA" --device "$DEV" --init "$init" \
    --seed "$s" --out "$RUNS/results/affine_s$s.json"
done
ssl "$RUNS/s42/freeze_enc" --init "$INIT42" --path freeze_enc
ssl "$RUNS/s42/on_h"       --init "$INIT42" --path on_h
for k in 2 4 8; do ssl "$RUNS/s42/k$k" --init "$INIT42" --proj-dim "$k"; done
for run in freeze_enc on_h k2 k4 k8; do diagnose "$RUNS/s42/$run" "$INIT42"; done

# --- tables and figures -----------------------------------------------------------
ldc report --runs "$RUNS" --out results
