#!/usr/bin/env bash
# Train the runs behind the paper's results.
#
#   bash scripts/reproduce.sh main|ablations|physics_heads|chem_ood|seeds|all [train.py options]
#
# Runs are sequential; each takes several GPU hours.
set -euo pipefail
cd "$(dirname "$0")/.."

group="${1:?usage: reproduce.sh main|ablations|physics_heads|chem_ood|seeds|all [train.py options]}"
shift

train() { echo "== $*"; python scripts/train.py "$@"; }

run_group() {
  case "$1" in
    main)
      for c in configs/mixuni.yaml configs/single_task/[!_]*.yaml; do train "$c" "${@:2}"; done ;;
    ablations)
      for c in configs/ablations/*.yaml; do train "$c" "${@:2}"; done ;;
    physics_heads)
      for c in configs/physics_heads/*.yaml; do train "$c" "${@:2}"; done ;;
    chem_ood)
      for c in configs/chem_ood/mixuni.yaml configs/chem_ood/single_task/*.yaml; do train "$c" "${@:2}"; done ;;
    seeds)
      for s in 41 42 43 44 45; do
        for c in configs/mixuni.yaml configs/chem_ood/mixuni.yaml \
                 configs/chem_ood/single_task/il_conductivity.yaml configs/chem_ood/single_task/pe_conductivity.yaml; do
          train "$c" --seed "$s" "${@:2}"
        done
      done ;;
    *) echo "unknown group: $1" >&2; exit 1 ;;
  esac
}

if [[ "$group" == all ]]; then
  for g in main ablations physics_heads chem_ood seeds; do run_group "$g" "$@"; done
else
  run_group "$group" "$@"
fi
