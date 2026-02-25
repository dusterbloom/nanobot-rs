#!/usr/bin/env bash
set -euo pipefail

# Find local MLX model directories.
# A directory qualifies when it contains config.json and at least one
# *.safetensors or *.mlx weight file.

mode="list"
if [[ "${1:-}" == "--first" ]]; then
  mode="first"
fi

roots=(
  "$HOME/models"
  "$HOME/.cache/lm-studio/models"
  "$HOME/Library/Application Support/LM Studio/models"
)

declare -A seen=()
models=()

for root in "${roots[@]}"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r dir; do
    [[ -f "$dir/config.json" ]] || continue
    if find "$dir" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.mlx' \) | grep -q .; then
      canonical="$(cd "$dir" && pwd -P)"
      if [[ -z "${seen[$canonical]:-}" ]]; then
        seen[$canonical]=1
        models+=("$canonical")
      fi
    fi
  done < <(find "$root" -type d 2>/dev/null)
done

if [[ "${#models[@]}" -eq 0 ]]; then
  if [[ "$mode" == "first" ]]; then
    exit 1
  fi
  echo "No MLX model directories found."
  echo "Checked:"
  for root in "${roots[@]}"; do
    echo "  - $root"
  done
  exit 0
fi

IFS=$'\n' models=($(printf '%s\n' "${models[@]}" | sort))
unset IFS

if [[ "$mode" == "first" ]]; then
  printf '%s\n' "${models[0]}"
  exit 0
fi

echo "Found ${#models[@]} MLX model director$( [[ "${#models[@]}" -eq 1 ]] && echo "y" || echo "ies" ):"
for model in "${models[@]}"; do
  echo "  - $model"
done
