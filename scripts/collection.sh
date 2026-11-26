#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-check}"
CONTROL="$PROJECT/outputs/five_task_speed_batch2_control"
if [[ -n "${COLLECTION_PYTHON:-}" ]]; then PYTHON="$COLLECTION_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/mujoco/bin/python" ]]; then PYTHON="$HOME/miniconda3/envs/mujoco/bin/python"
elif [[ -x "$HOME/anaconda3/envs/mujoco/bin/python" ]]; then PYTHON="$HOME/anaconda3/envs/mujoco/bin/python"
else PYTHON="$(command -v python3)"; fi
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
if [[ "$MODE" == deploy ]]; then
  [[ $# == 2 ]] || { echo 'Usage: bash scripts/collection.sh deploy /actual/mount/dex_hand_dataset_batch2' >&2; exit 2; }
  exec "$PYTHON" apps/deploy_collection_update.py --project "$PROJECT" --dataset-root "$2"
fi
[[ -f "$CONTROL/collection_deployment.json" ]] || { echo 'Run deploy first; original server queue must exist.' >&2; exit 78; }
DATASET="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_root"])' "$CONTROL/collection_deployment.json")"
if [[ "$MODE" == install-service ]]; then
  /bin/bash "$PROJECT/scripts/collection.sh" check
  UNIT="$HOME/.config/systemd/user/dex-collection.service"
  mkdir -p "$(dirname "$UNIT")"
  [[ ! -f "$UNIT" ]] || cp -- "$UNIT" "$UNIT.before_$(date +%Y%m%d_%H%M%S)"
  # Standard server checkout path; reject systemd expansion/control characters.
  [[ "$PROJECT" != *'%'* && "$PROJECT" != *'"'* && "$PYTHON" != *'%'* && "$PYTHON" != *'"'* ]] || exit 2
  cat > "$UNIT" <<EOF
[Unit]
Description=Dex collection with bounded automatic recovery
StartLimitIntervalSec=3600
StartLimitBurst=6
[Service]
Type=simple
Environment="COLLECTION_PYTHON=$PYTHON"
ExecStart=/bin/bash "$PROJECT/scripts/collection.sh" run
Restart=on-failure
RestartSec=60
RestartPreventExitStatus=78 130
KillMode=control-group
TimeoutStopSec=150
[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now dex-collection.service
  exec systemctl --user status dex-collection.service --no-pager
fi
exec "$PYTHON" apps/collection_autorecover.py "$MODE" --project "$PROJECT" --dataset-root "$DATASET"
