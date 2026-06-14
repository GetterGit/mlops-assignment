#!/usr/bin/env bash
# scripts/bootstrap_vm.sh - one-shot VM bring-up after re-provisioning.
set -euo pipefail

# OS deps that bit us last time
sudo apt-get update
sudo apt-get install -y python3.12-dev build-essential

# uv + Python env
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/mlops-assignment
uv sync   # respects the transformers>=4.50,<5.0 pin you added

# o11y stack
docker compose up -d prometheus grafana langfuse langfuse-worker langfuse-db langfuse-clickhouse langfuse-redis

# BIRD data
uv run python scripts/load_data.py

# Now: scp ~/vm.env.backup back to ~/mlops-assignment/.env, then:
#   tmux new -s vllm 'bash scripts/start_vllm.sh 2>&1 | tee /tmp/vllm.log'
#   uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001