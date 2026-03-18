#!/bin/bash
set -euo pipefail

docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  -v "$(pwd):/mnt/workspace" \
  -w "/mnt/workspace" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e WANDB_MODE=disabled \
  -e PYTHONPATH="/mnt/workspace:${PYTHONPATH:-}" \
  seokmin/franka:gpu_v2 \
  /bin/bash -c "
    echo 'Updating MuJoCo XML limits (njmax/nconmax)...';
    find /opt/robohive/ -name '*.xml' -type f -exec sed -i \"s/njmax=['\\\"][0-9]*['\\\"]/njmax='4000'/g\" {} +;
    find /opt/robohive/ -name '*.xml' -type f -exec sed -i \"s/nconmax=['\\\"][0-9]*['\\\"]/nconmax='2000'/g\" {} +;
    echo 'Update complete. Starting bash...';
    /bin/bash
  "