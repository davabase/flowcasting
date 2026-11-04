#!/usr/bin/env bash
# Keep training alive on a shared GPU: if another process's memory spike kills a step with a CUDA
# OOM, wait a minute and resume from the latest checkpoint (saved every --ckpt-every steps).
# Stop it with:  taskkill /T /F /PID $(cat logs/supervisor.pid)   (kills the loop and its workers)
mkdir -p logs
cat /proc/$$/winpid > logs/supervisor.pid 2>/dev/null || echo $$ > logs/supervisor.pid
for attempt in $(seq 1 40); do
  python -u train.py --resume "$@" && exit 0
  echo "[supervisor] training exited (attempt $attempt); resuming in 60 s" >&2
  sleep 60
done
exit 1
