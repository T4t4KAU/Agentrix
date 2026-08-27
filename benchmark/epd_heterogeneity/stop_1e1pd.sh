#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTRIX_ROOT="${AGENTRIX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${AGENTRIX_ROOT}")}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/epd_experiments/runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/current}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "No current EPD run found at ${RUN_DIR}."
  exit 0
fi

for name in proxy pd encoder; do
  pid_file="${RUN_DIR}/${name}.pid"
  [[ -f "${pid_file}" ]] || continue
  pid=$(<"${pid_file}")
  if kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  fi
done

for _ in $(seq 1 30); do
  alive=0
  for name in proxy pd encoder; do
    pid_file="${RUN_DIR}/${name}.pid"
    [[ -f "${pid_file}" ]] || continue
    pid=$(<"${pid_file}")
    kill -0 "${pid}" 2>/dev/null && alive=1
  done
  [[ "${alive}" -eq 0 ]] && break
  sleep 1
done

if [[ -f "${RUN_DIR}/run.env" ]]; then
  cache_path=$(sed -n 's/^EC_SHARED_STORAGE_PATH=//p' "${RUN_DIR}/run.env")
  if [[ "${cache_path}" == /dev/shm/agentrix_epd_ec_* && -d "${cache_path}" ]]; then
    find "${cache_path}" -mindepth 1 -delete
    rmdir "${cache_path}"
  fi
fi

echo "Stopped EPD services from ${RUN_DIR}."
