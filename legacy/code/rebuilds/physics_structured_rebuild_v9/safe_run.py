#!/usr/bin/env python3
"""Run one child process with conservative laptop resource stop limits."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minimum-available-mib", type=float, default=2048.0)
    parser.add_argument("--maximum-gpu-memory-mib", type=float, default=8192.0)
    parser.add_argument("--maximum-gpu-temperature-c", type=float, default=80.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    return args


def memory_status() -> dict[str, float]:
    values: dict[str, float] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            name, rest = line.split(":", 1)
            if name in {"MemAvailable", "SwapFree", "SwapTotal"}:
                values[name] = float(rest.split()[0]) / 1024.0
    return {
        "available_mib": values["MemAvailable"],
        "swap_used_mib": values["SwapTotal"] - values["SwapFree"],
    }


def gpu_status() -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = [
        float(value.strip())
        for value in completed.stdout.strip().split(",")
    ]
    return {
        "temperature_c": values[0],
        "memory_used_mib": values[1],
        "utilization_percent": values[2],
    }


def stop_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main() -> None:
    args = parse_args()
    started = time.time()
    process = subprocess.Popen(
        args.command,
        start_new_session=True,
    )
    minimum_available = float("inf")
    maximum_swap = 0.0
    maximum_gpu_memory = 0.0
    maximum_gpu_temperature = 0.0
    samples = 0
    stop_reason = None
    next_report = started + 60.0
    try:
        while process.poll() is None:
            memory = memory_status()
            gpu = gpu_status()
            samples += 1
            minimum_available = min(
                minimum_available,
                memory["available_mib"],
            )
            maximum_swap = max(maximum_swap, memory["swap_used_mib"])
            maximum_gpu_memory = max(
                maximum_gpu_memory,
                gpu["memory_used_mib"],
            )
            maximum_gpu_temperature = max(
                maximum_gpu_temperature,
                gpu["temperature_c"],
            )
            if memory["available_mib"] < float(args.minimum_available_mib):
                stop_reason = "available system memory below limit"
            elif gpu["memory_used_mib"] > float(
                args.maximum_gpu_memory_mib
            ):
                stop_reason = "GPU memory above limit"
            elif gpu["temperature_c"] >= float(
                args.maximum_gpu_temperature_c
            ):
                stop_reason = "GPU temperature at or above limit"
            if stop_reason is not None:
                print(
                    f"SAFE STOP: {stop_reason}",
                    file=sys.stderr,
                    flush=True,
                )
                stop_child(process)
                break
            now = time.time()
            if now >= next_report:
                print(
                    json.dumps(
                        {
                            "safe_run": "healthy",
                            "elapsed_seconds": now - started,
                            "memory": memory,
                            "gpu": gpu,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                next_report = now + 60.0
            time.sleep(float(args.poll_seconds))
    except BaseException:
        stop_child(process)
        raise
    return_code = process.wait()
    audit = {
        "version": "physics_structured_rebuild_v9_safe_run",
        "command": args.command,
        "return_code": return_code,
        "safe_stop_reason": stop_reason,
        "samples": samples,
        "seconds": time.time() - started,
        "minimum_available_mib": minimum_available,
        "maximum_swap_used_mib": maximum_swap,
        "maximum_gpu_memory_used_mib": maximum_gpu_memory,
        "maximum_gpu_temperature_c": maximum_gpu_temperature,
        "limits": {
            "minimum_available_mib": float(args.minimum_available_mib),
            "maximum_gpu_memory_mib": float(args.maximum_gpu_memory_mib),
            "maximum_gpu_temperature_c": float(
                args.maximum_gpu_temperature_c
            ),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True), file=sys.stderr, flush=True)
    if stop_reason is not None:
        raise SystemExit(75)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
